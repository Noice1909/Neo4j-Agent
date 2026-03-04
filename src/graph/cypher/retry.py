"""
Cypher retry engine with LLM self-correction (Strategy #1).
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from langchain_neo4j import GraphCypherQAChain
from langchain_core.language_models import BaseChatModel
from neo4j.exceptions import ServiceUnavailable, SessionExpired
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.graph.cypher.callback import CypherSafetyCallback
from src.graph.cypher.prompts import UNIVERSAL_CYPHER_RULES, _FALLBACK_PROMPT
from src.graph.cypher.safety import validate_read_only
from src.graph.cypher.validation import pre_validate_cypher
from src.core.exceptions import ReadOnlyViolationError
from src.core.tracing import trace_event

logger = logging.getLogger(__name__)

MAX_CYPHER_RETRIES = 2


def build_correction_prompt(
    question: str,
    schema: str,
    failed_cypher: str,
    error: str,
    topology_section: str = "",
) -> str:
    """Build a prompt asking the LLM to fix a failed Cypher query."""
    parts = [
        "The following Cypher query was generated to answer a question but "
        "failed when executed against Neo4j.  Fix the query and return ONLY "
        "the corrected Cypher — no explanations, no markdown fences.",
        f"\nDatabase schema:\n{schema}",
    ]
    if topology_section:
        parts.append(f"\n{topology_section}")
    parts += [
        f"\nRules:\n{UNIVERSAL_CYPHER_RULES}",
        f"\nOriginal question: {question}",
        f"\nFailed Cypher:\n{failed_cypher}",
        f"\nError message:\n{error}",
        "\nCorrected Cypher:",
    ]
    return "\n".join(parts)


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences that LLMs sometimes wrap around Cypher."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def is_cypher_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a Cypher generation / execution error.

    Neo4j driver transient errors (``ServiceUnavailable``, ``SessionExpired``)
    are NOT Cypher errors — they should be handled by the tenacity retry layer,
    not the LLM self-correction loop.
    """
    # Transient connectivity errors → let tenacity handle them
    if isinstance(exc, (ServiceUnavailable, SessionExpired)):
        return False
    msg = str(exc).lower()
    indicators = [
        "syntax", "cypher", "invalid input", "unexpected",
        "expected", "unresolved", "unknown function", "type mismatch",
        "variable", "not defined", "pre-validation",
    ]
    return any(ind in msg for ind in indicators)


def extract_cypher_from_error(error_msg: str) -> str | None:
    """Best-effort extraction of the failed Cypher from a Neo4j error."""
    for pat in [
        r"Generated Cypher:\n(.+?)(?:\n\n|\nFull)",
        r"(?:Invalid input|Syntax error).*?'(.+?)'",
    ]:
        m = re.search(pat, error_msg, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


async def _run_initial_chain(
    question: str,
    llm: BaseChatModel,
    graph: Any,
    safety_cb: CypherSafetyCallback,
    cypher_prompt: Any = None,
) -> str:
    """Run the initial Cypher chain with tenacity retry for transient errors."""
    loop = asyncio.get_running_loop()

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=logger.isEnabledFor(logging.DEBUG),
        return_intermediate_steps=True,
        allow_dangerous_requests=True,
        cypher_prompt=cypher_prompt or _FALLBACK_PROMPT,
    )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type((
            ConnectionError, TimeoutError, OSError,
            ServiceUnavailable, SessionExpired,
        )),
        reraise=True,
    )
    def _invoke(q: str) -> dict:
        return chain.invoke(
            {"query": q}, config={"callbacks": [safety_cb]},
        )

    result = await loop.run_in_executor(None, lambda: _invoke(question))

    for step in result.get("intermediate_steps", []):
        if isinstance(step, dict) and "query" in step:
            validate_read_only(step["query"])

    answer: str = result.get("result", "I could not determine an answer.")
    logger.info("Graph query answered: %r → %r", question[:60], answer[:120])
    trace_event("CYPHER_EXECUTED", "ok", f"Attempt 0 succeeded: {answer[:80]}")
    return answer


async def _run_correction_attempt(
    question: str,
    llm: BaseChatModel,
    graph: Any,
    schema: str,
    last_cypher: str | None,
    last_error: str | None,
    attempt: int,
    topology_section: str = "",
) -> str:
    """Run a single LLM self-correction attempt and return the answer."""
    loop = asyncio.get_running_loop()
    correction = build_correction_prompt(
        question=question,
        schema=schema,
        failed_cypher=last_cypher or "N/A",
        error=last_error or "unknown error",
        topology_section=topology_section,
    )

    resp = await loop.run_in_executor(
        None, lambda cp=correction: llm.invoke(cp),
    )
    corrected_cypher = strip_code_fences(str(resp.content))
    logger.info("Retry %d corrected Cypher: %s", attempt, corrected_cypher[:200])
    trace_event("CYPHER_CORRECTION", "info", f"Retry {attempt}: {corrected_cypher}")

    validate_read_only(corrected_cypher)
    issues = pre_validate_cypher(corrected_cypher)
    if issues:
        raise ValueError(f"Pre-validation: {', '.join(issues)}")

    raw = await loop.run_in_executor(
        None, lambda cc=corrected_cypher: graph.query(cc),
    )

    qa_prompt = (
        f"Question: {question}\n"
        f"Database results: {raw}\n"
        "Provide a concise, natural-language answer based on these results."
    )
    qa_resp = await loop.run_in_executor(
        None, lambda qp=qa_prompt: llm.invoke(qp),
    )
    answer = str(qa_resp.content).strip()
    logger.info(
        "Graph query answered (retry %d): %r → %r",
        attempt, question[:60], answer[:120],
    )
    trace_event("CYPHER_EXECUTED", "ok", f"Retry {attempt} succeeded: {answer[:80]}")
    return answer


async def execute_with_retries(
    question: str,
    llm: BaseChatModel,
    graph: Any,
    schema: str,
    cypher_prompt: Any = None,
    topology_section: str = "",
) -> str:
    """Run the Cypher chain, retrying with LLM self-correction on failure."""
    safety_cb = CypherSafetyCallback()

    last_error: str | None = None
    last_cypher: str | None = None

    try:
        return await _run_initial_chain(question, llm, graph, safety_cb, cypher_prompt)
    except ReadOnlyViolationError:
        raise
    except Exception as exc:
        if not is_cypher_error(exc):
            raise
        last_error = str(exc)
        last_cypher = (
            safety_cb.last_generated_cypher
            or extract_cypher_from_error(last_error)
        )
        logger.warning("Cypher attempt 0 failed (retryable): %s", last_error[:200])
        trace_event("CYPHER_ATTEMPT_0", "fail", last_error[:120])

    for attempt in range(1, MAX_CYPHER_RETRIES + 1):
        try:
            return await _run_correction_attempt(
                question, llm, graph, schema,
                last_cypher, last_error, attempt,
                topology_section=topology_section,
            )
        except ReadOnlyViolationError:
            raise
        except ValueError as exc:
            # Pre-validation failure — update error and continue
            last_error = str(exc)
            logger.warning("Corrected Cypher still invalid: %s", last_error)
            trace_event("CYPHER_CORRECTION", "fail", f"Retry {attempt} invalid: {last_error[:80]}")
        except Exception as exc:
            last_error = str(exc)
            logger.warning("Cypher retry %d failed: %s", attempt, last_error[:200])
            trace_event("CYPHER_CORRECTION", "fail", f"Retry {attempt} error: {last_error[:80]}")

    trace_event("CYPHER_ALL_FAILED", "fail", f"All {MAX_CYPHER_RETRIES + 1} attempts failed")
    raise RuntimeError(
        f"All {MAX_CYPHER_RETRIES + 1} Cypher attempts failed.  "
        f"Last error: {last_error}"
    )

