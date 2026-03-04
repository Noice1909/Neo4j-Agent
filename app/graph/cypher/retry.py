"""
Cypher retry engine with LLM self-correction (Strategy #1).

Orchestrates:
  - Attempt 0: Normal ``GraphCypherQAChain`` invocation with few-shot prompt
  - Attempts 1‥N: Feed the error back to the LLM for self-correction
  - Transient-error retry via ``tenacity`` (network / timeout)
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

from app.graph.cypher.callback import CypherSafetyCallback
from app.graph.cypher.prompts import ENHANCED_CYPHER_PROMPT
from app.graph.cypher.safety import validate_read_only
from app.graph.cypher.validation import pre_validate_cypher
from app.core.exceptions import ReadOnlyViolationError

logger = logging.getLogger(__name__)

MAX_CYPHER_RETRIES = 2  # additional self-correction attempts after first failure


# ── Helpers ───────────────────────────────────────────────────────────────────


def build_correction_prompt(
    question: str, schema: str, failed_cypher: str, error: str,
) -> str:
    """Build a prompt asking the LLM to fix a failed Cypher query."""
    return (
        "The following Cypher query was generated to answer a question but "
        "failed when executed against Neo4j.  Fix the query and return ONLY "
        "the corrected Cypher — no explanations, no markdown fences.\n\n"
        f"Database schema:\n{schema}\n\n"
        f"Original question: {question}\n\n"
        f"Failed Cypher:\n{failed_cypher}\n\n"
        f"Error message:\n{error}\n\n"
        "Corrected Cypher:"
    )


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences that LLMs sometimes wrap around Cypher."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # drop closing fence
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
        r"Generated Cypher:\n(.+?)(?:\n\n|\nFull)",   # LangChain format
        r"(?:Invalid input|Syntax error).*?'(.+?)'",  # neo4j driver format
    ]:
        m = re.search(pat, error_msg, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


# ── Retry engine ──────────────────────────────────────────────────────────────


async def execute_with_retries(
    question: str,
    llm: BaseChatModel,
    graph: Any,
    schema: str,
) -> str:
    """
    Run the Cypher chain, retrying with LLM self-correction on failure.

    Attempt 0: Normal chain invocation with few-shot prompt.
    Attempts 1‥N: LLM self-correction using the error as context.
    """
    loop = asyncio.get_running_loop()
    safety_cb = CypherSafetyCallback()

    # ── Strategy #3: Chain with few-shot Cypher prompt ───────────────────
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=logger.isEnabledFor(logging.DEBUG),
        return_intermediate_steps=True,
        allow_dangerous_requests=True,
        cypher_prompt=ENHANCED_CYPHER_PROMPT,
    )

    # Transient-error retry (network / timeout — NOT Cypher errors)
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

    # ── Attempt 0: normal chain invocation ───────────────────────────────
    last_error: str | None = None
    last_cypher: str | None = None

    try:
        result = await loop.run_in_executor(None, lambda: _invoke(question))

        # Belt-and-suspenders: validate intermediate Cypher post-execution
        for step in result.get("intermediate_steps", []):
            if isinstance(step, dict) and "query" in step:
                validate_read_only(step["query"])

        answer: str = result.get("result", "I could not determine an answer.")
        logger.info("Graph query answered: %r → %r", question[:60], answer[:120])
        return answer

    except ReadOnlyViolationError:
        raise  # never retry write attempts

    except Exception as exc:
        if not is_cypher_error(exc):
            raise  # non-Cypher errors (auth, connectivity) bubble up

        last_error = str(exc)
        last_cypher = (
            safety_cb.last_generated_cypher
            or extract_cypher_from_error(last_error)
        )
        logger.warning(
            "Cypher attempt 0 failed (retryable): %s", last_error[:200],
        )

    # ── Attempts 1‥N: LLM self-correction (Strategy #1) ─────────────────
    for attempt in range(1, MAX_CYPHER_RETRIES + 1):
        try:
            correction = build_correction_prompt(
                question=question,
                schema=schema,
                failed_cypher=last_cypher or "N/A",
                error=last_error or "unknown error",
            )

            # Ask LLM for corrected Cypher
            resp = await loop.run_in_executor(
                None, lambda cp=correction: llm.invoke(cp),
            )
            corrected_cypher = strip_code_fences(str(resp.content))
            logger.info(
                "Retry %d corrected Cypher: %s", attempt, corrected_cypher[:200],
            )

            # Strategy #4 — pre-validate before hitting Neo4j
            validate_read_only(corrected_cypher)
            issues = pre_validate_cypher(corrected_cypher)
            if issues:
                last_error = f"Pre-validation: {', '.join(issues)}"
                last_cypher = corrected_cypher
                logger.warning("Corrected Cypher still invalid: %s", last_error)
                continue

            # Execute directly against Neo4j
            raw = await loop.run_in_executor(
                None, lambda cc=corrected_cypher: graph.query(cc),
            )

            # Generate natural-language answer from raw results
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
            return answer

        except ReadOnlyViolationError:
            raise

        except Exception as exc:
            last_error = str(exc)
            logger.warning("Cypher retry %d failed: %s", attempt, last_error[:200])

    raise RuntimeError(
        f"All {MAX_CYPHER_RETRIES + 1} Cypher attempts failed.  "
        f"Last error: {last_error}"
    )
