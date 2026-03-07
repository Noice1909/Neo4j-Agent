"""
Agent 9: Retry Decision

Decides whether to retry Cypher generation (with correction) or proceed to synthesis.
Analyzes errors and determines if retry will help.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from src.agent.state import PipelineState
from src.graph.cypher.retry import build_correction_prompt
from src.core.tracing import trace_event

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def build_retry_decision_node(llm: "BaseChatModel"):
    """
    Build the retry decision agent node.

    Parameters
    ----------
    llm:
        Language model for analyzing errors and deciding retry strategy.

    Returns
    -------
    async function
        Node function for LangGraph: `(PipelineState) -> dict`
    """
    async def retry_decision_node(state: PipelineState) -> dict:
        """
        Decide whether to retry Cypher generation or proceed to synthesis.

        Reads
        -----
        - state["results_valid"]: Whether results are valid
        - state["execution_error"]: Error message if execution failed
        - state["retry_count"]: Current retry attempt
        - state["generated_cypher"]: The Cypher that was attempted
        - state["validation_errors"]: Validation errors if applicable
        - state["verification_message"]: Verification result message

        Returns
        -------
        dict
            State update with:
            - "should_retry": True if retry recommended
            - "retry_count": Incremented if retrying
            - "correction_guidance": Hints for next generation attempt
        """
        results_valid = state.get("results_valid", False)
        execution_error = state.get("execution_error", "")
        retry_count = state.get("retry_count", 0)
        generated_cypher = state.get("generated_cypher", "")
        validation_errors = state.get("validation_errors", [])
        verification_message = state.get("verification_message", "")

        trace_event(
            "RETRY_DECISION_START",
            "info",
            f"Attempt {retry_count}, valid={results_valid}",
        )

        try:
            # ── Case 1: Results are valid → no retry needed ─────────────
            if results_valid:
                trace_event("RETRY_DECISION_PROCEED", "ok", "Results valid, proceeding to synthesis")
                logger.info("Retry decision: results valid, proceeding to synthesis")
                return {
                    "should_retry": False,
                    "correction_guidance": "",
                }

            # ── Case 2: Max retries exhausted → give up ─────────────────
            if retry_count >= MAX_RETRIES:
                trace_event(
                    "RETRY_DECISION_EXHAUSTED",
                    "fail",
                    f"Max retries ({MAX_RETRIES}) exhausted",
                )
                logger.warning(
                    "Retry decision: max retries (%d) exhausted, proceeding to fallback",
                    MAX_RETRIES,
                )
                return {
                    "should_retry": False,
                    "correction_guidance": "",
                }

            # ── Case 3: Determine if retry will help ────────────────────
            # Build error context
            error_context = []
            if validation_errors:
                error_context.append(f"Validation errors: {'; '.join(validation_errors)}")
            if execution_error:
                error_context.append(f"Execution error: {execution_error}")
            if verification_message:
                error_context.append(f"Verification: {verification_message}")

            error_summary = " | ".join(error_context)

            # Ask LLM: should we retry?
            decision_prompt = (
                f"The following Cypher query failed:\n```\n{generated_cypher}\n```\n\n"
                f"Errors: {error_summary}\n\n"
                "Analyze the errors and decide:\n"
                "- If the errors are fixable (syntax, schema, logic), respond: RETRY\n"
                "- If the errors are unfixable (permissions, database issues), respond: GIVE_UP\n"
                "If RETRY, provide a brief hint (1 sentence) for correcting the query."
            )

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: llm.invoke(decision_prompt),
            )

            response_text = str(response.content)

            should_retry = "RETRY" in response_text.upper() and "GIVE_UP" not in response_text.upper()

            if should_retry:
                # Extract correction guidance (everything after "RETRY")
                correction_guidance = response_text.split("RETRY", 1)[-1].strip()
                if not correction_guidance:
                    correction_guidance = f"Fix errors: {error_summary[:200]}"

                new_retry_count = retry_count + 1
                trace_event(
                    "RETRY_DECISION_RETRY",
                    "info",
                    f"Retrying (attempt {new_retry_count}/{MAX_RETRIES})",
                )
                logger.info(
                    "Retry decision: retrying (attempt %d/%d). Guidance: %s",
                    new_retry_count,
                    MAX_RETRIES,
                    correction_guidance[:100],
                )

                return {
                    "should_retry": True,
                    "retry_count": new_retry_count,
                    "correction_guidance": correction_guidance,
                }
            else:
                trace_event("RETRY_DECISION_GIVE_UP", "fail", "LLM decided retry won't help")
                logger.warning("Retry decision: LLM decided retry won't help")
                return {
                    "should_retry": False,
                    "correction_guidance": "",
                }

        except Exception as exc:
            logger.error(
                "Retry decision encountered error: %s",
                exc,
                exc_info=True,
            )
            trace_event("RETRY_DECISION_ERROR", "fail", str(exc)[:120])
            # Conservative: don't retry on error
            return {
                "should_retry": False,
                "correction_guidance": "",
            }

    return retry_decision_node
