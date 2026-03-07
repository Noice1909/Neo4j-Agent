"""
Agent 8: Result Verification

Analyzes execution results to determine if they are valid, meaningful, and answer the question.
Uses lightweight LLM call for semantic analysis.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from src.agent.state import PipelineState
from src.core.tracing import trace_event

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


def build_result_verification_node(llm: "BaseChatModel"):
    """
    Build the result verification agent node.

    Parameters
    ----------
    llm:
        Language model for semantic result analysis.

    Returns
    -------
    async function
        Node function for LangGraph: `(PipelineState) -> dict`
    """
    async def result_verification_node(state: PipelineState) -> dict:
        """
        Verify that execution results are valid and answer the question.

        Reads
        -----
        - state["raw_results"]: Neo4j query results (JSON string)
        - state["execution_succeeded"]: Whether execution succeeded
        - state["execution_error"]: Error message if failed
        - state["entity_resolved_question"]: The question being answered

        Returns
        -------
        dict
            State update with:
            - "results_valid": True if results answer the question
            - "verification_message": Explanation of verification result
        """
        execution_succeeded = state.get("execution_succeeded", False)
        execution_error = state.get("execution_error", "")
        raw_results = state.get("raw_results", "[]")
        question = state.get("entity_resolved_question", state.get("user_question", ""))

        trace_event("RESULT_VERIFICATION_START", "info", f"Execution succeeded: {execution_succeeded}")

        try:
            # Case 1: Execution failed
            if not execution_succeeded:
                trace_event("RESULT_VERIFICATION_FAIL", "fail", "Execution failed")
                logger.info("Result verification: execution failed")
                return {
                    "results_valid": False,
                    "verification_message": f"Execution failed: {execution_error}",
                }

            # Case 2: Empty results
            results = json.loads(raw_results)
            if not results:
                trace_event("RESULT_VERIFICATION_EMPTY", "warn", "No results returned")
                logger.info("Result verification: empty results")
                # Ask LLM: is empty result valid for this question?
                verification_prompt = (
                    f"Question: {question}\n"
                    f"Cypher query returned no results (empty list).\n\n"
                    "Is this a valid answer to the question? "
                    "Respond with 'VALID' if an empty result makes sense (e.g., 'how many' → 0), "
                    "or 'INVALID' if the query likely failed."
                )

                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: llm.invoke(verification_prompt),
                )

                is_valid = "VALID" in str(response.content).upper()
                trace_event(
                    "RESULT_VERIFICATION_EMPTY_DECIDED",
                    "ok" if is_valid else "warn",
                    f"Empty result is {'valid' if is_valid else 'invalid'}",
                )

                return {
                    "results_valid": is_valid,
                    "verification_message": "Empty result (deemed valid)" if is_valid else "Empty result (likely invalid query)",
                }

            # Case 3: Results returned — verify they answer the question
            results_preview = str(results[:3])[:500]  # First 3 results, max 500 chars
            verification_prompt = (
                f"Question: {question}\n"
                f"Results (preview): {results_preview}\n\n"
                "Do these results answer the question? "
                "Respond with 'YES' if they provide a meaningful answer, 'NO' if they seem unrelated or incorrect."
            )

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: llm.invoke(verification_prompt),
            )

            results_answer_question = "YES" in str(response.content).upper()

            if results_answer_question:
                trace_event(
                    "RESULT_VERIFICATION_PASS",
                    "ok",
                    f"{len(results)} result(s), semantically valid",
                )
                logger.info(
                    "Result verification passed: %d result(s) answer the question",
                    len(results),
                )
                return {
                    "results_valid": True,
                    "verification_message": f"{len(results)} result(s) found, answer question",
                }
            else:
                trace_event(
                    "RESULT_VERIFICATION_SEMANTIC_FAIL",
                    "warn",
                    f"{len(results)} result(s), but don't answer question",
                )
                logger.warning(
                    "Result verification failed: results don't answer the question",
                )
                return {
                    "results_valid": False,
                    "verification_message": f"{len(results)} result(s) found, but don't answer question",
                }

        except Exception as exc:
            logger.error(
                "Result verification encountered error: %s",
                exc,
                exc_info=True,
            )
            trace_event("RESULT_VERIFICATION_ERROR", "fail", str(exc)[:120])
            # Conservative: treat as invalid
            return {
                "results_valid": False,
                "verification_message": f"Verification error: {exc}",
            }

    return result_verification_node
