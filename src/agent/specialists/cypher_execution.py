"""
Agent 7: Cypher Execution

Executes the validated Cypher query against Neo4j.
Single attempt only — retry logic is handled by Retry Decision Agent.
Uses tenacity retry for transient network errors only.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from neo4j.exceptions import ServiceUnavailable, SessionExpired
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.agent.state import PipelineState
from src.core.tracing import trace_event

if TYPE_CHECKING:
    from langchain_neo4j import Neo4jGraph

logger = logging.getLogger(__name__)


def build_cypher_execution_node(graph: "Neo4jGraph"):
    """
    Build the Cypher execution agent node.

    Parameters
    ----------
    graph:
        Neo4j graph connection.

    Returns
    -------
    async function
        Node function for LangGraph: `(PipelineState) -> dict`
    """
    async def cypher_execution_node(state: PipelineState) -> dict:
        """
        Execute the Cypher query against Neo4j.

        Reads
        -----
        - state["generated_cypher"]: The validated Cypher query

        Returns
        -------
        dict
            State update with:
            - "raw_results": Neo4j query results (JSON-serialized)
            - "execution_succeeded": True if execution succeeded
            - "execution_error": Error message if execution failed
        """
        cypher = state.get("generated_cypher", "")
        retry_count = state.get("retry_count", 0)

        trace_event(
            "CYPHER_EXECUTION_START",
            "info",
            f"Attempt {retry_count}: {cypher[:100]}",
        )

        try:
            # Execute with tenacity retry for transient errors
            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, max=10),
                retry=retry_if_exception_type((
                    ConnectionError,
                    TimeoutError,
                    OSError,
                    ServiceUnavailable,
                    SessionExpired,
                )),
                reraise=True,
            )
            def _execute_cypher(query: str):
                return graph.query(query)

            # Run in executor to avoid blocking async event loop
            loop = asyncio.get_running_loop()
            raw_results = await loop.run_in_executor(
                None,
                lambda: _execute_cypher(cypher),
            )

            # Serialize results to JSON
            results_json = json.dumps(raw_results, default=str)

            trace_event(
                "CYPHER_EXECUTION_SUCCESS",
                "ok",
                f"Attempt {retry_count}: {len(raw_results)} result(s)",
            )
            logger.info(
                "Cypher executed successfully (attempt %d): %d result(s)",
                retry_count,
                len(raw_results),
            )

            return {
                "raw_results": results_json,
                "execution_succeeded": True,
                "execution_error": "",
            }

        except Exception as exc:
            error_msg = str(exc)
            trace_event(
                "CYPHER_EXECUTION_FAIL",
                "fail",
                f"Attempt {retry_count}: {error_msg[:120]}",
            )
            logger.warning(
                "Cypher execution failed (attempt %d): %s",
                retry_count,
                error_msg[:200],
            )

            return {
                "raw_results": "[]",
                "execution_succeeded": False,
                "execution_error": error_msg,
            }

    return cypher_execution_node
