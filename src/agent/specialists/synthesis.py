"""
Agent 10: Synthesis

Converts raw Neo4j results into a natural-language answer.
QA synthesis — the final step in the pipeline.
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


def build_synthesis_node(llm: "BaseChatModel"):
    """
    Build the synthesis agent node.

    Parameters
    ----------
    llm:
        Language model for natural language synthesis.

    Returns
    -------
    async function
        Node function for LangGraph: `(PipelineState) -> dict`
    """
    async def synthesis_node(state: PipelineState) -> dict:
        """
        Synthesize raw results into a natural-language answer.

        Reads
        -----
        - state["entity_resolved_question"]: The question being answered
        - state["raw_results"]: Neo4j query results (JSON string)

        Returns
        -------
        dict
            State update with:
            - "final_answer": Natural language answer
        """
        question = state.get("entity_resolved_question", state.get("user_question", ""))
        raw_results = state.get("raw_results", "[]")

        trace_event("SYNTHESIS_START", "info", f"Question: {question[:100]}")

        try:
            # Deserialize results
            results = json.loads(raw_results)

            # Handle empty results
            if not results:
                trace_event("SYNTHESIS_EMPTY", "info", "No results to synthesize")
                logger.info("Synthesis: no results, returning empty answer")
                return {
                    "final_answer": "I couldn't find any information to answer that question.",
                }

            # Build QA synthesis prompt
            results_text = str(results)[:2000]  # Limit to 2000 chars for prompt
            qa_prompt = (
                f"Question: {question}\n"
                f"Data: {results_text}\n\n"
                "Answer the question using ONLY the data provided above.\n\n"
                "MANDATORY RULES - YOU MUST FOLLOW THESE EXACTLY:\n"
                "1. Write your answer in simple, natural language\n"
                "2. FORBIDDEN WORDS - Never use: database, query, Cypher, Neo4j, graph, schema, nodes, relationships, system, technical\n"
                "3. FORBIDDEN PHRASES - Never use:\n"
                "   - 'The database contains...'\n"
                "   - 'In the database...'\n"
                "   - 'According to the database...'\n"
                "   - 'The query shows...'\n"
                "   - 'Based on the database...'\n"
                "4. CORRECT examples:\n"
                "   ✓ 'There are 8 movies'\n"
                "   ✓ 'Christopher Nolan directed 3 movies'\n"
                "   ✓ 'The top actors are...'\n"
                "5. WRONG examples (DO NOT USE):\n"
                "   ✗ 'There are 8 movies in the database'\n"
                "   ✗ 'The database shows Christopher Nolan directed 3 movies'\n"
                "   ✗ 'According to the database, the top actors are...'\n"
                "6. Answer as if you naturally know this information\n\n"
                "Your answer:"
            )

            # Invoke LLM for synthesis
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: llm.invoke(qa_prompt),
            )

            final_answer = str(response.content).strip()

            trace_event(
                "SYNTHESIS_DONE",
                "ok",
                f"Answer: {final_answer[:100]}",
            )
            logger.info(
                "Synthesis complete: %r → %r",
                question[:80],
                final_answer[:120],
            )

            return {"final_answer": final_answer}

        except Exception as exc:
            logger.error(
                "Synthesis failed: %s",
                exc,
                exc_info=True,
            )
            trace_event("SYNTHESIS_FAIL", "fail", str(exc)[:120])
            # Fallback: generic message
            return {
                "final_answer": "I encountered an error while formatting the answer. Please try again.",
            }

    return synthesis_node
