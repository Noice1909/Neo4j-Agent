"""
Agent 2: Coreference Resolution

Resolves pronouns ("it", "those", "them") to entity names using conversation context.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.agent.state import PipelineState
from src.graph.cypher.coreference import resolve_coreferences
from src.core.tracing import trace_event

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


def build_coreference_node(llm: BaseChatModel):
    """
    Build the coreference resolution agent node.

    Parameters
    ----------
    llm:
        Language model for rewriting questions with coreference resolution.

    Returns
    -------
    async function
        Node function for LangGraph: `(PipelineState) -> dict`
    """
    async def coreference_node(state: PipelineState) -> dict:
        """
        Resolve pronouns in the user question using conversation context.

        Reads
        -----
        - state["user_question"]: Raw user question
        - state["conversation_context"]: Recent conversation for context (optional)

        Returns
        -------
        dict
            State update with:
            - "coreferenced_question": Question with pronouns replaced by entities
        """
        question = state.get("user_question", "")
        context = state.get("conversation_context", "")

        trace_event("COREFERENCE_START", "info", f"Question: {question[:100]}")

        try:
            coreferenced = await resolve_coreferences(
                question=question,
                conversation_context=context,
                llm=llm,
            )

            if coreferenced != question:
                trace_event(
                    "COREFERENCE_RESOLVED",
                    "ok",
                    f"'{question[:60]}' → '{coreferenced[:60]}'",
                )
                logger.info(
                    "Coreference resolved: %r → %r",
                    question[:80],
                    coreferenced[:80],
                )
            else:
                trace_event("COREFERENCE_SKIP", "ok", "No coreferences detected")
                logger.debug("No coreferences detected in question")

            return {"coreferenced_question": coreferenced}

        except Exception as exc:
            logger.error(
                "Coreference resolution failed: %s",
                exc,
                exc_info=True,
            )
            trace_event("COREFERENCE_FAIL", "fail", str(exc)[:120])
            # Fallback: pass question through unchanged
            return {"coreferenced_question": question}

    return coreference_node
