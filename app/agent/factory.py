"""
Agent factory — assembles the LangGraph agent from injected dependencies.

`get_compiled_agent()` is the single point used by all API routes.
Swapping the agent implementation means changing only this file.

Module-level cache holds the compiled graph so it is built once per process
startup (graph compilation is synchronous and moderately expensive).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.agent.graph import build_agent_graph  # noqa: E402

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph as CompiledGraph

# Compiled agent singleton — set during lifespan startup.
_compiled_agent: "CompiledGraph | None" = None


def init_agent(
    llm: BaseChatModel,
    tools: list,
    checkpointer: BaseCheckpointSaver,
    *,
    schema_labels: list[str] | None = None,
    label_descriptions: dict[str, str] | None = None,
    max_conversation_tokens: int = 100_000,
    token_budget_reserve: int = 4096,
) -> None:
    """
    Build and cache the compiled LangGraph agent.

    Must be called once in the application lifespan startup after both the
    LLM and checkpointer are initialised.

    Parameters
    ----------
    llm:
        A configured `BaseChatModel` instance.
    tools:
        List of LangChain `BaseTool` instances to bind to the agent.
    checkpointer:
        An initialised `BaseCheckpointSaver` (e.g. `AsyncRedisSaver`).
    schema_labels:
        Canonical Neo4j node labels from the live schema (used to build the
        domain-aware system prompt).
    label_descriptions:
        Optional per-label descriptions from Concept nodes.
    max_conversation_tokens:
        Maximum token budget for conversation history.
    token_budget_reserve:
        Tokens reserved for model output.
    """
    global _compiled_agent
    logger.info("Building LangGraph agent with %d tool(s)...", len(tools))
    _compiled_agent = build_agent_graph(
        llm,
        tools,
        checkpointer,
        schema_labels=schema_labels,
        label_descriptions=label_descriptions,
        max_conversation_tokens=max_conversation_tokens,
        token_budget_reserve=token_budget_reserve,
    )
    logger.info("LangGraph agent ready.")


def get_compiled_agent() -> "CompiledGraph":
    """Return the compiled LangGraph agent (must call `init_agent` first)."""
    if _compiled_agent is None:
        raise RuntimeError(
            "Agent has not been initialised. "
            "Ensure `init_agent()` is called during application lifespan startup."
        )
    return _compiled_agent
