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

from src.agent.supervisor import build_supervisor_graph  # noqa: E402

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph as CompiledGraph

# Compiled agent singleton — set during lifespan startup.
_compiled_agent: "CompiledGraph | None" = None


def init_agent(
    llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver,
    schema_cache,
    topology,
    graph,
    settings,
) -> None:
    """
    Build and cache the compiled multi-agent supervisor system.

    Must be called once in the application lifespan startup after all
    dependencies are initialized.

    Parameters
    ----------
    llm:
        A configured `BaseChatModel` instance.
    checkpointer:
        An initialized `BaseCheckpointSaver` (e.g. `AsyncRedisSaver`).
    schema_cache:
        Schema cache instance (SchemaCache).
    topology:
        Graph topology (GraphTopology).
    graph:
        Neo4j graph connection (Neo4jGraph).
    settings:
        Application settings (Settings).
    """
    global _compiled_agent
    logger.info("Building multi-agent supervisor system...")
    _compiled_agent = build_supervisor_graph(
        llm=llm,
        checkpointer=checkpointer,
        schema_cache=schema_cache,
        topology=topology,
        graph=graph,
        settings=settings,
    )
    logger.info("Multi-agent system ready (1 supervisor + 9 pipeline agents).")


def get_compiled_agent() -> "CompiledGraph":
    """Return the compiled LangGraph agent (must call `init_agent` first)."""
    if _compiled_agent is None:
        raise RuntimeError(
            "Agent has not been initialised. "
            "Ensure `init_agent()` is called during application lifespan startup."
        )
    return _compiled_agent
