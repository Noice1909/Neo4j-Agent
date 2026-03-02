"""
FastAPI dependency injection providers.

Every injectable component is defined here.  Route handlers import
``Depends(get_*)`` from this module — never directly from singletons.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import Depends

from src.core.config import Settings, get_settings

if TYPE_CHECKING:
    from langchain_neo4j import Neo4jGraph
    from langchain_ollama import ChatOllama
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph as CompiledGraph

    from src.graph.schema_cache import SchemaCache
    from src.services.query_dedup import QueryDeduplicator

# Module-level schema_cache reference — set during lifespan startup.
_schema_cache_instance: "SchemaCache | None" = None

# Module-level query dedup reference — set during lifespan startup.
_query_dedup_instance: "QueryDeduplicator | None" = None


def set_schema_cache_instance(sc: "SchemaCache") -> None:
    """Store the schema cache singleton (called once during lifespan startup)."""
    global _schema_cache_instance
    _schema_cache_instance = sc


def get_schema_cache_instance() -> "SchemaCache":
    """Return the schema cache singleton (raises if not yet initialised)."""
    if _schema_cache_instance is None:
        raise RuntimeError("SchemaCache has not been initialised.")
    return _schema_cache_instance


def set_query_dedup_instance(qd: "QueryDeduplicator") -> None:
    """Store the query dedup singleton (called once during lifespan startup)."""
    global _query_dedup_instance
    _query_dedup_instance = qd


def get_query_dedup_instance() -> "QueryDeduplicator":
    """Return the query dedup singleton (raises if not yet initialised)."""
    if _query_dedup_instance is None:
        raise RuntimeError("QueryDeduplicator has not been initialised.")
    return _query_dedup_instance


# ── FastAPI Depends providers ─────────────────────────────────────────────────


def get_neo4j_graph() -> "Neo4jGraph":
    """Return the Neo4j graph singleton."""
    from src.graph.connection import get_graph

    return get_graph()


def get_llm(settings: Settings = Depends(get_settings)) -> "ChatOllama":
    """Return the configured LLM singleton."""
    from src.llm.factory import get_llm_from_settings

    return get_llm_from_settings(settings)


def get_checkpointer() -> "BaseCheckpointSaver":
    """Return the checkpointer singleton (SQLite/Redis/Memory, based on config)."""
    from src.agent.checkpointer import get_checkpointer as _get

    return _get()


def get_schema_cache() -> "SchemaCache":
    """Return the in-memory schema cache singleton."""
    return get_schema_cache_instance()


def get_agent() -> "CompiledGraph":
    """Return the compiled LangGraph agent."""
    from src.agent.factory import get_compiled_agent

    return get_compiled_agent()


def get_query_dedup() -> "QueryDeduplicator":
    """Return the query deduplicator singleton."""
    return get_query_dedup_instance()
