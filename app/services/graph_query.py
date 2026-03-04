"""
Graph query service — natural language → Cypher → Neo4j → answer.

Applies all five production-hardening strategies:

1. **Retry with error feedback** (Strategy #1)
2. **Graceful degradation** (Strategy #2)
3. **Few-shot examples** (Strategy #3)
4. **Cypher syntax pre-validation** (Strategy #4)
5. **Query decomposition / coreference resolution** (Strategy #5)

This module is framework-agnostic; see ``app.mcp.tools.graph_query`` for the
LangChain ``@tool`` wrapper and FastMCP registration.
"""
from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from app.core.config import get_settings
from app.core.exceptions import ReadOnlyViolationError
from app.graph.connection import get_graph, ensure_connected
from app.graph.cypher.coreference import resolve_coreferences
from app.graph.cypher.entity_resolver import resolve_entities
from app.graph.cypher.retry import execute_with_retries
from app.graph.schema_cache import SchemaCache

logger = logging.getLogger(__name__)


async def run_graph_query(
    question: str,
    llm: BaseChatModel,
    schema_cache: SchemaCache,
    *,
    conversation_context: str | None = None,
) -> str:
    """
    Translate a natural-language question into Cypher, execute, and answer.

    Parameters
    ----------
    question:
        Natural-language question from the user.
    llm:
        The configured LLM for Cypher generation.
    schema_cache:
        Schema cache used to inject the DB schema into the Cypher prompt.
    conversation_context:
        Optional prior-conversation summary for coreference resolution.

    Returns
    -------
    str
        A natural-language answer, or a friendly fallback message on failure.

    Raises
    ------
    ReadOnlyViolationError
        If the generated Cypher contains write operations (security — always raised).
    """
    # Verify Neo4j is reachable — auto-reconnects on stale/dead connections
    graph = ensure_connected()
    schema = await schema_cache.get_schema()
    graph.schema = schema

    # ── Strategy #5: Resolve coreferences ────────────────────────────────
    resolved_question = await resolve_coreferences(
        question, conversation_context, llm,
    )

    # ── Entity resolution: correct typos, wrong labels, and synonyms ─────
    settings = get_settings()
    resolution = await resolve_entities(
        question=resolved_question,
        schema=schema,
        graph=graph,
        llm=llm,
        enabled=settings.entity_resolution_enabled,
        fuzzy_threshold=settings.entity_fuzzy_threshold,
        synonym_overrides=settings.entity_synonym_overrides,
        max_candidates=settings.entity_max_candidates,
        fulltext_index_name=settings.entity_fulltext_index_name,
    )
    resolved_question = resolution.resolved_question

    # ── Strategy #2: Graceful degradation (outer safety net) ─────────────
    try:
        return await execute_with_retries(resolved_question, llm, graph, schema)
    except ReadOnlyViolationError:
        raise  # Security: always propagate write-attempt errors
    except Exception as exc:
        logger.error(
            "All Cypher attempts failed for %r: %s",
            question[:80], exc, exc_info=True,
        )
        return (
            "I'm sorry, I couldn't find that information right now. "
            "Could you try asking in a different way?"
        )
