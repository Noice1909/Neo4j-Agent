"""
Graph query service — natural language → Cypher → Neo4j → answer.

Applies all five production-hardening strategies.
"""
from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from src.core.config import get_settings
from src.core.exceptions import ReadOnlyViolationError
from src.graph.connection import get_graph
from src.graph.cypher.coreference import resolve_coreferences
from src.graph.cypher.entity_resolver import resolve_entities
from src.graph.cypher.retry import execute_with_retries
from src.graph.schema_cache import SchemaCache

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

    Returns
    -------
    str
        A natural-language answer, or a friendly fallback message on failure.

    Raises
    ------
    ReadOnlyViolationError
        If the generated Cypher contains write operations.
    """
    graph = get_graph()
    schema = await schema_cache.get_schema()
    graph.schema = schema

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
    )
    resolved_question = resolution.resolved_question

    try:
        return await execute_with_retries(resolved_question, llm, graph, schema)
    except ReadOnlyViolationError:
        raise
    except Exception as exc:
        logger.error(
            "All Cypher attempts failed for %r: %s",
            question[:80], exc, exc_info=True,
        )
        return (
            "I'm sorry, I couldn't find that information right now. "
            "Could you try asking in a different way?"
        )
