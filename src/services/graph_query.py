"""
Graph query service — natural language → Cypher → Neo4j → answer.

Applies all five production-hardening strategies.
"""
from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from src.core.exceptions import ReadOnlyViolationError
from src.graph.connection import get_graph
from src.graph.cypher.coreference import resolve_coreferences
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
