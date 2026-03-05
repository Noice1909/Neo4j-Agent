"""
Graph query service — natural language → Cypher → Neo4j → answer.

Applies all five production-hardening strategies.
"""
from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from src.core.config import get_settings
from src.core.exceptions import ReadOnlyViolationError
from src.graph.connection import ensure_connected
from src.graph.cypher.coreference import resolve_coreferences
from src.graph.cypher.dynamic_examples import generate_few_shot_examples
from src.graph.cypher.entity_resolution import resolve_entities
from src.graph.cypher.prompts import build_cypher_prompt, build_topology_section
from src.graph.cypher.retry import execute_with_retries
from src.graph.cypher.topology_filter import filter_topology
from src.graph.schema_cache import SchemaCache
from src.core.tracing import trace_event

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
    # Verify Neo4j is reachable — auto-reconnects on stale/dead connections
    graph = ensure_connected()
    schema = await schema_cache.get_schema()
    topology = await schema_cache.get_topology()
    graph.schema = schema

    trace_event("GRAPH_QUERY_START", "info", question[:100])

    # ── Coreference resolution ────────────────────────────────────────────
    resolved_question = await resolve_coreferences(
        question, conversation_context, llm,
    )

    # ── Entity resolution: correct typos, wrong labels, and synonyms ─────
    settings = get_settings()
    # Build a basic topology section for entity resolution context
    topology_section_full = build_topology_section(topology)
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
        id_index_name=settings.entity_id_index_name,
        display_properties=topology.display_properties,
        topology_section=topology_section_full,
        concept_nlp_terms=topology.nlp_terms_by_label,
    )
    resolved_question = resolution.resolved_question

    # ── Filter topology to question-relevant subset ───────────────────────
    filtered = filter_topology(resolved_question, topology, resolution)

    # ── Build prompt from filtered topology ───────────────────────────────
    few_shot = generate_few_shot_examples(filtered, question=resolved_question)
    cypher_prompt = build_cypher_prompt(filtered, few_shot)
    topology_section = build_topology_section(
        filtered, full_valid_types=topology.valid_rel_types,
    )

    try:
        return await execute_with_retries(
            resolved_question, llm, graph, schema,
            cypher_prompt=cypher_prompt,
            topology_section=topology_section,
            valid_rel_types=topology.valid_rel_types,
            topology=topology,
        )
    except ReadOnlyViolationError:
        raise
    except Exception as exc:
        logger.error(
            "All Cypher attempts failed for %r: %s",
            question[:80], exc, exc_info=True,
        )
        trace_event("GRAPH_QUERY_FALLBACK", "fail", f"Graceful degradation: {exc}"[:120])
        return (
            "I'm sorry, I couldn't find that information right now. "
            "Could you try asking in a different way?"
        )
