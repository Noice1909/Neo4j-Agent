"""
Agent 3: Entity Resolution

Corrects entity names, fixes typos, maps synonyms, resolves label names.
Uses the 4-layer resolution pipeline (FT index → label resolver → name resolver → LLM fallback).
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import TYPE_CHECKING

from src.agent.state import PipelineState
from src.graph.cypher.entity_resolution import resolve_entities
from src.graph.cypher.prompts import build_topology_section
from src.core.tracing import trace_event

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_neo4j import Neo4jGraph
    from src.core.config import Settings
    from src.graph.schema_cache import SchemaCache

logger = logging.getLogger(__name__)


def build_entity_resolution_node(
    llm: BaseChatModel,
    schema_cache: "SchemaCache",
    graph: "Neo4jGraph",
    settings: "Settings",
):
    """
    Build the entity resolution agent node.

    Parameters
    ----------
    llm:
        Language model for LLM fallback resolution (Layer 3).
    schema_cache:
        Schema cache for fetching live schema + topology.
    graph:
        Neo4j graph connection for querying full-text indexes.
    settings:
        Application settings (entity resolution config).

    Returns
    -------
    async function
        Node function for LangGraph: `(PipelineState) -> dict`
    """
    async def entity_resolution_node(state: PipelineState) -> dict:
        """
        Resolve entity names, labels, and synonyms.

        Reads
        -----
        - state["coreferenced_question"]: Question after coreference resolution

        Returns
        -------
        dict
            State update with:
            - "entity_resolved_question": Corrected question
            - "resolution_corrections": List of applied corrections (serialized)
            - "full_topology_json": Full topology for downstream agents (serialized)
        """
        question = state.get("coreferenced_question", state.get("user_question", ""))

        trace_event("ENTITY_RES_START", "info", f"Question: {question[:100]}")

        try:
            # Fetch live schema and topology
            schema = await schema_cache.get_schema()
            topology = await schema_cache.get_topology()

            # Build topology section for entity resolution context
            topology_section_full = build_topology_section(topology)

            # Run 4-layer resolution pipeline
            resolution = await resolve_entities(
                question=question,
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

            if resolution.corrections:
                corrections_summary = ", ".join(
                    f"{c.original}→{c.corrected}" for c in resolution.corrections[:3]
                )
                trace_event(
                    "ENTITY_RES_DONE",
                    "ok",
                    f"{len(resolution.corrections)} correction(s): {corrections_summary}",
                )
                logger.info(
                    "Entity resolution: %d correction(s) applied",
                    len(resolution.corrections),
                )
            else:
                trace_event("ENTITY_RES_SKIP", "ok", "No corrections needed")
                logger.debug("No entity resolution corrections needed")

            # Serialize corrections and topology for state
            import json
            corrections_json = [asdict(c) for c in resolution.corrections]

            # Serialize topology to JSON
            from src.graph.schema_cache import _topology_to_json
            topology_json = _topology_to_json(topology)

            return {
                "entity_resolved_question": resolution.resolved_question,
                "resolution_corrections": corrections_json,
                "full_topology_json": topology_json,
            }

        except Exception as exc:
            logger.error(
                "Entity resolution failed: %s",
                exc,
                exc_info=True,
            )
            trace_event("ENTITY_RES_FAIL", "fail", str(exc)[:120])
            # Fallback: pass question through unchanged
            return {
                "entity_resolved_question": question,
                "resolution_corrections": [],
            }

    return entity_resolution_node
