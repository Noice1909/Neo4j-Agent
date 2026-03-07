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
    from src.graph.topology import GraphTopology

logger = logging.getLogger(__name__)


# ── Entity label lookup helper ───────────────────────────────────────────


async def _lookup_entity_labels(
    graph: "Neo4jGraph",
    question: str,
    topology: "GraphTopology",
) -> list[dict]:
    """
    Extract capitalized name candidates from *question* and query Neo4j
    to find which label each entity belongs to.

    Returns a list of hints like:
        [{"entity_name": "Tom Hanks", "label": "Actor", "property": "name", "match_type": "exact"}]
    """
    import asyncio
    import re

    # Extract candidates: quoted strings + capitalized multi-word phrases
    candidates: list[str] = []
    for m in re.finditer(r"""["']([^"']+)["']""", question):
        candidates.append(m.group(1))
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", question):
        phrase = m.group(1)
        if phrase not in candidates:
            candidates.append(phrase)

    if not candidates:
        return []

    # Use name and title for lookup (avoid 'label' property which conflicts
    # with the labels() function alias)
    lookup_props = [p for p in (topology.display_properties or ["name", "title"])
                    if p not in ("label",)][:3]
    if not lookup_props:
        lookup_props = ["name", "title"]
    hints: list[dict] = []

    loop = asyncio.get_running_loop()
    for candidate in candidates:
        where_clauses = " OR ".join(f"n.{p} = $name" for p in lookup_props)
        return_props = ", ".join(f"n.{p} AS prop_{p}" for p in lookup_props)
        cypher = (
            f"MATCH (n) WHERE {where_clauses} "
            f"RETURN labels(n)[0] AS node_label, {return_props} "
            "LIMIT 1"
        )
        try:
            results = await loop.run_in_executor(
                None,
                lambda c=candidate, q=cypher: graph.query(q, params={"name": c}),
            )
            if results:
                row = results[0]
                node_label = row.get("node_label", "")
                if not node_label:
                    continue
                # Find which property matched
                matched_prop = lookup_props[0]
                for p in lookup_props:
                    val = row.get(f"prop_{p}")
                    if val and str(val) == candidate:
                        matched_prop = p
                        break
                hints.append({
                    "entity_name": candidate,
                    "label": node_label,
                    "property": matched_prop,
                    "match_type": "exact",
                })
        except Exception as exc:
            logger.debug(
                "Entity label lookup failed for '%s': %s", candidate, exc,
            )

    return hints


def build_entity_resolution_node(
    llm: BaseChatModel,
    schema_cache: "SchemaCache",
    graph: "Neo4jGraph",
    settings: "Settings",
    semantic_layer=None,
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
    semantic_layer:
        Optional SchemaSemanticLayer for NL-aware property/relationship mapping.

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

            # ── Layer 1.5: Property term resolution ──────────────────
            property_mappings = []
            try:
                from src.graph.cypher.synonyms import build_property_synonym_map
                prop_syn_map = build_property_synonym_map(topology, semantic_layer)

                # Check each word in the question against property synonyms
                resolved_q = resolution.resolved_question
                for word in set(resolved_q.lower().split()):
                    candidate = word.strip("?.,!'\"-")
                    if len(candidate) >= 3 and candidate in prop_syn_map:
                        label, prop_name = prop_syn_map[candidate]
                        # Don't add if the word is also a label synonym (avoid ambiguity here)
                        property_mappings.append({
                            "nl_term": candidate,
                            "label": label,
                            "property": prop_name,
                            "confidence": 0.9,
                        })
                if property_mappings:
                    logger.info(
                        "Property resolution: %d mapping(s) found: %s",
                        len(property_mappings),
                        ", ".join(f"{m['nl_term']}→{m['label']}.{m['property']}" for m in property_mappings[:3]),
                    )
                    trace_event("PROPERTY_RES", "ok", f"{len(property_mappings)} property mapping(s)")
            except Exception as exc:
                logger.debug("Property resolution failed (non-fatal): %s", exc)

            # ── Layer 2.5: Entity label lookup ───────────────────
            # For entity names in the question (exact or fuzzy matched),
            # discover which label they belong to so Cypher gen knows
            # e.g. "Tom Hanks" is an Actor.name
            # Use ORIGINAL question for candidate extraction (before entity
            # resolution capitalizes label words, which merges with names)
            entity_hints = []
            try:
                entity_hints = await _lookup_entity_labels(
                    graph, question, topology,
                )
                if entity_hints:
                    logger.info(
                        "Entity hints: %s",
                        ", ".join(
                            f"{h['entity_name']}→{h['label']}.{h['property']}"
                            for h in entity_hints[:5]
                        ),
                    )
                    trace_event(
                        "ENTITY_HINTS",
                        "ok",
                        f"{len(entity_hints)} hint(s)",
                    )
            except Exception as exc:
                logger.debug("Entity label lookup failed (non-fatal): %s", exc)

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
                "property_mappings": property_mappings,
                "entity_hints": entity_hints,
            }

        except Exception as exc:
            logger.error(
                "Entity resolution failed: %s",
                exc,
                exc_info=True,
            )
            trace_event("ENTITY_RES_FAIL", "fail", str(exc)[:120])
            # Fallback: pass question through unchanged, still provide topology
            fallback_topology_json = ""
            try:
                topo = await schema_cache.get_topology()
                from src.graph.schema_cache import _topology_to_json
                fallback_topology_json = _topology_to_json(topo)
            except Exception:
                pass
            return {
                "entity_resolved_question": question,
                "resolution_corrections": [],
                "full_topology_json": fallback_topology_json,
            }

    return entity_resolution_node
