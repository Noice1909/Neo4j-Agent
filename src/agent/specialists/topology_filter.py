"""
Agent 4: Topology Filter + Schema Reasoning

Reduces the full graph topology to a question-relevant subset,
then runs schema reasoning to detect and resolve ambiguities
(e.g., "genre" could be a property OR a node label).

Deterministic path when no ambiguity exists; ONE lightweight LLM call
only when a term maps to multiple schema element types.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from src.agent.state import PipelineState
from src.graph.cypher.topology_filter import filter_topology
from src.graph.cypher.entity_resolution import ResolutionResult
from src.graph.cypher.schema_reasoning import (
    build_schema_context,
    detect_ambiguities,
    resolve_ambiguity,
)
from src.graph.cypher.synonyms import (
    build_property_synonym_map,
    build_relationship_synonym_map,
    build_synonym_map,
)
from src.core.tracing import trace_event

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from src.graph.schema_cache import SchemaCache
    from src.graph.semantic_layer import SchemaSemanticLayer

logger = logging.getLogger(__name__)


def build_topology_filter_node(
    schema_cache: "SchemaCache",
    llm: "BaseChatModel | None" = None,
    semantic_layer: "SchemaSemanticLayer | None" = None,
):
    """
    Build the topology filter + schema reasoning agent node.

    Parameters
    ----------
    schema_cache:
        Schema cache for fetching full topology.
    llm:
        Language model for ambiguity resolution (only used when ambiguous).
    semantic_layer:
        Optional SchemaSemanticLayer for NL-aware property/relationship mapping.

    Returns
    -------
    async function
        Node function for LangGraph: `(PipelineState) -> dict`
    """
    async def topology_filter_node(state: PipelineState) -> dict:
        """
        Filter topology to question-relevant subset and run schema reasoning.

        Reads
        -----
        - state["entity_resolved_question"]: Corrected question
        - state["resolution_corrections"]: Applied corrections
        - state["full_topology_json"]: Full topology (optional, will fetch if missing)
        - state["property_mappings"]: Property NL→schema mappings from entity resolution

        Returns
        -------
        dict
            State update with:
            - "filtered_topology_json": Filtered topology (serialized)
            - "schema_context": Resolved schema hints for Cypher generation
        """
        question = state.get("entity_resolved_question", state.get("user_question", ""))

        trace_event("TOPOLOGY_FILTER_START", "info", f"Question: {question[:100]}")

        try:
            # Deserialize full topology (or fetch fresh)
            from src.graph.schema_cache import _topology_from_json

            topology_json = state.get("full_topology_json")
            if topology_json:
                topology = _topology_from_json(topology_json)
            else:
                topology = await schema_cache.get_topology()

            # Reconstruct ResolutionResult from corrections
            corrections = state.get("resolution_corrections", [])
            from src.graph.cypher.entity_resolution import Correction

            corrections_objs = []
            if corrections:
                for c_dict in corrections:
                    correction = Correction(
                        original=c_dict["original"],
                        corrected=c_dict["corrected"],
                        layer=c_dict.get("layer", "unknown"),
                        confidence=c_dict.get("confidence", 0.0),
                    )
                    corrections_objs.append(correction)

            resolution = ResolutionResult(
                original_question=state.get("user_question", question),
                resolved_question=question,
                corrections=corrections_objs,
            )

            # ── Step 1: Filter topology (existing deterministic logic) ─────
            filtered = filter_topology(question, topology, resolution)

            original_count = len(topology.triples)
            filtered_count = len(filtered.triples)
            trace_event(
                "TOPOLOGY_FILTER_DONE",
                "ok",
                f"{original_count} → {filtered_count} triples",
            )
            logger.info(
                "Topology filtered: %d → %d triples, %d → %d labels",
                original_count,
                filtered_count,
                len(topology.labels),
                len(filtered.labels),
            )

            # ── Step 2: Schema reasoning (ambiguity detection + resolution) ─
            schema_context = ""
            property_mappings = state.get("property_mappings", [])

            try:
                # Build synonym maps for reasoning
                label_synonym_map = build_synonym_map(
                    topology.label_names,
                    concept_nlp_terms=topology.nlp_terms_by_label,
                )
                property_synonym_map = build_property_synonym_map(
                    topology,
                    semantic_layer=semantic_layer,
                )
                relationship_synonym_map = build_relationship_synonym_map(
                    topology,
                    semantic_layer=semantic_layer,
                )

                # Detect ambiguities
                ambiguities = detect_ambiguities(
                    question=question,
                    topology=topology,
                    label_synonym_map=label_synonym_map,
                    property_synonym_map=property_synonym_map,
                    relationship_synonym_map=relationship_synonym_map,
                )

                resolved_ambiguities = None

                if ambiguities:
                    amb_summary = ", ".join(
                        f"'{a.user_term}' ({len(a.candidates)} candidates)"
                        for a in ambiguities[:3]
                    )
                    trace_event(
                        "SCHEMA_AMBIGUITY",
                        "info",
                        f"{len(ambiguities)} ambiguity(ies): {amb_summary}",
                    )
                    logger.info(
                        "Schema ambiguity detected: %d term(s): %s",
                        len(ambiguities),
                        amb_summary,
                    )

                    # Resolve via LLM (only when ambiguous)
                    if llm is not None:
                        resolved_ambiguities = await resolve_ambiguity(
                            question=question,
                            ambiguities=ambiguities,
                            topology=topology,
                            llm=llm,
                        )
                        resolved_summary = ", ".join(
                            f"'{r.user_term}'→{r.kind}:{r.element_name}"
                            for r in resolved_ambiguities[:3]
                        )
                        trace_event(
                            "SCHEMA_RESOLVED",
                            "ok",
                            f"Resolved: {resolved_summary}",
                        )
                        logger.info(
                            "Schema ambiguity resolved: %s",
                            resolved_summary,
                        )
                    else:
                        logger.warning(
                            "Schema ambiguity detected but no LLM available for resolution"
                        )

                # Build schema context string
                schema_context = build_schema_context(
                    property_mappings=property_mappings,
                    resolved_ambiguities=resolved_ambiguities,
                    topology=topology,
                )

                if schema_context:
                    trace_event(
                        "SCHEMA_CONTEXT",
                        "ok",
                        f"Schema context: {schema_context[:120]}",
                    )
                    logger.info("Schema context built (%d chars)", len(schema_context))

            except Exception as exc:
                logger.debug(
                    "Schema reasoning failed (non-fatal, continuing without): %s",
                    exc,
                )

            # Serialize filtered topology
            from src.graph.schema_cache import _topology_to_json
            filtered_json = _topology_to_json(filtered)

            return {
                "filtered_topology_json": filtered_json,
                "schema_context": schema_context,
            }

        except Exception as exc:
            logger.error(
                "Topology filtering failed: %s",
                exc,
                exc_info=True,
            )
            trace_event("TOPOLOGY_FILTER_FAIL", "fail", str(exc)[:120])
            # Fallback: pass full topology through
            return {
                "filtered_topology_json": state.get("full_topology_json", "{}"),
                "schema_context": "",
            }

    return topology_filter_node
