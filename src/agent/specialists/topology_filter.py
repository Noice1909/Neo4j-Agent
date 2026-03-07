"""
Agent 4: Topology Filter

Reduces the full graph topology to a question-relevant subset.
Deterministic agent — no LLM calls, pure filtering based on mentioned labels/entities.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from src.agent.state import PipelineState
from src.graph.cypher.topology_filter import filter_topology
from src.graph.cypher.entity_resolution import ResolutionResult
from src.core.tracing import trace_event

if TYPE_CHECKING:
    from src.graph.schema_cache import SchemaCache

logger = logging.getLogger(__name__)


def build_topology_filter_node(schema_cache: "SchemaCache"):
    """
    Build the topology filter agent node.

    Parameters
    ----------
    schema_cache:
        Schema cache for fetching full topology.

    Returns
    -------
    async function
        Node function for LangGraph: `(PipelineState) -> dict`
    """
    async def topology_filter_node(state: PipelineState) -> dict:
        """
        Filter topology to question-relevant subset.

        Reads
        -----
        - state["entity_resolved_question"]: Corrected question
        - state["resolution_corrections"]: Applied corrections
        - state["full_topology_json"]: Full topology (optional, will fetch if missing)

        Returns
        -------
        dict
            State update with:
            - "filtered_topology_json": Filtered topology (serialized)
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
            from dataclasses import fields
            from src.graph.cypher.entity_resolution import Correction

            corrections_objs = []
            if corrections:
                # Reconstruct Correction objects from serialized dicts
                for c_dict in corrections:
                    correction = Correction(
                        original=c_dict["original"],
                        corrected=c_dict["corrected"],
                        correction_type=c_dict["correction_type"],
                        metadata=c_dict.get("metadata", {}),
                    )
                    corrections_objs.append(correction)

            resolution = ResolutionResult(
                resolved_question=question,
                corrections=corrections_objs,
            )

            # Filter topology
            filtered = filter_topology(question, topology, resolution)

            # Log reduction
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

            # Serialize filtered topology
            from src.graph.schema_cache import _topology_to_json
            filtered_json = _topology_to_json(filtered)

            return {"filtered_topology_json": filtered_json}

        except Exception as exc:
            logger.error(
                "Topology filtering failed: %s",
                exc,
                exc_info=True,
            )
            trace_event("TOPOLOGY_FILTER_FAIL", "fail", str(exc)[:120])
            # Fallback: pass full topology through
            return {"filtered_topology_json": state.get("full_topology_json", "{}")}

    return topology_filter_node
