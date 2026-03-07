"""
Graph Query Pipeline Subgraph

Wires the 9 specialist agents into a sequential pipeline with retry loop.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

from src.agent.state import PipelineState
from src.agent.specialists import (
    build_coreference_node,
    build_entity_resolution_node,
    build_topology_filter_node,
    build_cypher_generation_node,
    build_cypher_validation_node,
    build_cypher_execution_node,
    build_result_verification_node,
    build_retry_decision_node,
    build_synthesis_node,
)
from src.core.tracing import trace_event

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_neo4j import Neo4jGraph
    from langgraph.graph.state import CompiledStateGraph as CompiledGraph
    from src.core.config import Settings
    from src.graph.schema_cache import SchemaCache

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def build_pipeline_subgraph(
    llm: "BaseChatModel",
    schema_cache: "SchemaCache",
    graph: "Neo4jGraph",
    settings: "Settings",
) -> "CompiledGraph":
    """
    Build the graph query pipeline subgraph.

    Parameters
    ----------
    llm:
        Language model for agents.
    schema_cache:
        Schema cache for topology.
    graph:
        Neo4j graph connection.
    settings:
        Application settings.

    Returns
    -------
    CompiledGraph
        Compiled pipeline subgraph with 9 agents + fallback node.
    """
    # ── Build all specialist agent nodes ─────────────────────────────────────
    coreference_node = build_coreference_node(llm)
    entity_resolution_node = build_entity_resolution_node(llm, schema_cache, graph, settings)
    topology_filter_node = build_topology_filter_node(schema_cache)
    cypher_generation_node = build_cypher_generation_node(llm)
    cypher_validation_node = build_cypher_validation_node()
    cypher_execution_node = build_cypher_execution_node(graph)
    result_verification_node = build_result_verification_node(llm)
    retry_decision_node = build_retry_decision_node(llm)
    synthesis_node = build_synthesis_node(llm)

    # ── Fallback node for graceful degradation ───────────────────────────────
    async def fallback_node(state: PipelineState) -> dict:
        """Graceful degradation when pipeline fails."""
        error = state.get("validation_errors", [])
        if error:
            error_summary = "; ".join(error[:2])
        else:
            error_summary = state.get("execution_error", "Unknown error")

        trace_event("GRAPH_QUERY_FALLBACK", "fail", f"Graceful degradation: {error_summary[:100]}")
        logger.warning("Pipeline fallback triggered: %s", error_summary[:200])

        return {
            "final_answer": (
                "I'm sorry, I couldn't find that information right now. "
                "Could you try asking in a different way?"
            )
        }

    # ── Routing functions ─────────────────────────────────────────────────────
    def route_after_validation(state: PipelineState) -> str:
        """Route based on validation result."""
        if state.get("validation_passed"):
            return "execution"
        else:
            return "fallback"

    def route_after_retry_decision(state: PipelineState) -> str:
        """Route based on retry decision."""
        should_retry = state.get("should_retry", False)
        retry_count = state.get("retry_count", 0)

        if should_retry and retry_count <= MAX_RETRIES:
            # Loop back to cypher generation
            return "cypher_generation"
        else:
            # Proceed to synthesis
            return "synthesis"

    # ── Build the graph ───────────────────────────────────────────────────────
    pipeline = StateGraph(PipelineState)

    # Add nodes
    pipeline.add_node("coreference", coreference_node)
    pipeline.add_node("entity_resolution", entity_resolution_node)
    pipeline.add_node("topology_filter", topology_filter_node)
    pipeline.add_node("cypher_generation", cypher_generation_node)
    pipeline.add_node("cypher_validation", cypher_validation_node)
    pipeline.add_node("cypher_execution", cypher_execution_node)
    pipeline.add_node("result_verification", result_verification_node)
    pipeline.add_node("retry_decision", retry_decision_node)
    pipeline.add_node("synthesis", synthesis_node)
    pipeline.add_node("fallback", fallback_node)

    # Wire sequential flow
    pipeline.set_entry_point("coreference")
    pipeline.add_edge("coreference", "entity_resolution")
    pipeline.add_edge("entity_resolution", "topology_filter")
    pipeline.add_edge("topology_filter", "cypher_generation")
    pipeline.add_edge("cypher_generation", "cypher_validation")

    # Conditional routing after validation
    pipeline.add_conditional_edges(
        "cypher_validation",
        route_after_validation,
        {
            "execution": "cypher_execution",
            "fallback": "fallback",
        },
    )

    # Continue to verification
    pipeline.add_edge("cypher_execution", "result_verification")
    pipeline.add_edge("result_verification", "retry_decision")

    # Conditional routing after retry decision
    pipeline.add_conditional_edges(
        "retry_decision",
        route_after_retry_decision,
        {
            "cypher_generation": "cypher_generation",  # retry loop
            "synthesis": "synthesis",                   # proceed
        },
    )

    # Terminal nodes
    pipeline.add_edge("synthesis", END)
    pipeline.add_edge("fallback", END)

    compiled = pipeline.compile()
    logger.info("Pipeline subgraph compiled (9 agents + fallback)")
    return compiled
