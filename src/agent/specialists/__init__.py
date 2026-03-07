"""
Specialist agents for the multi-agent pipeline.

Each agent module exports a `build_*_node()` function that creates an async
node function with injected dependencies (llm, schema_cache, graph, settings).

The pipeline subgraph wires these agents together in sequence:
  1. Coreference Resolution
  2. Entity Resolution
  3. Topology Filter
  4. Cypher Generation
  5. Cypher Validation
  6. Cypher Execution
  7. Result Verification
  8. Retry Decision
  9. Synthesis
"""
from __future__ import annotations

from src.agent.specialists.coreference import build_coreference_node
from src.agent.specialists.entity_resolution import build_entity_resolution_node
from src.agent.specialists.topology_filter import build_topology_filter_node
from src.agent.specialists.cypher_generation import build_cypher_generation_node
from src.agent.specialists.cypher_validation import build_cypher_validation_node
from src.agent.specialists.cypher_execution import build_cypher_execution_node
from src.agent.specialists.result_verification import build_result_verification_node
from src.agent.specialists.retry_decision import build_retry_decision_node
from src.agent.specialists.synthesis import build_synthesis_node

__all__ = [
    "build_coreference_node",
    "build_entity_resolution_node",
    "build_topology_filter_node",
    "build_cypher_generation_node",
    "build_cypher_validation_node",
    "build_cypher_execution_node",
    "build_result_verification_node",
    "build_retry_decision_node",
    "build_synthesis_node",
]
