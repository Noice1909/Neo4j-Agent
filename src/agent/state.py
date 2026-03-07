"""
LangGraph agent state schema.
"""
from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    The state flowing through every node in the graph.

    ``messages`` uses the ``add_messages`` reducer — new messages are appended
    rather than overwriting the list, giving a persistent conversation history.
    """

    messages: Annotated[list[BaseMessage], add_messages]


class PipelineState(TypedDict, total=False):
    """
    State schema for the multi-agent pipeline (supervisor + 9 specialist agents).

    All fields are optional (total=False) to support incremental state updates
    and backward compatibility with checkpointer.
    """

    # ── Conversation ───────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]
    user_question: str
    conversation_context: str

    # ── Routing ────────────────────────────────────────────
    route: str  # "graph_query" | "schema_info" | "vector_search" | "direct"

    # ── Coreference Resolution Agent (2) ───────────────────
    coreferenced_question: str

    # ── Entity Resolution Agent (3) ────────────────────────
    entity_resolved_question: str
    resolution_corrections: list[dict]  # serialized Correction objects

    # ── Topology Filter Agent (4) ──────────────────────────
    filtered_topology_json: str         # serialized GraphTopology
    full_topology_json: str             # for validation agent

    # ── Cypher Generation Agent (5) ────────────────────────
    generated_cypher: str
    cypher_prompt_text: str             # for retry context
    topology_section: str               # rendered topology text

    # ── Cypher Validation Agent (6) ────────────────────────
    validation_passed: bool
    validation_errors: list[str]

    # ── Cypher Execution Agent (7) ─────────────────────────
    raw_results: str                    # JSON-serialized Neo4j results
    execution_succeeded: bool
    execution_error: str

    # ── Result Verification Agent (8) ──────────────────────
    results_valid: bool
    verification_message: str

    # ── Retry Decision Agent (9) ───────────────────────────
    should_retry: bool
    retry_count: int
    correction_guidance: str            # hints for next generation

    # ── Synthesis Agent (10) ───────────────────────────────
    final_answer: str

    # ── Error fallback ─────────────────────────────────────
    error_message: str
