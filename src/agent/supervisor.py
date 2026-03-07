"""
Supervisor Agent

Top-level orchestrator that classifies questions and routes to appropriate specialists:
- graph_query: Graph query pipeline (9-agent subgraph)
- schema_info: Schema information
- vector_search: Semantic similarity search
- direct: Direct conversational answer

Replaces the old single-agent + tool-calling approach.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import END, StateGraph

from src.agent.graph import build_system_prompt
from src.agent.pipeline import build_pipeline_subgraph
from src.agent.state import PipelineState
from src.agent.trimming import trim_conversation
from src.core.exceptions import VectorSearchUnavailableError
from src.core.tracing import trace_event

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_neo4j import Neo4jGraph
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph as CompiledGraph
    from src.core.config import Settings
    from src.graph.schema_cache import SchemaCache
    from src.graph.topology import GraphTopology

logger = logging.getLogger(__name__)


def build_supervisor_graph(
    llm: "BaseChatModel",
    checkpointer: "BaseCheckpointSaver",
    schema_cache: "SchemaCache",
    topology: "GraphTopology",
    graph: "Neo4jGraph",
    settings: "Settings",
) -> "CompiledGraph":
    """
    Build the supervisor agent graph.

    Parameters
    ----------
    llm:
        Language model for all agents.
    checkpointer:
        Checkpointer for session persistence.
    schema_cache:
        Schema cache.
    topology:
        Graph topology (for system prompt).
    graph:
        Neo4j graph connection.
    settings:
        Application settings.

    Returns
    -------
    CompiledGraph
        Compiled supervisor graph with routing to all specialists.
    """
    # ── Build pipeline subgraph ───────────────────────────────────────────────
    pipeline_subgraph = build_pipeline_subgraph(llm, schema_cache, graph, settings)

    # ── Build system prompt ───────────────────────────────────────────────────
    system_prompt = build_system_prompt(
        schema_labels=topology.label_names,
        label_descriptions={
            li.label: li.description
            for li in topology.labels
            if li.description
        },
    )

    # ── Supervisor node ───────────────────────────────────────────────────────
    async def supervisor_node(state: PipelineState) -> dict:
        """
        Classify the question and set routing.

        Reads
        -----
        - state["messages"]: Conversation messages

        Returns
        -------
        dict
            State update with:
            - "route": Classification ("graph_query" | "schema_info" | "vector_search" | "direct")
            - "user_question": Extracted question text
            - "conversation_context": Recent messages for context
            - "messages": Trimmed conversation (for memory management)
        """
        messages = state.get("messages", [])

        # Trim conversation to fit context window
        max_tokens = settings.max_conversation_tokens - settings.token_budget_reserve
        trimmed_messages = trim_conversation(messages, max_tokens=max_tokens)

        # Extract last user message
        user_question = ""
        for msg in reversed(trimmed_messages):
            if isinstance(msg, HumanMessage):
                user_question = str(msg.content)
                break

        trace_event("SUPERVISOR_START", "info", f"Question: {user_question[:100]}")

        # Build conversation context (last 3 exchanges)
        conversation_context = ""
        recent_messages = trimmed_messages[-6:] if len(trimmed_messages) >= 6 else trimmed_messages
        for msg in recent_messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            conversation_context += f"{role}: {str(msg.content)[:200]}\n"

        # Build dynamic examples from schema (domain-agnostic)
        example_labels = topology.label_names[:3] if len(topology.label_names) >= 3 else topology.label_names
        data_query_examples = []

        # Add label-based examples
        if example_labels:
            data_query_examples.append(f"'How many {example_labels[0]} are there?'")
            if len(example_labels) > 1:
                data_query_examples.append(f"'List all {example_labels[1]}'")
            if len(topology.triples) > 0:
                triple = topology.triples[0]
                data_query_examples.append(f"'Find {triple.source_label} connected to {triple.target_label}'")

        # Add property-based examples (to help classify property queries)
        property_examples = []
        for li in topology.labels[:2]:  # Use first 2 labels
            if li.properties:
                # Pick first property as example
                prop = li.properties[0]
                property_examples.append(f"'What are the most popular {prop}?'")
                break

        if property_examples:
            data_query_examples.extend(property_examples)

        # Pre-classification: Check if question mentions any properties
        # If it does, it's very likely a graph_query, not schema_info
        all_properties = set()
        for li in topology.labels:
            all_properties.update(li.properties)

        question_lower = user_question.lower()
        mentions_property = any(prop.lower() in question_lower for prop in all_properties if len(prop) > 3)

        # Classification prompt (fully generic, property-aware)
        classification_prompt = (
            "Given the user's message, classify it into one of these categories:\n\n"
            "- 'graph_query': Questions asking for SPECIFIC DATA, entities, relationships, counts, listings, or property values.\n"
            "  This includes questions about property values like 'most popular X', 'all Y', 'find Z'.\n"
            f"  Examples: {', '.join(data_query_examples) if data_query_examples else 'Questions about specific entities or data'}\n\n"
            "- 'schema_info': Questions asking WHAT TYPES of data exist or what labels/properties are available.\n"
            "  Only classify as schema_info if asking about the schema itself, not data values.\n"
            "  Examples: 'What types of entities exist?', 'What labels are available?', 'What data do you have?'\n\n"
            "- 'vector_search': Requests for semantic similarity or 'similar to' searches.\n"
            "  Examples: 'Find things similar to X', 'What's related to Y?'\n\n"
            "- 'direct': Greetings, thank you, clarifications, or general conversation.\n"
            "  Examples: 'Hello', 'Thank you', 'What can you do?'\n\n"
            f"User message: {user_question}\n\n"
        )

        # Add hint if property is mentioned
        if mentions_property:
            classification_prompt += (
                "IMPORTANT: This question mentions a property name, so it's asking for DATA, not schema info.\n\n"
            )

        classification_prompt += "Respond with ONLY the category name (no explanation)."

        # Invoke LLM for classification
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: llm.invoke(classification_prompt),
        )

        response_text = str(response.content).strip().lower()

        # Extract route
        if "graph_query" in response_text:
            route = "graph_query"
        elif "schema_info" in response_text:
            route = "schema_info"
        elif "vector_search" in response_text:
            route = "vector_search"
        elif "direct" in response_text:
            route = "direct"
        else:
            # Default to graph_query
            route = "graph_query"

        # Override: If question mentions a property, it's definitely graph_query
        # This prevents misclassification of property queries as schema_info
        if mentions_property and route == "schema_info":
            logger.info(
                "Overriding schema_info → graph_query (question mentions property)"
            )
            trace_event("CLASSIFICATION_OVERRIDE", "info", "Property mention detected")
            route = "graph_query"

        trace_event("SUPERVISOR_ROUTE", "info", f"Route: {route}")
        logger.info("Supervisor classified question as: %s", route)

        return {
            "route": route,
            "user_question": user_question,
            "conversation_context": conversation_context,
            "messages": trimmed_messages,
        }

    # ── Schema info node ──────────────────────────────────────────────────────
    async def schema_info_node(state: PipelineState) -> dict:
        """Return schema information."""
        trace_event("SCHEMA_INFO_START", "info", "Fetching schema")
        schema = await schema_cache.get_schema()

        # Format schema nicely
        answer = (
            f"I can help you with information about {len(topology.label_names)} types of entities:\n\n"
            + "\n".join(f"- {label}" for label in topology.label_names[:15])
        )
        if len(topology.label_names) > 15:
            answer += f"\n...and {len(topology.label_names) - 15} more."

        trace_event("SCHEMA_INFO_DONE", "ok", f"{len(topology.label_names)} labels")
        return {"final_answer": answer}

    # ── Vector search node ────────────────────────────────────────────────────
    async def vector_search_node(state: PipelineState) -> dict:
        """Semantic similarity search (placeholder)."""
        trace_event("VECTOR_SEARCH_START", "info", "Vector search requested")
        # Placeholder — vector search would require vector index setup
        raise VectorSearchUnavailableError(
            "Vector search is not configured. Please set up a vector index in Neo4j."
        )

    # ── Direct answer node ────────────────────────────────────────────────────
    async def direct_answer_node(state: PipelineState) -> dict:
        """Direct conversational answer without tools."""
        question = state.get("user_question", "")
        trace_event("DIRECT_ANSWER_START", "info", f"Question: {question[:100]}")

        # Build prompt with system context
        messages_for_llm = [system_prompt, HumanMessage(content=question)]

        # Invoke LLM
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: llm.invoke(messages_for_llm),
        )

        answer = str(response.content).strip()
        trace_event("DIRECT_ANSWER_DONE", "ok", f"Answer: {answer[:100]}")
        return {"final_answer": answer}

    # ── Respond node ──────────────────────────────────────────────────────────
    async def respond_node(state: PipelineState) -> dict:
        """Wrap final answer as AIMessage."""
        final_answer = state.get("final_answer", "I couldn't process that request.")

        trace_event("SUPERVISOR_RESPOND", "ok", f"Final answer: {final_answer[:100]}")
        logger.info("Supervisor responding: %r", final_answer[:120])

        return {"messages": [AIMessage(content=final_answer)]}

    # ── Routing function ──────────────────────────────────────────────────────
    def route_from_supervisor(state: PipelineState) -> str:
        """Route based on classification."""
        return state.get("route", "graph_query")

    # ── Build the graph ───────────────────────────────────────────────────────
    supervisor = StateGraph(PipelineState)

    # Add nodes
    supervisor.add_node("supervisor", supervisor_node)
    supervisor.add_node("graph_query_pipeline", pipeline_subgraph)
    supervisor.add_node("schema_info", schema_info_node)
    supervisor.add_node("vector_search", vector_search_node)
    supervisor.add_node("direct_answer", direct_answer_node)
    supervisor.add_node("respond", respond_node)

    # Set entry point
    supervisor.set_entry_point("supervisor")

    # Conditional routing from supervisor
    supervisor.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "graph_query": "graph_query_pipeline",
            "schema_info": "schema_info",
            "vector_search": "vector_search",
            "direct": "direct_answer",
        },
    )

    # All specialists converge at respond
    supervisor.add_edge("graph_query_pipeline", "respond")
    supervisor.add_edge("schema_info", "respond")
    supervisor.add_edge("vector_search", "respond")
    supervisor.add_edge("direct_answer", "respond")

    # Respond terminates
    supervisor.add_edge("respond", END)

    # Compile with checkpointer
    compiled = supervisor.compile(checkpointer=checkpointer)
    logger.info("Supervisor graph compiled (1 supervisor + 4 routing paths)")
    return compiled
