"""
LangGraph agent StateGraph definition.

Graph topology:
    START → [agent] → (tool_calls?) → [tools] → [agent] → … → END

The agent node invokes the LLM with tool bindings.
The tools node executes whichever tools the LLM requested.
The conditional edge loops back until the LLM produces a final answer.

The compiled graph is stateful per `thread_id` (= `session_id`) via
the Redis checkpointer.  Every invocation with the same thread_id continues
the conversation where it left off — across reboots.

Plug-and-play:
  - Replace the tool list in `build_agent_graph()` to add/remove capabilities.
  - Replace the checkpointer arg to swap persistence backends.
  - Replace the llm arg to swap LLM providers.
"""
from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent.state import AgentState

logger = logging.getLogger(__name__)


# ── Graph builder ─────────────────────────────────────────────────────────────


def build_agent_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver,
) -> "CompiledGraph":  # type: ignore[name-defined]
    """
    Compile and return the LangGraph agent.

    Parameters
    ----------
    llm:
        Any `BaseChatModel` with tool-calling support (ChatOllama, ChatOpenAI, …).
    tools:
        List of LangChain `BaseTool` instances the agent may call.
    checkpointer:
        LangGraph checkpoint saver for persistent session memory.

    Returns
    -------
    CompiledGraph
        The compiled LangGraph application. Call with::

            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=user_message)]},
                config={"configurable": {"thread_id": session_id}},
            )
    """
    model_with_tools = llm.bind_tools(tools)

    # System prompt: hide all implementation details from the user
    _SYSTEM_PROMPT = SystemMessage(content=(
        "You are a helpful and knowledgeable assistant. "
        "You can answer questions about movies, actors, directors, and their "
        "relationships. You have access to tools that help you find information.\n\n"
        "CRITICAL RULES:\n"
        "- NEVER mention databases, queries, Cypher, Neo4j, graphs, schemas, "
        "nodes, relationships, or any technical implementation details.\n"
        "- NEVER tell the user to 'use specific names', 'avoid pronouns', or "
        "'rephrase for the query'. The user must not know how you find answers.\n"
        "- If the user's question is vague or uses pronouns like 'those', 'them', "
        "'they', you MAY ask a brief, natural clarifying question like: "
        "'Which movies do you mean?' or 'Could you tell me which ones you're "
        "referring to?' — but NEVER explain WHY you need clarification in "
        "technical terms.\n"
        "- When you have enough context, use your tools and respond with a "
        "direct, conversational answer.\n"
        "- If something goes wrong internally, just say you couldn't find the "
        "information — never expose error details or suggest query formatting."
    ))

    # ── Nodes ─────────────────────────────────────────────────────────────────

    async def call_model(state: AgentState) -> dict:
        """Invoke LLM with the current message history and bound tools."""
        messages = state["messages"]
        # Inject system prompt if not already present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [_SYSTEM_PROMPT] + list(messages)
        logger.debug("Agent node: invoking model (messages=%d).", len(messages))
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools=tools)

    # ── Routing ───────────────────────────────────────────────────────────────

    def should_continue(state: AgentState) -> str:
        """Route to 'tools' if the LLM issued tool calls, otherwise END."""
        last = state["messages"][-1]
        # tool_calls is present on AIMessage but not on BaseMessage — check dynamically
        tool_calls = getattr(last, "tool_calls", None)
        if tool_calls:
            return "tools"
        return END

    # ── Graph wiring ──────────────────────────────────────────────────────────

    graph = StateGraph(AgentState)

    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("LangGraph agent compiled (%d tool(s) registered).", len(tools))
    return compiled
