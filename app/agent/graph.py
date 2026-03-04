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
from app.agent.trimming import trim_conversation
from app.core.tracing import trace_event

logger = logging.getLogger(__name__)


# ── Graph builder ─────────────────────────────────────────────────────────────


def build_agent_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver,
    *,
    max_conversation_tokens: int = 100_000,
    token_budget_reserve: int = 4096,
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
    max_conversation_tokens:
        Maximum token budget for conversation history sent to the LLM.
    token_budget_reserve:
        Tokens reserved for model output, subtracted from the budget.

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
    effective_budget = max(max_conversation_tokens - token_budget_reserve, 1024)

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
        "information — never expose error details or suggest query formatting.\n"
        "- The system may automatically correct minor typos or alternate names "
        "in the user's question. Trust and use the corrected form when querying."
    ))

    # ── Nodes ─────────────────────────────────────────────────────────────────

    async def call_model(state: AgentState) -> dict:
        """Invoke LLM with the current message history and bound tools."""
        messages = state["messages"]
        # Inject system prompt if not already present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [_SYSTEM_PROMPT] + list(messages)

        # ── Trim to fit context window ────────────────────────────────────
        original_count = len(messages)
        messages = trim_conversation(messages, max_tokens=effective_budget)
        if len(messages) < original_count:
            logger.info(
                "Context window trim: %d → %d messages (budget=%d tokens).",
                original_count,
                len(messages),
                effective_budget,
            )
            trace_event(
                "CONTEXT_TRIM", "warn",
                f"{original_count} → {len(messages)} msgs (budget={effective_budget})",
            )

        logger.debug("Agent node: invoking model (messages=%d).", len(messages))
        trace_event("AGENT_LLM_CALL", "info", f"{len(messages)} messages")
        response = await model_with_tools.ainvoke(messages)

        # Trace the LLM decision (tool call vs final answer)
        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls:
            tool_names = ", ".join(tc.get("name", "?") for tc in tool_calls)
            trace_event("AGENT_DECISION", "info", f"Tool call(s): {tool_names}")
        else:
            trace_event("AGENT_DECISION", "ok", "Final answer")

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
