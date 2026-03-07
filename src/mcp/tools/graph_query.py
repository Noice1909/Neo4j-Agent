"""
Graph query tool — MCP integration for multi-agent system.

Since the old tool-based approach has been replaced with the multi-agent supervisor,
this module now provides a simplified MCP interface that invokes the supervisor directly.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage


def register_mcp_tool(mcp) -> None:
    """
    Register the graph-query tool on a FastMCP server instance.

    Note: No longer needs llm or schema_cache — the supervisor agent has them.
    """
    from src.core.dependencies import get_agent

    @mcp.tool(
        name="query_graph",
        description=(
            "Query the Neo4j knowledge graph using natural language. "
            "Translates the question into Cypher (read-only), executes it, "
            "and returns a human-readable answer. "
            "Always attempt to answer directly — never ask for clarification."
        ),
    )
    async def _mcp_query_graph(question: str) -> str:
        """
        MCP tool: invokes the supervisor agent with a stateless query.

        Since MCP has no session_id, we use a fresh state for each query.
        No conversation history is maintained across MCP calls.
        """
        agent = get_agent()

        # Build minimal state for supervisor
        state = {
            "messages": [HumanMessage(content=question)],
        }

        # Invoke supervisor (no session persistence for MCP)
        result = await agent.ainvoke(state)

        # Extract final answer from AIMessage
        final_message = result.get("messages", [])[-1]
        return str(final_message.content)

    _ = _mcp_query_graph  # registered by @mcp.tool decorator
