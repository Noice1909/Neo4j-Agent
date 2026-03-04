"""
Graph query tool — thin wrappers for LangChain agent and FastMCP.

The core business logic lives in ``src.services.graph_query.run_graph_query()``.
"""
from __future__ import annotations

from langchain.tools import tool
from langchain_core.language_models import BaseChatModel

from src.graph.schema_cache import SchemaCache
from src.services.graph_query import run_graph_query


def build_graph_query_tool(llm: BaseChatModel, schema_cache: SchemaCache):
    """Create the LangChain @tool with bound dependencies and return it."""

    @tool
    async def query_graph_tool(question: str) -> str:
        """
        Query the Neo4j knowledge graph using natural language.

        Use this tool to answer questions about data stored in the graph database.
        Translate the natural language question into Cypher, execute it against
        Neo4j (read-only), and return a human-readable answer.

        If the question uses pronouns or references previous results, use
        conversation context to resolve them before querying.

        Args:
            question: A natural-language question about the graph data.
        """
        return await run_graph_query(question, llm, schema_cache)

    return query_graph_tool


query_graph_tool = None


def register_mcp_tool(mcp, llm: BaseChatModel, schema_cache: SchemaCache) -> None:
    """Register the graph-query tool on a ``FastMCP`` server instance."""

    @mcp.tool(
        name="query_graph",
        description=(
            "Query the Neo4j knowledge graph using natural language. "
            "Translates the question into Cypher (read-only), executes it, "
            "and returns a human-readable answer. "
            "Always attempt to answer directly — never ask for clarification."
        ),
    )
    async def _mcp_query_graph(question: str) -> str:  # noqa: F841
        return await run_graph_query(question, llm, schema_cache)
