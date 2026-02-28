"""Core tool implementations — framework-agnostic.
 
These async functions are the single source of truth for all tool logic.
Both the MCP server (@mcp.tool wrappers) and the LangGraph agent
(@tool wrappers) delegate to these implementations.

Dependency resolution happens via module-level accessors
(get_graph, get_compiled_agent, etc.) which are set during lifespan startup.
"""

from app.mcp.tools.graph_query import run_graph_query, query_graph_tool
from app.mcp.tools.schema_info import get_schema_info, schema_info_tool
from app.mcp.tools.vector_search import run_vector_search, vector_search_tool

__all__ = [
    "run_graph_query",
    "query_graph_tool",
    "get_schema_info",
    "schema_info_tool",
    "run_vector_search",
    "vector_search_tool",
]
