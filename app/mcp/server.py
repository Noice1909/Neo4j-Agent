"""
FastMCP server instance and tool registration.

The `mcp` instance is created here and mounted on the FastAPI app in
`app/main.py` via `app.mount(settings.mcp_path, mcp.http_app(path="/"))`.

Tools are registered lazily via `register_all_tools()` which is called
during lifespan startup after all dependencies are ready.  This lets tool
implementations access the Neo4j graph, LLM, and schema cache which are
not available at module import time.

Plug-and-play:
  - Add a new tool: create `app/mcp/tools/my_tool.py`, implement
    `register_mcp_tool(mcp, ...)`, and call it in `register_all_tools()`.
  - Remove a tool: delete its file and remove its call from `register_all_tools()`.
  - Swap the MCP server: replace this file's `mcp` with a different server
    implementation that implements the same `http_app()` interface.
"""
from __future__ import annotations

import logging

from fastmcp import FastMCP

from app.core.config import Settings

logger = logging.getLogger(__name__)

# ── FastMCP server ─────────────────────────────────────────────────────────────
# stateless_http=True: no sticky sessions — safe for multiple Uvicorn workers.
mcp = FastMCP(
    name="neo4j-agent",
    instructions=(
        "A read-only Neo4j knowledge graph assistant. "
        "Use `get_schema` to discover the data model, then `query_graph` to "
        "answer questions. Use `vector_search` for semantic similarity queries."
    ),
)


def register_all_tools(settings: Settings) -> None:
    """
    Register all MCP tools on the server.

    Called once during application lifespan startup after Neo4jGraph,
    Redis schema cache, and LLM are all initialised.

    Parameters
    ----------
    settings:
        Application settings (used to thread config into tool registrations).
    """
    from app.graph.connection import get_graph  # noqa: F401 (import triggers validation)
    from app.llm.factory import get_llm_from_settings
    from app.mcp.tools import graph_query as gq_module
    from app.mcp.tools import schema_info as si_module
    from app.mcp.tools import vector_search as vs_module
    from app.core.dependencies import get_schema_cache_instance

    llm = get_llm_from_settings(settings)
    schema_cache = get_schema_cache_instance()

    gq_module.register_mcp_tool(mcp, llm, schema_cache)
    si_module.register_mcp_tool(mcp, schema_cache)
    vs_module.register_mcp_tool(mcp, llm)

    logger.info("FastMCP: all tools registered (query_graph, get_schema, vector_search).")
