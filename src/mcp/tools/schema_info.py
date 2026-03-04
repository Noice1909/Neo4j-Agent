"""
Schema info tool — exposes the Neo4j graph schema.

Useful when the agent or an MCP client needs to understand the data model
before formulating queries.  Always served from the Redis schema cache.
"""
from __future__ import annotations

import logging

from langchain.tools import tool

from src.graph.schema_cache import SchemaCache

logger = logging.getLogger(__name__)


async def get_schema_info(schema_cache: SchemaCache) -> str:
    """
    Return the current Neo4j graph schema (node labels, relationship types,
    and property keys).  The result is served from the Redis cache — no DB
    round-trip unless the cache is cold or expired.

    Parameters
    ----------
    schema_cache:
        The application schema cache instance.
    """
    schema = await schema_cache.get_schema()
    logger.debug("Schema info returned (%d chars).", len(schema))
    return schema


def build_schema_info_tool(schema_cache: SchemaCache):
    """Create the LangChain @tool with bound dependencies and return it."""

    @tool
    async def schema_info_tool() -> str:
        """
        Retrieve the Neo4j graph schema.

        Returns node labels, relationship types, and property keys present in
        the database. Call this tool first when you need to understand the data
        model before generating a query.
        """
        return await get_schema_info(schema_cache)

    return schema_info_tool


# Module-level placeholder — replaced by `build_schema_info_tool()` in lifespan.
schema_info_tool = None


def register_mcp_tool(mcp, schema_cache: SchemaCache) -> None:
    """Register the schema-info tool on a `FastMCP` server instance."""

    @mcp.tool(
        name="get_schema",
        description=(
            "Return the Neo4j graph schema: node labels, relationship types, "
            "and property keys.  Call this before formulating a graph query."
        ),
    )
    async def _mcp_get_schema() -> str:
        return await get_schema_info(schema_cache)

    _ = _mcp_get_schema  # registered by @mcp.tool decorator
