"""
Vector search tool — semantic similarity search over Neo4j node embeddings.

Prerequisites (Neo4j side):
  • Embeddings must be pre-computed and stored on nodes (e.g. via APOC or
    a separate embedding pipeline).
  • A vector index named `<NEO4J_VECTOR_INDEX>` must exist.

If the index is not configured, the tool raises `VectorSearchUnavailableError`
gracefully — the rest of the agent continues to work.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_neo4j import Neo4jVector

from app.core.exceptions import VectorSearchUnavailableError

logger = logging.getLogger(__name__)

# Configurable via env — default matches common Neo4j vector index name.
_DEFAULT_INDEX = "vector"
_DEFAULT_K = 5


async def run_vector_search(
    query: str,
    llm: BaseChatModel,
    index_name: str = _DEFAULT_INDEX,
    k: int = _DEFAULT_K,
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    neo4j_database: str | None = None,
) -> str:
    """
    Run a semantic similarity search over Neo4j node embeddings.

    Parameters
    ----------
    query:
        Natural-language search query.
    llm:
        LLM used to generate the query embedding.
    index_name:
        Name of the Neo4j vector index to search.
    k:
        Number of results to return.

    Returns
    -------
    str
        Formatted list of similar nodes and their content.

    Raises
    ------
    VectorSearchUnavailableError
        If the vector index is missing or embeddings are not configured.
    """
    import asyncio

    try:
        # Neo4jVector requires an embedding model — use the LLM's embedding
        # or a dedicated embedding model if configured.
        from langchain_ollama import OllamaEmbeddings
        from app.core.config import get_settings

        # Extract base_url from llm if possible
        base_url = getattr(llm, "base_url", "http://localhost:11434")
        model_name = getattr(llm, "model", "mistral:latest")

        # Get Neo4j connection params from settings if not provided
        settings = get_settings()
        _uri = neo4j_uri or settings.neo4j_uri
        _user = neo4j_user or settings.neo4j_user
        _pwd = neo4j_password or settings.neo4j_password
        _db = neo4j_database or settings.neo4j_database

        embeddings = OllamaEmbeddings(base_url=base_url, model=model_name)

        def _search() -> list:
            vector_store = Neo4jVector.from_existing_index(
                embedding=embeddings,
                url=_uri,
                username=_user,
                password=_pwd,
                database=_db,
                index_name=index_name,
            )
            return vector_store.similarity_search(query, k=k)

        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(None, _search)

        if not docs:
            return f"No results found for: {query!r}"

        lines = [f"Found {len(docs)} similar node(s):"]
        for i, doc in enumerate(docs, 1):
            lines.append(f"\n[{i}] {doc.page_content}")
            if doc.metadata:
                lines.append(f"    Metadata: {doc.metadata}")

        return "\n".join(lines)

    except Exception as exc:
        logger.warning("Vector search failed (index=%s): %s", index_name, exc)
        raise VectorSearchUnavailableError(
            f"Vector search unavailable. Ensure a vector index named '{index_name}' "
            f"exists in Neo4j with pre-computed embeddings. Error: {exc}"
        ) from exc


def build_vector_search_tool(llm: BaseChatModel, index_name: str = _DEFAULT_INDEX):
    """Create the LangChain @tool with bound dependencies and return it."""

    @tool
    async def vector_search_tool(query: str) -> str:
        """
        Semantic similarity search over Neo4j node embeddings.

        Use this tool to find graph nodes that are semantically similar to
        the query — useful when you don't know exact property values.
        Falls back gracefully if a vector index is not configured.

        Args:
            query: Natural-language description of what to search for.
        """
        try:
            return await run_vector_search(query, llm, index_name=index_name)
        except VectorSearchUnavailableError:
            return (
                "Vector search is not available in this database configuration. "
                "Use the query_graph tool instead."
            )

    return vector_search_tool


# Module-level placeholder — replaced by `build_vector_search_tool()` in lifespan.
vector_search_tool = None


def register_mcp_tool(mcp, llm: BaseChatModel, index_name: str = _DEFAULT_INDEX) -> None:
    """Register the vector-search tool on a `FastMCP` server instance."""

    @mcp.tool(
        name="vector_search",
        description=(
            "Semantic similarity search over Neo4j node embeddings. "
            "Returns graph nodes similar to the query. "
            "Requires a pre-configured vector index."
        ),
    )
    async def _mcp_vector_search(query: str, k: int = _DEFAULT_K) -> str:  # noqa: F841
        try:
            return await run_vector_search(query, llm, index_name=index_name, k=k)
        except VectorSearchUnavailableError as exc:
            return f"Vector search unavailable: {exc}"
