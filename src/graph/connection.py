"""
Neo4j graph connection management.

Provides a module-level singleton `Neo4jGraph` instance initialised from
`Settings`.  The graph is accessed through `get_graph()` which is also
registered as a FastAPI dependency in `app/deps.py`.

The Neo4j user configured via `NEO4J_USER` / `NEO4J_PASSWORD` **must** be a
read-only account (see README for DDL commands).  The `CypherSafetyValidator`
in `app/graph/cypher_safety.py` acts as an additional defence-in-depth layer.
"""
from __future__ import annotations

import logging

from langchain_neo4j import Neo4jGraph

from src.core.config import Settings

logger = logging.getLogger(__name__)

# Module-level singleton — set during app lifespan startup.
_graph: Neo4jGraph | None = None


def init_graph(settings: Settings) -> Neo4jGraph:
    """
    Create and cache a `Neo4jGraph` singleton.

    Parameters
    ----------
    settings:
        Application settings (injected from `get_settings()`).

    Returns
    -------
    Neo4jGraph
        A connected, schema-aware Neo4j graph instance.
    """
    global _graph
    if _graph is not None:
        return _graph

    logger.info("Initialising Neo4j connection to %s", settings.neo4j_uri)

    # AuraDB (neo4j+s://) uses TLS but the certificate chain can fail Python's
    # SSL verification in some environments.  neo4j+ssc:// keeps full TLS
    # encryption while skipping the certificate chain check.
    # This rewrite is OPT-IN via NEO4J_SKIP_TLS_VERIFY=true to avoid silently
    # downgrading security in production self-hosted deployments.
    uri = settings.neo4j_uri
    if settings.neo4j_skip_tls_verify:
        if uri.startswith("neo4j+s://"):
            uri = "neo4j+ssc://" + uri[len("neo4j+s://"):]
            logger.warning(
                "TLS cert verification DISABLED (neo4j+ssc://). "
                "Only use this for managed cloud instances (AuraDB)."
            )
        elif uri.startswith("bolt+s://"):
            uri = "bolt+ssc://" + uri[len("bolt+s://"):]
            logger.warning(
                "TLS cert verification DISABLED (bolt+ssc://). "
                "Only use this for managed cloud instances (AuraDB)."
            )
    else:
        if uri.startswith(("neo4j+s://", "bolt+s://")):
            logger.info(
                "TLS with full cert verification enabled. "
                "Set NEO4J_SKIP_TLS_VERIFY=true if you encounter certificate errors "
                "with a managed cloud instance (AuraDB)."
            )

    _graph = Neo4jGraph(
        url=uri,
        username=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
        sanitize=True,
        refresh_schema=True,
    )
    logger.info("Neo4j connection established (database=%s)", settings.neo4j_database)
    return _graph


def get_graph() -> Neo4jGraph:
    """Return the existing `Neo4jGraph` singleton (must call `init_graph` first)."""
    if _graph is None:
        raise RuntimeError(
            "Neo4jGraph has not been initialised. "
            "Ensure `init_graph()` is called during application lifespan startup."
        )
    return _graph


def close_graph() -> None:
    """Release the Neo4j driver connection (called during lifespan shutdown)."""
    global _graph
    if _graph is not None:
        try:
            # Neo4jGraph wraps the neo4j driver; close the underlying driver.
            if hasattr(_graph, "_driver") and _graph._driver:
                _graph._driver.close()
        except Exception:
            logger.debug("Neo4j driver cleanup error (ignored)", exc_info=True)
        _graph = None
        logger.info("Neo4j connection closed.")
