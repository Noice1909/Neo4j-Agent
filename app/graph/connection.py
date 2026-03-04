"""
Neo4j graph connection management with production resilience.

Features:
  • Startup retry loop — keeps retrying until Neo4j is reachable
  • ``ensure_connected()`` — verifies the driver is alive before each query;
    auto-reconnects when a stale / dead connection is detected
  • ``reconnect_graph()`` — tears down the old singleton and creates a fresh one

The ``Neo4jGraph`` singleton is accessed via ``get_graph()`` and is also
registered as a FastAPI dependency.

The Neo4j user configured via ``NEO4J_USER`` / ``NEO4J_PASSWORD`` **must** be a
read-only account (see README for DDL commands).  The ``CypherSafetyValidator``
in ``app/graph/cypher_safety.py`` acts as an additional defence-in-depth layer.
"""
from __future__ import annotations

import logging
import time

from langchain_neo4j import Neo4jGraph

from app.core.config import Settings

logger = logging.getLogger(__name__)

# URI scheme constants — avoids duplicating literal strings.
_NEO4J_TLS_SCHEME = "neo4j+s://"
_NEO4J_TLS_UNVERIFIED_SCHEME = "neo4j+ssc://"
_BOLT_TLS_SCHEME = "bolt+s://"
_BOLT_TLS_UNVERIFIED_SCHEME = "bolt+ssc://"

# Module-level singleton — set during app lifespan startup.
_graph: Neo4jGraph | None = None
# Keep a reference to the settings used for reconnection.
_settings: Settings | None = None


# ── URI helper ────────────────────────────────────────────────────────────────

def _resolve_uri(settings: Settings) -> str:
    """Apply optional TLS-verification rewrite and return the final URI."""
    uri = settings.neo4j_uri
    if settings.neo4j_skip_tls_verify:
        if uri.startswith(_NEO4J_TLS_SCHEME):
            uri = _NEO4J_TLS_UNVERIFIED_SCHEME + uri[len(_NEO4J_TLS_SCHEME):]
            logger.warning(
                "TLS cert verification DISABLED (neo4j+ssc://). "
                "Only use this for managed cloud instances (AuraDB)."
            )
        elif uri.startswith(_BOLT_TLS_SCHEME):
            uri = _BOLT_TLS_UNVERIFIED_SCHEME + uri[len(_BOLT_TLS_SCHEME):]
            logger.warning(
                "TLS cert verification DISABLED (bolt+ssc://). "
                "Only use this for managed cloud instances (AuraDB)."
            )
    else:
        if uri.startswith((_NEO4J_TLS_SCHEME, _BOLT_TLS_SCHEME)):
            logger.info(
                "TLS with full cert verification enabled. "
                "Set NEO4J_SKIP_TLS_VERIFY=true if you encounter certificate errors "
                "with a managed cloud instance (AuraDB)."
            )
    return uri


def _create_graph(settings: Settings) -> Neo4jGraph:
    """Instantiate a Neo4jGraph — pure factory, no retry logic."""
    uri = _resolve_uri(settings)
    graph = Neo4jGraph(
        url=uri,
        username=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
        sanitize=True,
        refresh_schema=True,
    )
    return graph


# ── Public API ────────────────────────────────────────────────────────────────


def init_graph(settings: Settings) -> Neo4jGraph:
    """
    Create and cache a ``Neo4jGraph`` singleton **with startup retry**.

    If Neo4j is unreachable the function retries up to
    ``settings.neo4j_startup_max_retries`` times with exponential back-off
    so the application can survive container-orchestration start-order races.
    """
    global _graph, _settings
    if _graph is not None:
        return _graph

    _settings = settings
    max_retries = settings.neo4j_startup_max_retries
    delay = settings.neo4j_startup_retry_delay

    logger.info("Initialising Neo4j connection to %s", settings.neo4j_uri)

    for attempt in range(1, max_retries + 1):
        try:
            _graph = _create_graph(settings)
            logger.info(
                "Neo4j connection established (database=%s)",
                settings.neo4j_database,
            )
            return _graph
        except Exception as exc:
            if attempt == max_retries:
                logger.error(
                    "Neo4j connection failed after %d attempts — giving up.",
                    max_retries,
                )
                raise
            backoff = min(delay * (2 ** (attempt - 1)), 30)  # cap at 30s
            logger.warning(
                "Neo4j attempt %d/%d failed: %s — retrying in %.1fs",
                attempt, max_retries, exc, backoff,
            )
            time.sleep(backoff)

    # Should never reach here, but satisfy type checker
    raise RuntimeError("Neo4j connection failed — exhausted all retries.")


def get_graph() -> Neo4jGraph:
    """Return the existing ``Neo4jGraph`` singleton (must call ``init_graph`` first)."""
    if _graph is None:
        raise RuntimeError(
            "Neo4jGraph has not been initialised. "
            "Ensure `init_graph()` is called during application lifespan startup."
        )
    return _graph


def ensure_connected() -> Neo4jGraph:
    """
    Verify the Neo4j driver is alive; reconnect transparently if it is not.

    Call this at the top of every request-handling path that touches Neo4j
    to get automatic recovery from connection drops, server restarts, and
    network partitions without crashing the application.

    Returns the (possibly freshly reconnected) ``Neo4jGraph`` instance.
    """
    global _graph

    if _graph is None:
        raise RuntimeError(
            "Neo4jGraph has not been initialised. "
            "Ensure `init_graph()` is called during application lifespan startup."
        )

    try:
        # Lightweight connectivity probe — uses the existing driver pool
        _graph.query("RETURN 1 AS ping")
        return _graph
    except Exception as exc:
        logger.warning("Neo4j connectivity check failed: %s — attempting reconnect.", exc)
        return reconnect_graph()


def reconnect_graph() -> Neo4jGraph:
    """
    Tear down the current singleton and create a fresh connection.

    Uses the ``Settings`` captured during ``init_graph()``.
    """
    global _graph

    if _settings is None:
        raise RuntimeError("Cannot reconnect — init_graph() was never called.")

    # Best-effort cleanup of old driver
    if _graph is not None:
        try:
            if hasattr(_graph, "_driver") and _graph._driver:
                _graph._driver.close()
        except Exception:
            logger.debug("Old Neo4j driver cleanup error (ignored)", exc_info=True)
        _graph = None

    max_retries = _settings.neo4j_startup_max_retries
    delay = _settings.neo4j_startup_retry_delay

    for attempt in range(1, max_retries + 1):
        try:
            _graph = _create_graph(_settings)
            logger.info("Neo4j reconnected successfully (attempt %d).", attempt)
            return _graph
        except Exception as exc:
            if attempt == max_retries:
                logger.error(
                    "Neo4j reconnection failed after %d attempts.", max_retries,
                )
                raise
            backoff = min(delay * (2 ** (attempt - 1)), 30)
            logger.warning(
                "Neo4j reconnect attempt %d/%d failed: %s — retrying in %.1fs",
                attempt, max_retries, exc, backoff,
            )
            time.sleep(backoff)

    raise RuntimeError("Neo4j reconnection failed — exhausted all retries.")


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
