"""
In-memory TTL-based Neo4j schema cache.

Why cache the schema?
─────────────────────
Every `GraphCypherQAChain` call injects the full graph schema into the prompt.
Calling `Neo4jGraph.refresh_schema()` (which hits the DB) on every request is
expensive and unnecessary.  This module stores the schema in an in-memory dict
with a configurable TTL, so the schema is only fetched when it expires.

Schema-change flow:
───────────────────
1. At app startup, the schema is fetched and stored in memory.
2. On cache miss (TTL expired), the schema is re-fetched.
3. `invalidate()` is provided for manual invalidation (e.g.
   after a known schema migration) — call it from an admin endpoint.
4. A background proactive-refresh task fires at 80 % of the TTL so there is
   never a cold-start delay during normal operation.

This implementation does NOT require Redis — it is fully self-contained.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from src.core.exceptions import SchemaUnavailableError

if TYPE_CHECKING:
    from langchain_neo4j import Neo4jGraph

logger = logging.getLogger(__name__)


class SchemaCache:
    """
    Async in-memory cache for the Neo4j graph schema string with TTL expiry.

    Parameters
    ----------
    graph:
        The ``Neo4jGraph`` instance used to re-fetch schema on miss.
    ttl_seconds:
        Cache TTL; schema is considered stale after this many seconds.
    """

    def __init__(
        self,
        graph: "Neo4jGraph",
        ttl_seconds: int = 300,
    ) -> None:
        self._graph = graph
        self._ttl = ttl_seconds
        self._cached_schema: str | None = None
        self._cached_at: float = 0.0  # monotonic timestamp
        self._refresh_task: asyncio.Task | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_schema(self) -> str:
        """
        Return the cached schema string, re-fetching from Neo4j on miss/expiry.

        Raises
        ------
        SchemaUnavailableError
            If Neo4j is unreachable.
        """
        if self._cached_schema and (time.monotonic() - self._cached_at) < self._ttl:
            return self._cached_schema

        return await self._fetch_and_cache()

    async def warm_up(self) -> str:
        """Force an immediate fetch and cache of the schema (call at startup)."""
        schema = await self._fetch_and_cache()
        self._schedule_proactive_refresh()
        return schema

    async def invalidate(self) -> None:
        """Clear the cached schema so the next call re-fetches from Neo4j."""
        self._cached_schema = None
        self._cached_at = 0.0
        logger.info("Schema cache invalidated.")

    async def stop_refresh_task(self) -> None:
        """Cancel the background proactive-refresh task (call at shutdown)."""
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        self._refresh_task = None

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _fetch_and_cache(self) -> str:
        """Fetch schema from Neo4j, store in memory, and return it."""
        try:
            loop = asyncio.get_running_loop()
            # Neo4jGraph.refresh_schema() is synchronous — run in thread pool.
            await loop.run_in_executor(None, self._graph.refresh_schema)
            schema: str = self._graph.schema
            if not schema:
                raise SchemaUnavailableError("Neo4j returned an empty schema.")
        except SchemaUnavailableError:
            raise
        except Exception as exc:
            logger.error("Failed to fetch Neo4j schema: %s", exc)
            raise SchemaUnavailableError(str(exc)) from exc

        self._cached_schema = schema
        self._cached_at = time.monotonic()
        logger.info(
            "Schema fetched from Neo4j and cached in memory (TTL=%ds, %d chars).",
            self._ttl,
            len(schema),
        )
        return schema

    def _schedule_proactive_refresh(self) -> None:
        """Schedule a background task to refresh at 80 % of TTL."""
        delay = int(self._ttl * 0.8)

        async def _loop() -> None:
            while True:
                await asyncio.sleep(delay)
                logger.debug("Proactive schema refresh triggered.")
                try:
                    await self._fetch_and_cache()
                except SchemaUnavailableError as exc:
                    logger.error("Proactive schema refresh failed: %s", exc)

        self._refresh_task = asyncio.create_task(_loop())
        logger.info("Schema proactive refresh scheduled every %d seconds.", delay)

