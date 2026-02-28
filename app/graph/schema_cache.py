"""
Redis-backed Neo4j schema cache.

Why cache the schema?
─────────────────────
Every `GraphCypherQAChain` call injects the full graph schema into the prompt.
Calling `Neo4jGraph.refresh_schema()` (which hits the DB) on every request is
expensive and unnecessary.  This module stores the schema in Redis with a
configurable TTL, so all workers share a single cached schema.

Schema-change flow:
───────────────────
1. At app startup, the schema is fetched and stored in Redis.
2. On cache miss (TTL expired), the schema is re-fetched.
3. `invalidate_schema_cache()` is provided for manual invalidation (e.g.
   after a known schema migration) — call it from an admin endpoint.
4. A background proactive-refresh task fires at 80 % of the TTL so there is
   never a cold-start delay during normal operation.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import redis.asyncio as aioredis

from app.core.exceptions import SchemaUnavailableError

if TYPE_CHECKING:
    from langchain_neo4j import Neo4jGraph

logger = logging.getLogger(__name__)

_SCHEMA_REDIS_KEY = "neo4j:schema"


class SchemaCache:
    """
    Async Redis-backed cache for the Neo4j graph schema string.

    Parameters
    ----------
    redis_client:
        An async Redis client (from `redis.asyncio`).
    graph:
        The `Neo4jGraph` instance used to re-fetch schema on miss.
    ttl_seconds:
        Redis key TTL; schema is considered stale after this many seconds.
    """

    def __init__(
        self,
        redis_client: aioredis.Redis,
        graph: "Neo4jGraph",
        ttl_seconds: int = 300,
    ) -> None:
        self._redis = redis_client
        self._graph = graph
        self._ttl = ttl_seconds
        self._refresh_task: asyncio.Task | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_schema(self) -> str:
        """
        Return the cached schema string, re-fetching from Neo4j on miss.

        Raises
        ------
        SchemaUnavailableError
            If both Redis and Neo4j are unreachable.
        """
        try:
            cached = await self._redis.get(_SCHEMA_REDIS_KEY)
            if cached:
                return cached.decode() if isinstance(cached, bytes) else cached
        except Exception as exc:
            logger.warning("Redis schema cache read failed: %s", exc)

        return await self._fetch_and_cache()

    async def warm_up(self) -> str:
        """Force an immediate fetch and cache of the schema (call at startup)."""
        schema = await self._fetch_and_cache()
        self._schedule_proactive_refresh()
        return schema

    async def invalidate(self) -> None:
        """Delete the cached schema key so the next call re-fetches from Neo4j."""
        try:
            await self._redis.delete(_SCHEMA_REDIS_KEY)
            logger.info("Schema cache invalidated.")
        except Exception as exc:
            logger.warning("Failed to invalidate schema cache: %s", exc)

    def stop_refresh_task(self) -> None:
        """Cancel the background proactive-refresh task (call at shutdown)."""
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _fetch_and_cache(self) -> str:
        """Fetch schema from Neo4j, store in Redis (best-effort), and return it."""
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

        # Cache in Redis — best-effort; if Redis is down, we still return the schema.
        try:
            await self._redis.setex(_SCHEMA_REDIS_KEY, self._ttl, schema)
            logger.info(
                "Schema fetched from Neo4j and cached in Redis (TTL=%ds, %d chars).",
                self._ttl,
                len(schema),
            )
        except Exception as exc:
            logger.warning(
                "Redis unavailable — schema cached in memory only (%d chars). Error: %s",
                len(schema),
                exc,
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
