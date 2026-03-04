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
import json
import logging
from typing import TYPE_CHECKING

import redis.asyncio as aioredis

from app.core.exceptions import SchemaUnavailableError
from app.graph.topology import GraphTopology, LabelInfo, RelationshipTriple, extract_topology

if TYPE_CHECKING:
    from langchain_neo4j import Neo4jGraph

logger = logging.getLogger(__name__)

_SCHEMA_REDIS_KEY = "neo4j:schema"
_TOPOLOGY_REDIS_KEY = "neo4j:topology"


# ── Topology serialization helpers ────────────────────────────────────────────

def _topology_to_json(topology: GraphTopology) -> str:
    return json.dumps({
        "labels": [
            {
                "label": li.label,
                "properties": li.properties,
                "display_property": li.display_property,
                "sample_values": li.sample_values,
            }
            for li in topology.labels
        ],
        "triples": [
            {"source_label": t.source_label, "rel_type": t.rel_type, "target_label": t.target_label}
            for t in topology.triples
        ],
        "chains": [
            [{"source_label": t.source_label, "rel_type": t.rel_type, "target_label": t.target_label}
             for t in chain]
            for chain in topology.chains
        ],
    })


def _topology_from_json(raw: str) -> GraphTopology:
    data = json.loads(raw)
    labels = [LabelInfo(**li) for li in data["labels"]]
    triples = [RelationshipTriple(**t) for t in data["triples"]]
    chains = [[RelationshipTriple(**t) for t in chain] for chain in data["chains"]]
    return GraphTopology(labels=labels, triples=triples, chains=chains)


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
        self._cached_topology: GraphTopology | None = None  # in-memory fallback
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

    async def get_topology(self) -> GraphTopology:
        """
        Return the cached ``GraphTopology``, extracting from Neo4j on first call.

        Tries Redis first; falls back to the in-memory copy; extracts fresh if
        neither is available.
        """
        # Try Redis
        try:
            raw = await self._redis.get(_TOPOLOGY_REDIS_KEY)
            if raw:
                return _topology_from_json(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception as exc:
            logger.warning("Redis topology cache read failed: %s", exc)

        # Try in-memory fallback
        if self._cached_topology is not None:
            return self._cached_topology

        # Extract fresh (also caches it)
        await self._fetch_and_cache()
        if self._cached_topology is None:
            logger.warning("Topology unavailable — returning empty topology.")
            return GraphTopology()
        return self._cached_topology

    async def invalidate(self) -> None:
        """Delete the cached schema and topology keys so the next call re-fetches."""
        try:
            await self._redis.delete(_SCHEMA_REDIS_KEY, _TOPOLOGY_REDIS_KEY)
            logger.info("Schema cache invalidated.")
        except Exception as exc:
            logger.warning("Failed to invalidate schema cache: %s", exc)
        self._cached_topology = None

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

        # Cache schema in Redis — best-effort.
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

        # Extract and cache topology on every schema refresh
        try:
            topology = await extract_topology(self._graph)
            self._cached_topology = topology
            try:
                await self._redis.setex(_TOPOLOGY_REDIS_KEY, self._ttl, _topology_to_json(topology))
                logger.info("Topology cached in Redis (%d labels, %d triples).",
                            len(topology.labels), len(topology.triples))
            except Exception as exc:
                logger.warning("Redis topology write failed (in-memory fallback): %s", exc)
        except Exception as exc:
            logger.warning("Topology extraction failed (non-fatal): %s", exc)

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
