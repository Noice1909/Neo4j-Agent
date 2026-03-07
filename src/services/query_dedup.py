"""
Query deduplication service — supports in-memory and Redis backends.

Two-layer strategy to reduce redundant LLM calls:

**Layer 1 — Response Cache (Case 1: already answered)**
    Caches agent-level responses keyed by normalised query text
    (not session-specific prompt).  Backend is either an in-memory TTL dict
    or Redis, depending on whether a ``redis_client`` is provided.

**Layer 2 — In-Flight Coalescing (Case 2: concurrent duplicates)**
    Uses ``asyncio.Future`` objects so concurrent identical queries share
    a single agent invocation.  Process-local (asyncio is single-threaded).

Normalisation pipeline:
    1. Unicode NFKC + lowercase
    2. Punctuation strip
    3. Whitespace collapse
    4. Stopword removal
    5. Token sort (word-order-independent)
    6. SHA-256 hash → fixed-length cache key
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import unicodedata
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# ── Normalisation helpers ────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "am", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "because",
    "but", "and", "or", "if", "while", "about", "up", "its", "it",
    "this", "that", "these", "those", "i", "me", "my", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "they", "them",
    "their", "what", "which", "who", "whom", "tell", "show", "give",
    "get", "find", "list", "please", "know", "many",
})

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    """
    Normalise a user query into a deterministic SHA-256 cache key.

    Pipeline: NFKC → lowercase → strip punctuation → collapse whitespace →
    remove stopwords → sort tokens → SHA-256.
    """
    # 1. Unicode NFKC + lowercase
    text = unicodedata.normalize("NFKC", text).lower()
    # 2. Strip punctuation
    text = _PUNCT_RE.sub(" ", text)
    # 3. Collapse whitespace
    text = _SPACE_RE.sub(" ", text).strip()
    # 4. Remove stopwords
    tokens = [t for t in text.split() if t not in _STOPWORDS]
    if not tokens:
        # Edge case: query was entirely stopwords — fall back to full text
        tokens = text.split()
    # 5. Sort tokens (order-independent matching)
    tokens.sort()
    # 6. SHA-256 hash
    canonical = " ".join(tokens)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── In-memory TTL cache ──────────────────────────────────────────────────────


class _TTLCache:
    """
    Minimal in-memory TTL cache (dict + monotonic timestamps).

    Not thread-safe, but that is fine — asyncio is single-threaded.
    Entries are lazily evicted on access; a full sweep runs on every ``set``.
    """

    def __init__(self, ttl_seconds: int, max_size: int = 2048) -> None:
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: dict[str, tuple[float, Any]] = {}  # key → (expires_at, value)

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        # Lazy eviction sweep
        now = time.monotonic()
        if len(self._store) >= self._max_size:
            expired_keys = [k for k, (exp, _) in self._store.items() if now > exp]
            for k in expired_keys:
                del self._store[k]
            # If still over capacity, evict oldest entries
            if len(self._store) >= self._max_size:
                oldest = sorted(self._store, key=lambda k: self._store[k][0])
                for k in oldest[: len(oldest) // 4]:
                    del self._store[k]

        self._store[key] = (now + self._ttl, value)


# ── Redis cache key prefix ──────────────────────────────────────────────────

_CACHE_PREFIX = "query_dedup:"


# ── QueryDeduplicator ────────────────────────────────────────────────────────


class QueryDeduplicator:
    """
    Two-layer query deduplication.

    Layer 1 uses either Redis or an in-memory TTL cache depending on whether
    a ``redis_client`` is provided.  Layer 2 (in-flight coalescing) is always
    process-local.

    Parameters
    ----------
    ttl_seconds:
        Response cache TTL in seconds (default: 1800 = 30 min).
    enabled:
        Master switch — when ``False``, all calls pass straight through.
    max_cache_size:
        Maximum number of cached responses in memory (ignored in Redis mode).
    redis_client:
        Optional async Redis client.  ``None`` = in-memory mode.
    """

    def __init__(
        self,
        ttl_seconds: int = 1800,
        enabled: bool = True,
        max_cache_size: int = 2048,
        redis_client: "aioredis.Redis | None" = None,
    ) -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._cache = _TTLCache(ttl_seconds=ttl_seconds, max_size=max_cache_size)
        self._enabled = enabled
        # Layer 2: in-flight futures (process-local)
        self._in_flight: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    async def deduplicated_invoke(
        self,
        query: str,
        agent: Any,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Invoke the agent with deduplication.

        Parameters
        ----------
        query:
            Raw user query string.
        agent:
            Compiled LangGraph agent (must support ``ainvoke``).
        config:
            LangGraph config dict (``{"configurable": {"thread_id": ...}}``).

        Returns
        -------
        dict
            Agent result dict containing ``{"messages": [...]}``
            OR a cached dict ``{"answer": ..., "tokens_used": ...}``.
        """
        if not self._enabled:
            from langchain_core.messages import HumanMessage
            return await agent.ainvoke(
                {"messages": [HumanMessage(content=query)]}, config=config,
            )

        key = normalize_query(query)

        # ── Layer 1: response cache ──────────────────────────────────────
        cached = await self._get_cached(key)
        if cached is not None:
            logger.info("Query dedup CACHE HIT (key=%s…)", key[:12])
            from src.core.tracing import trace_event
            trace_event("DEDUP_CHECK", "ok", "CACHE HIT")
            return cached

        # ── Layer 2: in-flight coalescing ────────────────────────────────
        existing_future: asyncio.Future[dict[str, Any]] | None = None
        future: asyncio.Future[dict[str, Any]] | None = None
        async with self._lock:
            if key in self._in_flight:
                logger.info("Query dedup IN-FLIGHT JOIN (key=%s…)", key[:12])
                from src.core.tracing import trace_event
                trace_event("DEDUP_CHECK", "ok", "IN-FLIGHT JOIN")
                # Grab a reference BEFORE releasing the lock — the owner's
                # ``finally`` block may remove the key at any moment.
                existing_future = self._in_flight[key]
            else:
                # Atomically register ourselves as the owner within the
                # same lock acquisition — prevents race where multiple
                # tasks all see "no in-flight" and all become owners.
                future = asyncio.get_running_loop().create_future()
                self._in_flight[key] = future

        # If we joined an existing future, await the captured reference
        if existing_future is not None:
            return await existing_future

        # We are the owner (future is registered in _in_flight)
        assert future is not None
        try:
            from langchain_core.messages import HumanMessage
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=query)]}, config=config,
            )
            # Build the serialised payload (same shape for cache & joiners)
            payload = self._build_payload(result)
            # Store in cache
            if payload is not None:
                await self._store_cached(key, payload)

            # Resolve the future so joiners receive the same payload
            if not future.done():
                future.set_result(payload if payload is not None else result)

            from src.core.tracing import trace_event
            trace_event("DEDUP_CHECK", "info", "CACHE MISS → invoking agent")
            logger.info("Query dedup MISS → cached (key=%s…)", key[:12])
            return payload if payload is not None else result

        except Exception as exc:
            # Propagate exception to all waiters
            if not future.done():
                future.set_exception(exc)
            raise

        finally:
            async with self._lock:
                self._in_flight.pop(key, None)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_payload(result: dict[str, Any]) -> dict[str, Any] | None:
        """Extract a serialisable payload from the agent result."""
        messages = result.get("messages", [])
        if not messages:
            return None
        last_msg = messages[-1]
        payload: dict[str, Any] = {
            "answer": last_msg.content,
            "tokens_used": None,
            "cached_at": time.time(),
        }
        if hasattr(last_msg, "usage_metadata") and last_msg.usage_metadata:
            payload["tokens_used"] = last_msg.usage_metadata.get("total_tokens")
        return payload

    async def _get_cached(self, key: str) -> dict[str, Any] | None:
        """Retrieve a cached response (Redis or in-memory)."""
        if self._redis is not None:
            try:
                raw = await self._redis.get(f"{_CACHE_PREFIX}{key}")
                if raw is None:
                    return None
                return json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
            except Exception as exc:
                logger.warning("Query dedup cache read failed: %s", exc)
                return None
        else:
            return self._cache.get(key)

    async def _store_cached(self, key: str, payload: dict[str, Any]) -> None:
        """Store a serialised payload (Redis or in-memory)."""
        if self._redis is not None:
            try:
                await self._redis.setex(
                    f"{_CACHE_PREFIX}{key}",
                    self._ttl,
                    json.dumps(payload),
                )
            except Exception as exc:
                logger.warning("Query dedup cache write failed: %s", exc)
        else:
            self._cache.set(key, payload)
