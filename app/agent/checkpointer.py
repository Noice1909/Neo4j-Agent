"""
Redis-backed LangGraph checkpointer management.

The checkpointer persists the full conversation state (messages + metadata)
to Redis keyed by `thread_id` (= `session_id` supplied by the client).

Swapping from Redis to another backend (e.g. Neo4jSaver, SqliteSaver) only
requires changing this file — all call sites use `get_checkpointer()`.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

logger = logging.getLogger(__name__)

# Module-level singleton set during lifespan startup.
_checkpointer = None


async def init_checkpointer(redis_url: str) -> "BaseCheckpointSaver | None":
    """
    Create the `AsyncRedisSaver` and run its setup DDL against Redis.

    Must be called once during application lifespan startup.
    Tries the standard langgraph-checkpoint-redis v2 API and falls back
    gracefully to MemorySaver for offline/testing scenarios.

    Parameters
    ----------
    redis_url:
        Redis connection string, e.g. ``redis://localhost:6379``.
    """
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    logger.info("Initialising AsyncRedisSaver checkpointer (%s).", redis_url)
    try:
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver

        # v2 API: constructor accepts redis_url directly
        saver = AsyncRedisSaver(redis_url=redis_url)
        # asetup() creates required key-spaces / indices in Redis
        if hasattr(saver, "asetup"):
            await saver.asetup()
        elif hasattr(saver, "setup"):
            import asyncio
            await asyncio.get_running_loop().run_in_executor(None, saver.setup)
        _checkpointer = saver
        logger.info("AsyncRedisSaver checkpointer ready.")
    except Exception as exc:
        logger.warning(
            "AsyncRedisSaver init failed (%s). Falling back to MemorySaver.", exc
        )
        from langgraph.checkpoint.memory import MemorySaver
        _checkpointer = MemorySaver()
        logger.warning("Using in-memory MemorySaver — state will NOT persist on restart.")

    return _checkpointer


def get_checkpointer() -> "BaseCheckpointSaver":
    """Return the existing checkpointer singleton (must call `init_checkpointer` first)."""
    if _checkpointer is None:
        raise RuntimeError(
            "Checkpointer has not been initialised. "
            "Ensure `init_checkpointer()` is called during application lifespan startup."
        )
    return _checkpointer


async def _try_close(obj: object) -> bool:
    """Attempt to close *obj* via aclose/close. Returns True on success."""
    for attr in ("aclose", "close"):
        closer = getattr(obj, attr, None)
        if closer is not None and callable(closer):
            import asyncio
            result = closer()
            if asyncio.iscoroutine(result):
                await result
            return True
    return False


async def close_checkpointer() -> None:
    """Close the underlying Redis connection on shutdown."""
    global _checkpointer
    if _checkpointer is None:
        return
    try:
        if not await _try_close(_checkpointer):
            # If the saver wraps an internal Redis client, close it too.
            redis_conn = getattr(_checkpointer, "_redis", None) or getattr(_checkpointer, "conn", None)
            if redis_conn is not None:
                await _try_close(redis_conn)
    except Exception:
        logger.debug("Checkpointer cleanup error (ignored)", exc_info=True)
    _checkpointer = None
    logger.info("Checkpointer connection closed.")

