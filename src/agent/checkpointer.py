"""
LangGraph checkpointer management — multi-backend.

Supports three backends controlled by ``settings.checkpointer_backend``:

  - **sqlite** (default): Persistent, single-process.  Uses
    ``AsyncSqliteSaver`` backed by ``data/checkpoints.db``.
  - **redis**: Persistent, multi-worker.  Uses ``AsyncRedisSaver``.
  - **memory**: In-memory ``MemorySaver`` — sessions lost on restart.

Swapping backends requires only a config change — all call-sites use
``get_checkpointer()``.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

logger = logging.getLogger(__name__)

# Module-level singleton set during lifespan startup.
_checkpointer = None


async def init_checkpointer(
    backend: str = "sqlite",
    *,
    redis_url: str | None = None,
    sqlite_path: str = "data/checkpoints.db",
) -> "BaseCheckpointSaver":
    """
    Create and cache a checkpointer singleton.

    Parameters
    ----------
    backend:
        One of ``"sqlite"``, ``"redis"``, ``"memory"``.
    redis_url:
        Redis connection string (only used when backend is ``"redis"``).
    sqlite_path:
        File path for the SQLite database (only used when backend is ``"sqlite"``).
    """
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    backend = backend.lower().strip()
    logger.info("Initialising checkpointer (backend=%s).", backend)

    if backend == "sqlite":
        _checkpointer = await _init_sqlite(sqlite_path)
    elif backend == "redis":
        if not redis_url:
            raise ValueError("redis_url is required when backend='redis'")
        _checkpointer = await _init_redis(redis_url)
    elif backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        _checkpointer = MemorySaver()
        logger.warning("Using in-memory MemorySaver — state will NOT persist on restart.")
    else:
        raise ValueError(f"Unknown checkpointer backend: {backend!r}. Use 'sqlite', 'redis', or 'memory'.")

    return _checkpointer


async def _init_sqlite(sqlite_path: str) -> "BaseCheckpointSaver":
    """Initialise an ``AsyncSqliteSaver`` checkpointer."""
    # Ensure the parent directory exists
    parent = os.path.dirname(sqlite_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    # AsyncSqliteSaver.from_conn_string returns an async context manager.
    # We need to open the connection ourselves and create the saver directly.
    conn = await aiosqlite.connect(sqlite_path)
    saver = AsyncSqliteSaver(conn)
    # Run setup DDL (creates tables if not present)
    await saver.setup()
    logger.info("AsyncSqliteSaver checkpointer ready (path=%s).", sqlite_path)
    return saver


async def _init_redis(redis_url: str) -> "BaseCheckpointSaver":
    """Initialise an ``AsyncRedisSaver`` checkpointer (fallback to MemorySaver)."""
    try:
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver

        saver = AsyncRedisSaver(redis_url=redis_url)
        if hasattr(saver, "asetup"):
            await saver.asetup()
        elif hasattr(saver, "setup"):
            import asyncio
            await asyncio.get_running_loop().run_in_executor(None, saver.setup)
        logger.info("AsyncRedisSaver checkpointer ready.")
        return saver
    except Exception as exc:
        logger.warning(
            "AsyncRedisSaver init failed (%s). Falling back to MemorySaver.", exc
        )
        from langgraph.checkpoint.memory import MemorySaver
        saver = MemorySaver()
        logger.warning("Using in-memory MemorySaver — state will NOT persist on restart.")
        return saver


def get_checkpointer() -> "BaseCheckpointSaver":
    """Return the existing checkpointer singleton (must call ``init_checkpointer`` first)."""
    if _checkpointer is None:
        raise RuntimeError(
            "Checkpointer has not been initialised. "
            "Ensure `init_checkpointer()` is called during application lifespan startup."
        )
    return _checkpointer


async def close_checkpointer() -> None:
    """Close the underlying connection on shutdown."""
    global _checkpointer
    if _checkpointer is not None:
        try:
            for attr in ("aclose", "close"):
                closer = getattr(_checkpointer, attr, None)
                if closer is not None and callable(closer):
                    import asyncio
                    result = closer()
                    if asyncio.iscoroutine(result):
                        await result
                    break
        except Exception:
            logger.debug("Checkpointer cleanup error (ignored)", exc_info=True)
        _checkpointer = None
        logger.info("Checkpointer connection closed.")

