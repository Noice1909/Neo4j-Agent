"""
Session management routes.

GET    /api/v1/sessions/{session_id}/history
    Return the full message history for a session.

DELETE /api/v1/sessions/{session_id}
    Permanently delete all conversation history for a session.

GET    /api/v1/sessions/{session_id}/exists
    Check whether a session exists without fetching full history.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_checkpointer
from app.core.exceptions import SessionNotFoundError
from app.api.schemas.sessions import (
    DeleteSessionResponse,
    MessageRecord,
    SessionExistsResponse,
    SessionHistory,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["Sessions"])


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get(
    "/{session_id}/history",
    response_model=SessionHistory,
    summary="Retrieve full conversation history for a session.",
)
async def get_session_history(
    session_id: str,
    checkpointer=Depends(get_checkpointer),
) -> SessionHistory:
    """
    Return all messages exchanged in the given session, ordered oldest-first.
    """
    config = {"configurable": {"thread_id": session_id}}
    checkpoint_tuple = await checkpointer.aget_tuple(config)

    if checkpoint_tuple is None:
        raise SessionNotFoundError(session_id)

    checkpoint = checkpoint_tuple.checkpoint
    raw_messages: list[Any] = checkpoint.get("channel_values", {}).get("messages", [])

    messages: list[MessageRecord] = []
    for msg in raw_messages:
        role = _extract_role(msg)
        content = _extract_content(msg)
        metadata = _extract_metadata(msg)
        messages.append(MessageRecord(role=role, content=content, metadata=metadata))

    logger.info("Session history fetched: session_id=%s messages=%d", session_id, len(messages))

    return SessionHistory(
        session_id=session_id,
        message_count=len(messages),
        messages=messages,
    )


@router.get(
    "/{session_id}/exists",
    response_model=SessionExistsResponse,
    summary="Check whether a session exists.",
)
async def session_exists(
    session_id: str,
    checkpointer=Depends(get_checkpointer),
) -> SessionExistsResponse:
    """Return whether the session has any stored history."""
    config = {"configurable": {"thread_id": session_id}}
    checkpoint_tuple = await checkpointer.aget_tuple(config)
    return SessionExistsResponse(
        session_id=session_id,
        exists=checkpoint_tuple is not None,
    )


@router.delete(
    "/{session_id}",
    response_model=DeleteSessionResponse,
    summary="Delete all conversation history for a session.",
)
async def delete_session(
    session_id: str,
    checkpointer=Depends(get_checkpointer),
) -> DeleteSessionResponse:
    """
    Permanently delete the session state from Redis.
    The session_id can be reused for a fresh conversation afterwards.
    """
    config = {"configurable": {"thread_id": session_id}}

    # Verify the session exists first
    checkpoint_tuple = await checkpointer.aget_tuple(config)
    if checkpoint_tuple is None:
        raise SessionNotFoundError(session_id)

    # LangGraph Redis saver: delete the thread's checkpoint keys from Redis.
    try:
        from app.core.config import get_settings
        import redis.asyncio as aioredis

        settings = get_settings()
        redis_client = aioredis.from_url(settings.redis_url)

        # Use precise key prefix patterns to avoid matching unrelated keys.
        # LangGraph checkpoint keys: checkpoint:<namespace>:<thread_id>:<checkpoint_id>
        deleted = 0
        for prefix in (f"checkpoint:{session_id}:", f"checkpoint:*:{session_id}:"):
            async for key in redis_client.scan_iter(match=prefix + "*", count=200):
                await redis_client.delete(key)
                deleted += 1
        await redis_client.aclose()
        logger.info("Session deleted: session_id=%s (%d keys removed)", session_id, deleted)
    except Exception as exc:
        logger.error("Failed to delete session %s: %s", session_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete session. Please try again.") from exc

    return DeleteSessionResponse(session_id=session_id, deleted=True)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_role(msg: Any) -> str:
    if hasattr(msg, "type"):
        return msg.type  # "human", "ai", "tool"
    if hasattr(msg, "role"):
        return msg.role
    return "unknown"


def _extract_content(msg: Any) -> str:
    if hasattr(msg, "content"):
        content = msg.content
        if isinstance(content, list):
            # Handle multi-modal content blocks
            return " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return str(content)
    return ""


def _extract_metadata(msg: Any) -> dict | None:
    meta: dict = {}
    if hasattr(msg, "usage_metadata") and msg.usage_metadata:
        meta["usage"] = msg.usage_metadata
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        meta["tool_calls"] = [
            {"name": tc.get("name"), "id": tc.get("id")} for tc in msg.tool_calls
        ]
    return meta if meta else None
