"""
Session management response schemas.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class MessageRecord(BaseModel):
    """A single turn in the conversation."""

    role: str
    content: str
    metadata: dict[str, Any] | None = None


class SessionHistory(BaseModel):
    """Full conversation history for a session."""

    session_id: str
    message_count: int
    messages: list[MessageRecord]


class SessionExistsResponse(BaseModel):
    session_id: str
    exists: bool


class DeleteSessionResponse(BaseModel):
    session_id: str
    deleted: bool
