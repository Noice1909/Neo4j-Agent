"""
Chat request / response schemas.
"""
from __future__ import annotations

import uuid

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat message from a user."""

    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description=(
            "UUID identifying the user's conversation session.  "
            "Reuse the same value to continue a conversation; "
            "use a new value to start fresh."
        ),
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The user's natural-language question or message.",
        examples=["How many movies are in the database?"],
    )


class ChatResponse(BaseModel):
    """Agent reply to a chat message."""

    session_id: str = Field(description="The session_id of this conversation.")
    message: str = Field(description="The agent's response.")
    tokens_used: int | None = Field(
        default=None,
        description="Approximate token usage (if reported by the LLM).",
    )
