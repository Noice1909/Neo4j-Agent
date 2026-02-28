"""
Domain exception classes.

These are raised by the business-logic and infrastructure layers.
The corresponding FastAPI exception handlers live in
``src.core.exception_handlers``.
"""
from __future__ import annotations


class ReadOnlyViolationError(Exception):
    """Raised when a generated Cypher query contains write operations."""

    def __init__(self, query: str) -> None:
        self.query = query
        super().__init__(f"Write operation detected in query: {query[:120]!r}")


class SchemaUnavailableError(Exception):
    """Raised when the Neo4j schema cannot be retrieved (DB down, cold start)."""


class SessionNotFoundError(Exception):
    """Raised when a requested session_id has no history in the checkpointer."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


class AgentError(Exception):
    """Generic agent execution failure."""


class VectorSearchUnavailableError(Exception):
    """Raised when vector index is not configured in Neo4j."""
