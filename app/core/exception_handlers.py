"""
FastAPI exception → JSONResponse handlers.

Each handler converts a domain exception into a structured JSON error
response with an appropriate HTTP status code.  Register them in
``app.main.create_app()`` via ``app.add_exception_handler()``.
"""
from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    AgentError,
    ReadOnlyViolationError,
    SchemaUnavailableError,
    SessionNotFoundError,
    VectorSearchUnavailableError,
)

logger = logging.getLogger(__name__)


async def read_only_violation_handler(
    request: Request, exc: ReadOnlyViolationError
) -> JSONResponse:
    return JSONResponse(
        status_code=403,
        content={
            "error": "Write operations are not permitted.",
            "code": "READ_ONLY_VIOLATION",
        },
    )


async def schema_unavailable_handler(
    request: Request, exc: SchemaUnavailableError
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": "Neo4j schema is temporarily unavailable. Retry shortly.",
            "code": "SCHEMA_UNAVAILABLE",
        },
    )


async def session_not_found_handler(
    request: Request, exc: SessionNotFoundError
) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error": f"Session not found: {exc.session_id}",
            "code": "SESSION_NOT_FOUND",
        },
    )


async def agent_error_handler(
    request: Request, exc: AgentError
) -> JSONResponse:
    logger.error("Agent error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "An internal agent error occurred. Please try again.",
            "code": "AGENT_ERROR",
        },
    )


async def vector_search_unavailable_handler(
    request: Request, exc: VectorSearchUnavailableError
) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={
            "error": "Vector search is not configured.",
            "code": "VECTOR_SEARCH_UNAVAILABLE",
        },
    )
