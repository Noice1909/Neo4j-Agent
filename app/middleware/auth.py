"""
API key authentication middleware.

When ``settings.api_key`` is non-empty, every request (except health probes
and OpenAPI docs) must include the key in the ``X-API-Key`` header.
When the key is empty, authentication is disabled (development mode).
"""
from __future__ import annotations

import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Paths that never require authentication.
_PUBLIC_PATHS: set[str] = {
    "/health",
    "/health/live",
    "/health/ready",
    "/docs",
    "/redoc",
    "/openapi.json",
}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests without a valid ``X-API-Key`` header."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        settings = get_settings()

        # Auth disabled when no key is configured.
        if not settings.api_key:
            return await call_next(request)

        # Allow public endpoints through.
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # Allow MCP sub-app (has its own auth layer when needed).
        if request.url.path.startswith(settings.mcp_path):
            return await call_next(request)

        supplied_key = request.headers.get("X-API-Key", "")
        if supplied_key != settings.api_key:
            logger.warning(
                "Rejected request: invalid API key from %s %s",
                request.client.host if request.client else "unknown",
                request.url.path,
            )
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or missing API key.", "code": "UNAUTHORIZED"},
            )

        return await call_next(request)
