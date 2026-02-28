"""
Schema inspection route.

GET  /api/v1/schema          — Return the currently cached schema string.
POST /api/v1/schema/refresh  — Force-invalidate the cache and re-fetch from Neo4j.
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.core.dependencies import get_schema_cache
from app.middleware.rate_limit import limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/schema", tags=["Schema"])


@router.get("", summary="Return the currently cached graph schema.")
async def get_schema(schema_cache=Depends(get_schema_cache)) -> JSONResponse:
    """
    Returns the schema string that is injected into every Cypher-generation
    prompt.  The schema is fetched from the in-memory / Redis cache — no
    live Neo4j query is issued on this call.
    """
    t0 = time.perf_counter()
    schema: str = await schema_cache.get_schema()
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return JSONResponse(
        content={
            "schema": schema,
            "chars": len(schema),
            "latency_ms": latency_ms,
        }
    )


@router.post("/refresh", summary="Force-invalidate the schema cache and re-fetch from Neo4j.")
@limiter.limit("2/minute")
async def refresh_schema(request: Request, schema_cache=Depends(get_schema_cache)) -> JSONResponse:
    """
    Invalidates the cached schema key and immediately re-fetches from Neo4j.
    Call this endpoint after any schema migration (new labels, indexes, etc.)
    to make the agent aware of the changes without waiting for the TTL to expire.
    """
    t0 = time.perf_counter()
    await schema_cache.invalidate()
    schema: str = await schema_cache.get_schema()
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info("Schema cache manually refreshed (%d chars) in %.1f ms.", len(schema), latency_ms)
    return JSONResponse(
        content={
            "message": "Schema cache refreshed successfully.",
            "schema": schema,
            "chars": len(schema),
            "latency_ms": latency_ms,
        }
    )
