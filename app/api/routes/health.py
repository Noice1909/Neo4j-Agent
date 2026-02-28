"""
Health check route — verifies all critical dependencies are reachable.

GET /health
    Returns 200 if all checks pass, 503 if any dependency is down.
    Each check is independent — partial failures are reported.

GET /health/live
    Kubernetes liveness probe — simple 200 if the process is alive.

GET /health/ready
    Kubernetes readiness probe — 200 only if all dependencies are healthy.
"""
from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum

import httpx
import redis.asyncio as aioredis
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.api.schemas.health import CheckStatus, DependencyCheck, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("", response_model=HealthResponse, summary="Full health check.")
async def health_check() -> JSONResponse:
    """
    Check connectivity to Neo4j, Redis, and Ollama.
    Returns 200 when all checks pass, 503 if any fail.
    """
    neo4j_check, redis_check, ollama_check = await asyncio.gather(
        _check_neo4j(),
        _check_redis(),
        _check_ollama(),
        return_exceptions=False,
    )

    # Redis is optional — in-memory fallback is active when Redis is unavailable.
    # Only Neo4j and Ollama are critical for the agent to function.
    overall = (
        CheckStatus.ok
        if all(c.status == CheckStatus.ok for c in [neo4j_check, ollama_check])
        else CheckStatus.error
    )

    response = HealthResponse(
        status=overall,
        neo4j=neo4j_check,
        redis=redis_check,
        ollama=ollama_check,
    )

    http_status = 200 if overall == CheckStatus.ok else 503
    return JSONResponse(content=response.model_dump(), status_code=http_status)


@router.get("/live", summary="Liveness probe.", include_in_schema=False)
async def liveness() -> dict:
    """Always returns 200 if the process is alive."""
    return {"status": "alive"}


@router.get("/ready", summary="Readiness probe.", include_in_schema=False)
async def readiness() -> JSONResponse:
    """Returns 200 only if all external dependencies are reachable."""
    neo4j_check, redis_check, ollama_check = await asyncio.gather(
        _check_neo4j(),
        _check_redis(),
        _check_ollama(),
    )
    all_ok = all(
        c.status == CheckStatus.ok for c in [neo4j_check, ollama_check]
    )
    return JSONResponse(
        content={"status": "ready" if all_ok else "not_ready"},
        status_code=200 if all_ok else 503,
    )


# ── Individual checks ─────────────────────────────────────────────────────────


async def _check_neo4j() -> DependencyCheck:
    try:
        from app.graph.connection import get_graph

        graph = get_graph()
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        # Simple connectivity check — count any node
        await loop.run_in_executor(None, lambda: graph.query("RETURN 1 AS ping"))
        latency = round((time.perf_counter() - t0) * 1000, 2)
        return DependencyCheck(status=CheckStatus.ok, latency_ms=latency)
    except Exception as exc:
        logger.warning("Neo4j health check failed: %s", exc)
        return DependencyCheck(status=CheckStatus.error, detail=str(exc))


async def _check_redis() -> DependencyCheck:
    try:
        from app.core.dependencies import get_schema_cache_instance
        schema_cache = get_schema_cache_instance()
        t0 = time.perf_counter()
        pong = schema_cache._redis.ping()
        if hasattr(pong, "__await__"):
            await pong  # type: ignore[misc]
        latency = round((time.perf_counter() - t0) * 1000, 2)
        return DependencyCheck(status=CheckStatus.ok, latency_ms=latency)
    except RuntimeError:
        # SchemaCache not initialised yet
        return DependencyCheck(status=CheckStatus.skipped, detail="Redis not initialised yet")
    except Exception as exc:
        logger.warning("Redis health check failed: %s", exc)
        # Redis is optional — in-memory fallback is active; report as skipped, not error.
        return DependencyCheck(status=CheckStatus.skipped, detail="Redis unavailable — using in-memory fallback")


async def _check_ollama() -> DependencyCheck:
    settings = get_settings()
    try:
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            response.raise_for_status()
        latency = round((time.perf_counter() - t0) * 1000, 2)
        return DependencyCheck(status=CheckStatus.ok, latency_ms=latency)
    except Exception as exc:
        logger.warning("Ollama health check failed: %s", exc)
        return DependencyCheck(status=CheckStatus.error, detail=str(exc))
