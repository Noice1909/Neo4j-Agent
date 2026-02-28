"""
Health check response schemas.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class CheckStatus(str, Enum):
    ok = "ok"
    error = "error"
    skipped = "skipped"


class DependencyCheck(BaseModel):
    status: CheckStatus
    latency_ms: float | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    status: CheckStatus
    neo4j: DependencyCheck
    redis: DependencyCheck
    ollama: DependencyCheck
