"""
Application configuration via pydantic-settings.
All settings are read from environment variables (or .env file).
A single @lru_cache singleton is returned — swap the env file to change config.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Neo4j ────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str  # REQUIRED — no default; must be set via env / .env
    neo4j_database: str = "neo4j"
    # When True, rewrite neo4j+s:// → neo4j+ssc:// (skip TLS cert verification).
    # Only enable for managed cloud instances (AuraDB) where cert chain fails.
    neo4j_skip_tls_verify: bool = False

    # ── Ollama ───────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral:latest"
    # Temperature — lower = more deterministic Cypher generation
    ollama_temperature: float = 0.0

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"

    # ── Caches ───────────────────────────────────────────────────────────────
    schema_cache_ttl_seconds: int = 300
    llm_cache_ttl_seconds: int = 3600

    # ── Query Deduplication ──────────────────────────────────────────────────
    query_cache_ttl_seconds: int = 1800     # 30-minute TTL for cached responses
    query_dedup_enabled: bool = True        # Master switch for dedup layer

    # ── Entity Resolution ────────────────────────────────────────────────────
    entity_resolution_enabled: bool = True       # Master switch for entity correction
    entity_fuzzy_threshold: float = 0.75         # Levenshtein similarity cutoff (0.0–1.0)
    entity_synonym_overrides: str = ""           # JSON string of custom synonyms
    entity_max_candidates: int = 5               # Max candidates from DB lookup

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "Neo4j Agent"
    debug: bool = False
    log_level: str = "INFO"

    # ── Security ─────────────────────────────────────────────────────────────
    # API key for request authentication. Empty string = auth disabled (dev).
    api_key: str = ""
    # Comma-separated CORS origins. Empty = no origins allowed in production.
    cors_origins: str = ""

    # ── Rate Limiting ────────────────────────────────────────────────────────
    rate_limit_chat: str = "10/minute"
    rate_limit_general: str = "30/minute"

    # ── MCP ──────────────────────────────────────────────────────────────────
    mcp_path: str = "/mcp"


@lru_cache
def get_settings() -> Settings:
    """Return a singleton Settings instance (cached after first call)."""
    settings = Settings()  # type: ignore[call-arg]  # pydantic-settings fills from env
    from app.core.logging import setup_logging
    setup_logging(settings.log_level)
    return settings
