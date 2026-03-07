"""
FastAPI application factory with full lifespan management.

Backend mode is controlled by ``CACHE_BACKEND`` env var:
  - ``memory`` (default): In-memory caches + SQLite checkpointer (no Redis).
  - ``redis``: Redis-backed caches + Redis checkpointer (multi-worker safe).

Startup sequence:
  1. Validate configuration
  2. Initialise Neo4j graph connection
  3. Initialise schema cache (+ optional Redis client) → warm up schema
  4. Initialise LangGraph checkpointer
  5. Configure LangChain LLM response cache
  6. Build LangChain tools (with injected dependencies)
  7. Compile LangGraph agent
  8. Register FastMCP tools
  9. Mount FastMCP sub-app on /mcp
  10. Register exception handlers + API routers → app is ready

Shutdown sequence (reverse):
  A. Cancel schema proactive-refresh background task
  B. Close checkpointer connection
  C. Close Redis client (if applicable)
  D. Close Neo4j driver
"""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.globals import set_llm_cache

# Redis imports — only needed when cache_backend="redis"
try:
    import redis.asyncio as aioredis
    from langchain_community.cache import RedisCache
except ImportError:
    aioredis = None  # type: ignore[assignment]
    RedisCache = None  # type: ignore[assignment,misc]

from src.agent.checkpointer import close_checkpointer, init_checkpointer
from src.agent.factory import get_compiled_agent, init_agent
from src.api.routes import chat, health, schema, sessions
from src.core.config import Settings, get_settings
from src.core.dependencies import set_schema_cache_instance, set_query_dedup_instance
from src.core.exceptions import (
    AgentError,
    ReadOnlyViolationError,
    SchemaUnavailableError,
    SessionNotFoundError,
    VectorSearchUnavailableError,
)
from src.core.exception_handlers import (
    agent_error_handler,
    read_only_violation_handler,
    schema_unavailable_handler,
    session_not_found_handler,
    vector_search_unavailable_handler,
)
from src.graph.connection import close_graph, init_graph
from src.graph.cypher.coreference import build_coreference_regex, set_coreference_regex

from src.graph.schema_cache import SchemaCache
from src.llm.factory import get_llm_from_settings
from src.mcp.server import mcp, register_all_tools
from src.mcp.tools.graph_query import build_graph_query_tool
from src.mcp.tools.schema_info import build_schema_info_tool
from src.mcp.tools.vector_search import build_vector_search_tool
from src.middleware.auth import APIKeyMiddleware
from src.middleware.rate_limit import limiter
from src.services.query_dedup import QueryDeduplicator

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    _prometheus_available = True
except ImportError:
    Instrumentator = None  # type: ignore[assignment,misc]
    _prometheus_available = False

logger = logging.getLogger(__name__)

_API_V1_PREFIX = "/api/v1"


def _set_inmemory_llm_cache() -> None:
    """Configure an in-memory LLM response cache (no Redis)."""
    try:
        from langchain_community.cache import InMemoryCache
    except ImportError:
        from langchain.cache import InMemoryCache  # type: ignore[no-redef]
    set_llm_cache(InMemoryCache())
    logger.info("LLM cache: InMemoryCache (per-process).")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan context manager.

    Everything between ``yield`` and the final ``await`` is the "running" phase.
    Startup happens before yield; shutdown happens after.
    """
    settings: Settings = get_settings()

    # ── 0. Structured logging (must run AFTER uvicorn configures its loggers) ─
    from src.core.logging import setup_logging
    setup_logging(settings.log_level, settings=settings)

    variant = "Redis" if settings.use_redis else "in-memory"
    logger.info("═══ %s — startup (%s backend) ═══", settings.app_name, variant)

    # ── 1. Neo4j ──────────────────────────────────────────────────────────────
    logger.info("[1/8] Initialising Neo4j...")
    graph = init_graph(settings)

    # ── 2. Schema cache (+ optional Redis client) ─────────────────────────────
    redis_client = None
    if settings.use_redis:
        if aioredis is None:
            raise RuntimeError(
                "CACHE_BACKEND='redis' but redis package is not installed. "
                "Install with: pip install 'redis[asyncio]'"
            )
        logger.info("[2/8] Initialising Redis + schema cache...")
        redis_client = aioredis.from_url(settings.redis_url, decode_responses=False)
    else:
        logger.info("[2/8] Initialising in-memory schema cache...")

    schema_cache = SchemaCache(
        graph=graph,
        ttl_seconds=settings.schema_cache_ttl_seconds,
        redis_client=redis_client,
    )
    await schema_cache.warm_up()
    set_schema_cache_instance(schema_cache)

    # ── 2b. Extract graph topology (cached inside schema_cache) ───────────────
    logger.info("[2b] Extracting graph topology...")
    topology = await schema_cache.get_topology()
    logger.info(
        "Topology ready: %d labels, %d triples.",
        len(topology.labels), len(topology.triples),
    )

    # ── 2c. Build dynamic coreference regex from schema labels ────────────────
    set_coreference_regex(build_coreference_regex(
        topology.label_names,
        nlp_terms_by_label=topology.nlp_terms_by_label,
    ))
    logger.info("Coreference regex updated for %d labels.", len(topology.label_names))

    # ── 3. LangGraph checkpointer ─────────────────────────────────────────────
    if settings.use_redis:
        logger.info("[3/8] Initialising LangGraph checkpointer (backend=redis)...")
        await init_checkpointer(backend="redis", redis_url=settings.redis_url)
    else:
        logger.info("[3/8] Initialising LangGraph checkpointer (backend=%s)...", settings.checkpointer_backend)
        await init_checkpointer(
            backend=settings.checkpointer_backend,
            redis_url=settings.redis_url,
            sqlite_path=settings.sqlite_checkpoint_path,
        )

    # ── 4. LLM response cache ────────────────────────────────────────────────
    logger.info("[4/8] Configuring LLM response cache...")
    if settings.use_redis and RedisCache is not None:
        try:
            import redis as sync_redis
            sync_redis_client = sync_redis.from_url(
                settings.redis_url, socket_connect_timeout=2,
            )
            sync_redis_client.ping()
            set_llm_cache(
                RedisCache(
                    redis_=sync_redis_client,
                    ttl=settings.llm_cache_ttl_seconds,
                )
            )
            logger.info("LLM cache: Redis (TTL=%ds)", settings.llm_cache_ttl_seconds)
        except Exception as exc:
            logger.warning("Redis LLM cache unavailable (%s); falling back to in-memory.", exc)
            _set_inmemory_llm_cache()
    else:
        _set_inmemory_llm_cache()

    # ── 4b. Query deduplicator ────────────────────────────────────────────────
    if settings.query_dedup_enabled:
        query_dedup = QueryDeduplicator(
            ttl_seconds=settings.query_cache_ttl_seconds,
            enabled=True,
            redis_client=redis_client,
        )
        set_query_dedup_instance(query_dedup)
        cache_type = "Redis" if redis_client else "in-memory"
        logger.info(
            "Query deduplication ENABLED (TTL=%ds, %s).",
            settings.query_cache_ttl_seconds, cache_type,
        )
    else:
        query_dedup = QueryDeduplicator(enabled=False)
        set_query_dedup_instance(query_dedup)
        logger.info("Query deduplication DISABLED.")

    # ── 5. Build LangChain tools ──────────────────────────────────────────────
    logger.info("[5/8] Building LangChain tools...")
    llm = get_llm_from_settings(settings)
    tools = [
        build_graph_query_tool(llm, schema_cache),
        build_schema_info_tool(schema_cache),
        build_vector_search_tool(llm),
    ]

    # ── 5b. Concept-based synonym enrichment (no LLM call needed) ───────────
    logger.info("[5b] Synonym enrichment via Concept nodes: %d labels covered.", len(topology.nlp_terms_by_label))

    # ── 6. Compile LangGraph agent ────────────────────────────────────────────
    logger.info("[6/8] Compiling LangGraph agent...")
    from src.agent.checkpointer import get_checkpointer
    init_agent(
        llm=llm,
        tools=tools,
        checkpointer=get_checkpointer(),
        schema_labels=topology.label_names,
        label_descriptions={li.label: li.description for li in topology.labels if li.description},
        max_conversation_tokens=settings.max_conversation_tokens,
        token_budget_reserve=settings.token_budget_reserve,
    )

    # ── 7. Register FastMCP tools ─────────────────────────────────────────────
    logger.info("[7/8] Registering FastMCP tools...")
    register_all_tools(settings)

    # ── 8. Done ───────────────────────────────────────────────────────────────
    logger.info("[8/8] All systems ready. %s is serving requests.", settings.app_name)

    yield  # ←── Application is running

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("═══ %s — shutdown ═══", settings.app_name)
    await schema_cache.stop_refresh_task()
    await close_checkpointer()
    if redis_client is not None:
        await redis_client.aclose()
    close_graph()
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    """
    Application factory — returns a fully configured FastAPI instance.
    Separated from module-level instantiation for testability.
    """
    settings = get_settings()

    # ── Swagger / ReDoc: only exposed when DEBUG=true ─────────────────────────
    docs_url = "/docs" if settings.debug else None
    redoc_url = "/redoc" if settings.debug else None

    app = FastAPI(
        title=settings.app_name,
        description=(
            "Enterprise-grade read-only Neo4j knowledge graph agent. "
            "Powered by LangGraph + Ollama + FastMCP."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url=docs_url,
        redoc_url=redoc_url,
    )

    # ── Request ID middleware (outermost — runs first) ────────────────────────
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    _ = request_id_middleware  # registered by @app.middleware decorator

    # ── API key authentication middleware ─────────────────────────────────────
    app.add_middleware(APIKeyMiddleware)

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: list[str] = [
        o.strip() for o in settings.cors_origins.split(",") if o.strip()
    ]
    if settings.debug and not cors_origins:
        cors_origins = ["*"]

    allow_credentials = "*" not in cors_origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Rate limiting ─────────────────────────────────────────────────────────
    app.state.limiter = limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi import _rate_limit_exceeded_handler
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # ── Exception handlers ────────────────────────────────────────────────────
    app.add_exception_handler(ReadOnlyViolationError, read_only_violation_handler)  # type: ignore[arg-type]
    app.add_exception_handler(SchemaUnavailableError, schema_unavailable_handler)  # type: ignore[arg-type]
    app.add_exception_handler(SessionNotFoundError, session_not_found_handler)  # type: ignore[arg-type]
    app.add_exception_handler(AgentError, agent_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(VectorSearchUnavailableError, vector_search_unavailable_handler)  # type: ignore[arg-type]

    # ── API routers ───────────────────────────────────────────────────────────
    app.include_router(chat.router, prefix=_API_V1_PREFIX)
    app.include_router(sessions.router, prefix=_API_V1_PREFIX)
    app.include_router(schema.router, prefix=_API_V1_PREFIX)
    app.include_router(health.router)

    # ── FastMCP mount ─────────────────────────────────────────────────────────
    mcp_asgi = mcp.http_app(path="/", stateless_http=True)
    app.mount(settings.mcp_path, mcp_asgi)

    # ── Prometheus metrics ────────────────────────────────────────────────────
    if _prometheus_available and Instrumentator is not None:
        Instrumentator(
            should_group_status_codes=True,
            excluded_handlers=["/health", "/health/live", "/health/ready", "/metrics"],
        ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    return app


# Module-level app instance (used by uvicorn)
app = create_app()
