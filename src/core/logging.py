"""
Structured logging configuration using structlog.

Produces JSON-formatted log entries suitable for log aggregation
(ELK, CloudWatch, Datadog, Loki).
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def setup_logging(log_level: str = "INFO", settings: Any = None) -> None:
    """Configure structured JSON logging for the entire application.

    Parameters
    ----------
    log_level:
        Root log level string (e.g. "DEBUG", "INFO", "WARNING").
    settings:
        Optional ``Settings`` instance.  When provided, all ``.env`` values
        are automatically masked in every log message.
    """
    # ── Sensitive data masking ────────────────────────────────────────────
    # Wraps sys.stderr / sys.stdout so every string that hits the terminal
    # passes through _replace_secrets().  Works regardless of handler/formatter.
    if settings is not None:
        from src.core.masking import (
            init_masking, install_stream_masking, mask_sensitive_processor,
        )
        init_masking(settings)
        install_stream_masking()
        masking_processors: list[structlog.types.Processor] = [mask_sensitive_processor]
    else:
        masking_processors = []

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        *masking_processors,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer()
            if log_level.upper() == "DEBUG"
            else structlog.processors.JSONRenderer(),
        ],
    )

    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(log_level.upper())

    for name in ("httpx", "httpcore", "neo4j", "urllib3", "asyncio"):
        logging.getLogger(name).setLevel(logging.WARNING)
