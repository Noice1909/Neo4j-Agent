"""
Structured logging configuration using structlog.

Produces JSON-formatted log entries suitable for log aggregation
(ELK, CloudWatch, Datadog, Loki).  Each log entry includes:
  - timestamp (ISO8601)
  - log_level
  - event (message)
  - logger (module name)
  - request_id (when available via contextvars)

Usage:
    Call ``setup_logging(log_level)`` once during application startup.
    Then use standard ``logging.getLogger(__name__)`` everywhere — structlog
    wraps stdlib loggers transparently.
"""
from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure structured JSON logging for the entire application.

    Parameters
    ----------
    log_level:
        Root log level string (e.g. "DEBUG", "INFO", "WARNING").
    """
    # Shared processors for both structlog and stdlib loggers.
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            # Prepare for stdlib's formatting (structlog → stdlib bridge)
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Formatter that converts structlog events into JSON strings for stdlib handlers.
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer()
            if log_level.upper() == "DEBUG"
            else structlog.processors.JSONRenderer(),
        ],
    )

    # Apply to root logger
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(log_level.upper())

    # Suppress noisy third-party loggers
    for name in ("httpx", "httpcore", "neo4j", "urllib3", "asyncio"):
        logging.getLogger(name).setLevel(logging.WARNING)
