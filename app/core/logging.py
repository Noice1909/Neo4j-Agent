"""
Structured logging configuration using structlog.

Produces coloured, column-aligned log lines::

    06:39:58 │ main.py            │  INFO  │ [1/8] Initialising Neo4j...

Columns: Time (cyan) │ File (magenta) │ Level (colour-coded) │ Message.
"""
from __future__ import annotations

import logging
import os
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog


# ── ANSI colour helpers ──────────────────────────────────────────────────

_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BOLD_RED = "\033[1;31m"
_WHITE = "\033[37m"

_LEVEL_COLOURS: dict[str, str] = {
    "debug": _DIM,
    "info": _GREEN,
    "warning": _YELLOW,
    "error": _RED,
    "critical": _BOLD_RED,
}

_FILE_WIDTH = 20      # pad / truncate the file column
_LEVEL_WIDTH = 8      # pad the level column
_SEP = f"{_DIM}│{_RESET}"


def _module_to_filename(logger_name: str) -> str:
    """Convert a dotted module path to a short filename, e.g. ``app.main`` → ``main.py``."""
    if not logger_name:
        return "<unknown>"
    last = logger_name.rsplit(".", 1)[-1]
    return f"{last}.py"


class ColumnRenderer:
    """structlog processor that renders each event as a fixed-width, coloured row.

    Intended as the **final** processor in a ``ProcessorFormatter`` pipeline.
    """

    def __call__(
        self,
        logger: Any,
        method_name: str,
        event_dict: MutableMapping[str, Any],
    ) -> str:
        # ── Extract standard fields ──────────────────────────────────────
        timestamp: str = event_dict.pop("timestamp", "")
        # Keep only HH:MM:SS from an ISO timestamp (or the raw value)
        if "T" in timestamp:
            timestamp = timestamp.split("T", 1)[1][:8]

        level: str = event_dict.pop("level", method_name).upper()
        logger_name: str = event_dict.pop("logger", "")
        event: str = event_dict.pop("event", "")

        filename = _module_to_filename(logger_name)

        # ── Colour each column ───────────────────────────────────────────
        level_colour = _LEVEL_COLOURS.get(level.lower(), _WHITE)

        col_time = f"{_CYAN}{timestamp}{_RESET}"
        col_file = f"{_MAGENTA}{filename:<{_FILE_WIDTH}}{_RESET}"
        col_level = f"{level_colour}{level:^{_LEVEL_WIDTH}}{_RESET}"
        col_event = f"{_WHITE}{event}{_RESET}"

        # ── Extra context key=value pairs (structlog bindings) ───────────
        extras = ""
        # Remove internal keys that shouldn't be displayed
        for key in ("_record", "_from_structlog"):
            event_dict.pop(key, None)
        if event_dict:
            pairs = " ".join(f"{k}={v}" for k, v in event_dict.items())
            extras = f"  {_DIM}{pairs}{_RESET}"

        return f"{col_time} {_SEP} {col_file} {_SEP} {col_level} {_SEP} {col_event}{extras}"


# ── Public API ───────────────────────────────────────────────────────────


def setup_logging(log_level: str = "INFO", settings: Any = None) -> None:
    """Configure structured, coloured column logging for the entire application.

    Parameters
    ----------
    log_level:
        Root log level string (e.g. "DEBUG", "INFO", "WARNING").
    settings:
        Optional ``Settings`` instance.  When provided, all ``.env`` values
        are automatically masked in every log message.
    """
    # Enable ANSI escape processing on Windows 10+ consoles
    if sys.platform == "win32":
        os.system("")  # noqa: S605  — triggers ENABLE_VIRTUAL_TERMINAL_PROCESSING

    # ── Sensitive data masking ────────────────────────────────────────────
    if settings is not None:
        from app.core.masking import (
            init_masking, install_stream_masking, mask_sensitive_processor,
        )
        init_masking(settings)
        install_stream_masking()
        masking_processors: list[structlog.types.Processor] = [mask_sensitive_processor]
    else:
        masking_processors = []

    # Shared processors for both structlog and stdlib loggers.
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
            # Prepare for stdlib's formatting (structlog → stdlib bridge)
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Formatter using the custom column renderer for all log levels.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            ColumnRenderer(),
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
    for name in ("httpx", "httpcore", "neo4j", "urllib3", "asyncio", "aiosqlite"):
        logging.getLogger(name).setLevel(logging.WARNING)
