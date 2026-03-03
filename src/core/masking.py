"""
Sensitive data masking for log output.

Collects all string values from ``Settings`` (loaded from ``.env``) and
replaces them everywhere they appear in terminal output.  This ensures that
credentials, URIs, API keys, and other secrets never appear in plaintext.

The primary mechanism is ``MaskingStream`` — a wrapper around ``sys.stderr``
and ``sys.stdout`` that masks every string written to the terminal.  This is
100% formatter/handler-agnostic and catches everything.
"""
from __future__ import annotations

import re
import sys
from typing import Any, TextIO

# Fields whose values are never sensitive (safe to log as-is).
_SAFE_FIELDS: frozenset[str] = frozenset({
    "app_name",
    "debug",
    "log_level",
    "checkpointer_backend",
    "mcp_path",
    "entity_resolution_enabled",
    "query_dedup_enabled",
})

# Minimum length for a value to be worth masking.
_MIN_MASK_LENGTH = 6


def mask_value(value: str) -> str:
    """
    Mask a sensitive string, keeping only edge characters for recognition.

    - len ≤ 3  → ``***``
    - len 4-6  → first char + ``***`` + last char
    - len ≥ 7  → first 2 chars + ``****`` + last 2 chars
    """
    n = len(value)
    if n <= 3:
        return "***"
    if n <= 6:
        return f"{value[0]}***{value[-1]}"
    return f"{value[:2]}****{value[-2:]}"


def build_sensitive_map(settings: Any) -> dict[str, str]:
    """
    Build ``{raw_value: masked_value}`` from all Settings string fields.

    Sorted longest-first so longer matches are replaced first.
    """
    sensitive: dict[str, str] = {}

    for field_name, value in settings.model_dump().items():
        if not isinstance(value, str):
            continue
        if field_name in _SAFE_FIELDS:
            continue
        if len(value) < _MIN_MASK_LENGTH:
            continue
        if value.replace(".", "", 1).isdigit():
            continue
        sensitive[value] = mask_value(value)

    return dict(sorted(sensitive.items(), key=lambda kv: len(kv[0]), reverse=True))


# ── Module-level state ──────────────────────────────────────────────────

_sensitive_map: dict[str, str] = {}
_sensitive_pattern: re.Pattern[str] | None = None


def init_masking(settings: Any) -> None:
    """Build the sensitive-value map and compiled regex.  Call once at startup."""
    global _sensitive_map, _sensitive_pattern

    _sensitive_map = build_sensitive_map(settings)
    if _sensitive_map:
        escaped = [re.escape(raw) for raw in _sensitive_map]
        _sensitive_pattern = re.compile("|".join(escaped))


def _replace_secrets(text: str) -> str:
    """Replace all known sensitive values in *text*."""
    if not _sensitive_pattern:
        return text
    return _sensitive_pattern.sub(
        lambda m: _sensitive_map.get(m.group(0), m.group(0)),
        text,
    )


# ── MaskingStream — wraps sys.stderr / sys.stdout ────────────────────────


class MaskingStream:
    """
    A stream wrapper that masks sensitive values in every ``write()`` call.

    Usage::

        sys.stderr = MaskingStream(sys.stderr)
        sys.stdout = MaskingStream(sys.stdout)

    This is the **primary masking mechanism**.  It operates below the logging
    framework, below structlog, below uvicorn — on the raw stream.  Every
    character that hits the terminal passes through ``_replace_secrets()``.
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def write(self, text: str) -> int:
        masked = _replace_secrets(text) if isinstance(text, str) else text
        return self._stream.write(masked)

    def flush(self) -> None:
        self._stream.flush()

    def fileno(self) -> int:
        return self._stream.fileno()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def __getattr__(self, name: str) -> Any:
        # Proxy everything else (readline, encoding, etc.) to the real stream
        return getattr(self._stream, name)


def install_stream_masking() -> None:
    """
    Wrap ``sys.stderr`` and ``sys.stdout`` with ``MaskingStream``.

    Safe to call multiple times — skips if already wrapped.
    """
    if not isinstance(sys.stderr, MaskingStream):
        sys.stderr = MaskingStream(sys.stderr)  # type: ignore[assignment]
    if not isinstance(sys.stdout, MaskingStream):
        sys.stdout = MaskingStream(sys.stdout)  # type: ignore[assignment]


# ── structlog processor (belt-and-suspenders) ────────────────────────────


def mask_sensitive_processor(
    logger: Any, method_name: str, event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Structlog processor: mask ``.env`` values in log events."""
    if not _sensitive_map:
        return event_dict

    event = event_dict.get("event")
    if isinstance(event, str):
        event_dict["event"] = _replace_secrets(event)

    for key, val in event_dict.items():
        if key == "event":
            continue
        if isinstance(val, str):
            event_dict[key] = _replace_secrets(val)

    return event_dict