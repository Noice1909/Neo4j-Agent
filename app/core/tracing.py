"""
Session journey tracing — per-chat-turn pipeline visibility.

Provides a ``SessionTracer`` that collects events as a request flows through
the pipeline (dedup → agent → coreference → entity resolution → Cypher
generation → validation → execution → correction → answer) and prints a
coloured bordered summary block in the terminal after each chat turn.

Propagation is via ``contextvars.ContextVar`` so any module in the async
call-stack can call ``get_tracer()`` without explicit parameter threading.

Example output::

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║  SESSION TRACE — sess-abc123…  │  turn t-d4e5f6…                           ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║   #  STAGE                 STATUS     DETAIL                       ELAPSED  ║
    ║   1  USER_INPUT            ✓          "Who directed The Matrix?"      0ms   ║
    ║   2  …                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional


# ── ANSI colour helpers (re-use palette from logging.py) ─────────────────────

_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_WHITE = "\033[37m"
_BOLD_CYAN = "\033[1;36m"
_BOLD_GREEN = "\033[1;32m"
_BOLD_YELLOW = "\033[1;33m"
_BOLD_RED = "\033[1;31m"


_STATUS_STYLE = {
    "ok":   (_GREEN,  "✓"),
    "skip": (_DIM,    "—"),
    "fail": (_RED,    "✗"),
    "warn": (_YELLOW, "⚠"),
    "info": (_CYAN,   "ℹ"),
}


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class TraceEvent:
    """A single event in the session journey timeline."""

    stage: str
    status: str          # ok | skip | fail | warn | info
    detail: str = ""
    timestamp: float = field(default_factory=time.perf_counter)
    elapsed_ms: float = 0.0  # filled in by SessionTracer


@dataclass
class SessionTracer:
    """Collects pipeline events for one chat turn and renders a summary."""

    session_id: str
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    _events: list[TraceEvent] = field(default_factory=list)
    _start: float = field(default_factory=time.perf_counter)

    # ── Recording ─────────────────────────────────────────────────────────

    def record(
        self,
        stage: str,
        status: str = "ok",
        detail: str = "",
    ) -> None:
        """Append an event.  ``elapsed_ms`` is computed from the tracer start."""
        now = time.perf_counter()
        evt = TraceEvent(
            stage=stage,
            status=status,
            detail=detail,
            timestamp=now,
            elapsed_ms=round((now - self._start) * 1000, 1),
        )
        self._events.append(evt)

    # ── Rendering ─────────────────────────────────────────────────────────

    def print_journey(self) -> None:
        """Print a coloured, bordered journey summary to stderr."""
        if not self._events:
            return

        # Enable ANSI on Windows
        if sys.platform == "win32":
            import os
            os.system("")  # noqa: S605

        total_ms = round((time.perf_counter() - self._start) * 1000, 1)

        # ── Column widths ─────────────────────────────────────────────────
        COL_NUM = 4       # " 1 "
        COL_STAGE = 26    # stage name
        COL_STATUS = 10   # ✓ / ✗ / —
        COL_DETAIL = 48   # truncated detail
        COL_TIME = 10     # elapsed
        # inner width = sum + separators
        INNER = COL_NUM + COL_STAGE + COL_STATUS + COL_DETAIL + COL_TIME + 4  # 4 spaces between cols
        BOX_W = INNER + 4  # ║ + space on each side

        def _pad(text: str, width: int) -> str:
            """Pad or truncate *visible* text to *width* chars."""
            if len(text) > width:
                return text[: width - 1] + "…"
            return text.ljust(width)

        # ── Header ────────────────────────────────────────────────────────
        sess_short = self.session_id[:12] + ("…" if len(self.session_id) > 12 else "")
        header_text = f"SESSION TRACE — {sess_short}  │  turn {self.turn_id}"
        header_line = f"  {_BOLD_CYAN}{_pad(header_text, INNER)}{_RESET}"

        lines: list[str] = []
        border_top =    f"  {_DIM}╔{'═' * (BOX_W - 2)}╗{_RESET}"
        border_mid =    f"  {_DIM}╠{'═' * (BOX_W - 2)}╣{_RESET}"
        border_bot =    f"  {_DIM}╚{'═' * (BOX_W - 2)}╝{_RESET}"
        l_edge = f"{_DIM}║{_RESET} "
        r_edge = f" {_DIM}║{_RESET}"

        lines.append("")
        lines.append(border_top)
        lines.append(f"  {l_edge}{header_line}{r_edge}")
        lines.append(border_mid)

        # Column headers
        col_hdr = (
            f"{_BOLD}{_pad('#', COL_NUM)} "
            f"{_pad('STAGE', COL_STAGE)} "
            f"{_pad('STATUS', COL_STATUS)} "
            f"{_pad('DETAIL', COL_DETAIL)} "
            f"{_pad('ELAPSED', COL_TIME)}{_RESET}"
        )
        lines.append(f"  {l_edge}{col_hdr}{r_edge}")
        sep_line = f"  {l_edge}{_DIM}{'-' * INNER}{_RESET}{r_edge}"
        lines.append(sep_line)

        # Stages whose detail should be shown in full on continuation lines
        _MULTILINE_STAGES = {"CYPHER_GENERATED", "CYPHER_CORRECTION"}
        _CONT_INDENT = 8  # left margin for continuation lines
        _CONT_WIDTH = INNER - _CONT_INDENT  # usable chars per wrap line

        # ── Event rows ────────────────────────────────────────────────────
        for idx, evt in enumerate(self._events, 1):
            colour, symbol = _STATUS_STYLE.get(evt.status, (_WHITE, "?"))
            num_str = _pad(str(idx), COL_NUM)
            stage_str = _pad(evt.stage, COL_STAGE)
            status_str = f"{colour}{_pad(symbol, COL_STATUS)}{_RESET}"
            detail_str = _pad(evt.detail, COL_DETAIL)
            time_str = _pad(f"{evt.elapsed_ms:>7.0f}ms", COL_TIME)

            row = f"{num_str} {stage_str} {status_str} {detail_str} {_DIM}{time_str}{_RESET}"
            lines.append(f"  {l_edge}{row}{r_edge}")

            # Continuation lines for full Cypher query
            if evt.stage in _MULTILINE_STAGES and len(evt.detail) > COL_DETAIL:
                full = evt.detail
                for i in range(0, len(full), _CONT_WIDTH):
                    chunk = full[i:i + _CONT_WIDTH]
                    cont = f"{' ' * _CONT_INDENT}{_MAGENTA}{_pad(chunk, INNER - _CONT_INDENT)}{_RESET}"
                    lines.append(f"  {l_edge}{cont}{r_edge}")

        # ── Total row ─────────────────────────────────────────────────────
        lines.append(sep_line)
        total_row = (
            f"{_pad('', COL_NUM)} "
            f"{_BOLD}{_pad('TOTAL', COL_STAGE)}{_RESET} "
            f"{_pad('', COL_STATUS)} "
            f"{_pad('', COL_DETAIL)} "
            f"{_BOLD_GREEN}{_pad(f'{total_ms:>7.0f}ms', COL_TIME)}{_RESET}"
        )
        lines.append(f"  {l_edge}{total_row}{r_edge}")
        lines.append(border_bot)
        lines.append("")

        output = "\n".join(lines)
        sys.stderr.write(output + "\n")
        sys.stderr.flush()


# ── Context-var helpers ──────────────────────────────────────────────────────

_tracer_var: ContextVar[Optional[SessionTracer]] = ContextVar(
    "session_tracer", default=None,
)


def set_tracer(tracer: SessionTracer) -> None:
    """Bind *tracer* to the current async context."""
    _tracer_var.set(tracer)


def get_tracer() -> SessionTracer | None:
    """Return the ``SessionTracer`` bound to the current context, or ``None``."""
    return _tracer_var.get()


def trace_event(stage: str, status: str = "ok", detail: str = "") -> None:
    """Convenience: record an event on the current context tracer (no-op if unset)."""
    tracer = _tracer_var.get()
    if tracer is not None:
        tracer.record(stage, status, detail)
