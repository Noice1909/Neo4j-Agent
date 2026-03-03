"""
Message trimming utility for context window management.

Trims conversation history to fit within a token budget before sending
to the LLM.  The full history remains in the checkpoint — trimming is
applied only at inference time.

Strategy:
  1. Always keep the SystemMessage (index 0).
  2. Always keep the first HumanMessage (topic anchor).
  3. Fill remaining budget with the most recent messages.

Token estimation uses ~4 chars/token by default (good for English text).
Swap ``token_counter`` with ``model.get_num_tokens()`` for exact counting
when a model-specific tokeniser is available (e.g. Gemini, OpenAI).
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Callable

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


# ── Token estimation ──────────────────────────────────────────────────────────


def _estimate_tokens(message: BaseMessage) -> int:
    """Approximate token count for a single message (~4 chars per token)."""
    content = message.content if isinstance(message.content, str) else str(message.content)
    # Account for role/name overhead (~4 tokens) + content
    return 4 + len(content) // 4


# ── Trimming logic ────────────────────────────────────────────────────────────


def trim_conversation(
    messages: Sequence[BaseMessage],
    max_tokens: int = 100_000,
    token_counter: Callable[[BaseMessage], int] | None = None,
) -> list[BaseMessage]:
    """
    Trim a message list to fit within *max_tokens*.

    The full checkpoint history is never mutated — only the list passed to
    ``model.ainvoke()`` is shortened.

    Parameters
    ----------
    messages:
        Full conversation history (SystemMessage should be at index 0).
    max_tokens:
        Maximum token budget for the returned list.
    token_counter:
        Optional callable that returns the token count for a single message.
        Defaults to a character-based estimator (~4 chars/token).

    Returns
    -------
    list[BaseMessage]
        Trimmed message list that fits within *max_tokens*.
        Preserves: system prompt, first human message, and the newest
        messages that fit in the remaining budget.
    """
    if not messages:
        return list(messages)

    counter = token_counter or _estimate_tokens

    # ── Identify pinned messages ──────────────────────────────────────────
    pinned: list[tuple[int, BaseMessage]] = []  # (original_index, message)
    pinned_indices: set[int] = set()

    # Pin the system prompt (always index 0 if present)
    if isinstance(messages[0], SystemMessage):
        pinned.append((0, messages[0]))
        pinned_indices.add(0)

    # Pin the first HumanMessage (topic anchor)
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage) and i not in pinned_indices:
            pinned.append((i, msg))
            pinned_indices.add(i)
            break

    # ── Calculate pinned token cost ───────────────────────────────────────
    pinned_tokens = sum(counter(msg) for _, msg in pinned)

    if pinned_tokens >= max_tokens:
        # Even pinned messages exceed budget — return them anyway (best effort)
        logger.warning(
            "Pinned messages alone (%d tokens) exceed budget (%d tokens).",
            pinned_tokens,
            max_tokens,
        )
        return [msg for _, msg in pinned]

    remaining_budget = max_tokens - pinned_tokens

    # ── Fill from the tail (newest first) ─────────────────────────────────
    # Walk backwards through non-pinned messages, adding while budget allows.
    tail_messages: list[BaseMessage] = []
    for i in range(len(messages) - 1, -1, -1):
        if i in pinned_indices:
            continue
        cost = counter(messages[i])
        if cost > remaining_budget:
            break  # Stop — can't fit this message
        tail_messages.append(messages[i])
        remaining_budget -= cost

    tail_messages.reverse()  # Restore chronological order

    # ── Assemble final list ───────────────────────────────────────────────
    # Pinned messages first (in original order), then the tail.
    pinned_msgs = [msg for _, msg in pinned]
    result = pinned_msgs + tail_messages

    dropped = len(messages) - len(result)
    if dropped > 0:
        logger.info(
            "Trimmed conversation: kept %d/%d messages (dropped %d, budget=%d tokens).",
            len(result),
            len(messages),
            dropped,
            max_tokens,
        )

    return result
