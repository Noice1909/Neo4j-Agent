"""
Coreference resolution for follow-up questions (Strategy #5).
"""
from __future__ import annotations

import asyncio
import logging
import re

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# Static trigger phrases always active — independent of schema.
_STATIC_PATTERN = (
    r"those|these|them|they|the\s+same|the\s+above|mentioned|previous"
)

_coref_re = re.compile(
    rf"\b(?:{_STATIC_PATTERN})\b",
    re.IGNORECASE,
)


# ── Dynamic regex builder ─────────────────────────────────────────────────────


def _label_tokens(label: str) -> set[str]:
    """Return regex-escaped token variants for one schema label."""
    tokens: set[str] = {re.escape(label.lower())}
    parts = re.findall(r"[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)", label)
    if len(parts) > 1:
        tokens.add(re.escape(" ".join(p.lower() for p in parts)))
    for p in parts:
        if len(p) >= 3:
            tokens.add(re.escape(p.lower()))
    return tokens


def _nlp_tokens(nlp_terms_by_label: dict[str, list[str]]) -> set[str]:
    """Return regex-escaped tokens from Concept nlp_terms (min 3 chars)."""
    return {
        re.escape(term.lower().strip())
        for terms in nlp_terms_by_label.values()
        for term in terms
        if len(term.strip()) >= 3
    }


def build_coreference_regex(
    schema_labels: list[str],
    nlp_terms_by_label: dict[str, list[str]] | None = None,
) -> re.Pattern[str]:
    """
    Build a coreference pattern that includes ``that <label>`` variants for
    every label in *schema_labels*, plus any curated NLP terms from Concept nodes.

    CamelCase labels are split so ``SanitizedTable`` also generates
    ``that sanitized table``.  The static trigger phrases are always included.

    Parameters
    ----------
    schema_labels:
        Canonical Neo4j node labels from the live schema.
    nlp_terms_by_label:
        Optional mapping of label → nlp_terms from ``topology.nlp_terms_by_label``.
        These terms (e.g. ``"app"``, ``"repo"``) are added as coreference triggers.
    """
    label_tokens: set[str] = set()
    for label in schema_labels:
        label_tokens |= _label_tokens(label)
    if nlp_terms_by_label:
        label_tokens |= _nlp_tokens(nlp_terms_by_label)

    if label_tokens:
        that_clause = r"that\s+(?:" + "|".join(sorted(label_tokens)) + r")"
        pattern = rf"\b(?:{_STATIC_PATTERN}|{that_clause})\b"
    else:
        pattern = rf"\b(?:{_STATIC_PATTERN})\b"

    return re.compile(pattern, re.IGNORECASE)


def set_coreference_regex(regex: re.Pattern[str]) -> None:
    """
    Replace the module-level ``_coref_re`` with *regex*.

    Call this at startup after topology extraction and on every topology
    refresh so coreference detection reflects the live schema labels.
    """
    global _coref_re
    _coref_re = regex
    logger.info("Coreference regex updated.")


# ── Public API ────────────────────────────────────────────────────────────────


def has_coreferences(question: str) -> bool:
    """Detect referential / anaphoric language in a question."""
    return bool(_coref_re.search(question))


async def resolve_coreferences(
    question: str,
    conversation_context: str | None,
    llm: BaseChatModel,
) -> str:
    """Rewrite a follow-up question as standalone by resolving coreferences."""
    from app.core.tracing import trace_event

    if not conversation_context or not has_coreferences(question):
        trace_event("COREFERENCE", "skip", "No coreferences detected")
        return question

    rewrite_prompt = (
        "Rewrite the following follow-up question as a fully self-contained "
        "question.  Replace all pronouns and references (those, these, them, "
        "they, it, etc.) with the actual entity names from the context.\n\n"
        f"Context from conversation:\n{conversation_context}\n\n"
        f"Follow-up question: {question}\n\n"
        "Rewritten standalone question (output ONLY the question, nothing else):"
    )

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None, lambda: llm.invoke(rewrite_prompt)
    )
    rewritten = str(response.content).strip()

    if rewritten and len(rewritten) > 5 and not rewritten.lower().startswith("i "):
        logger.info("Coreference resolved: %r → %r", question, rewritten)
        trace_event("COREFERENCE", "ok", f"{question[:50]} → {rewritten[:50]}")
        return rewritten

    logger.warning("Coreference resolution produced dubious result; using original")
    trace_event("COREFERENCE", "warn", "Dubious result; kept original")
    return question
