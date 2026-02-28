"""
Coreference resolution for follow-up questions (Strategy #5).

Detects referential / anaphoric language (e.g. "those movies", "them")
and rewrites the question as a standalone query using conversation context.
"""
from __future__ import annotations

import asyncio
import logging
import re

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

_COREF_RE = re.compile(
    r"\b(?:those|these|them|they|"
    r"that\s+(?:movie|film|person|actor|director)|"
    r"the\s+same|the\s+above|mentioned|previous)\b",
    re.IGNORECASE,
)


def has_coreferences(question: str) -> bool:
    """Detect referential / anaphoric language in a question."""
    return bool(_COREF_RE.search(question))


async def resolve_coreferences(
    question: str,
    conversation_context: str | None,
    llm: BaseChatModel,
) -> str:
    """
    Rewrite a follow-up question as standalone by resolving coreferences.

    If *conversation_context* is ``None`` or the question has no coreferences
    the question is returned unchanged.
    """
    if not conversation_context or not has_coreferences(question):
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
        return rewritten

    logger.warning("Coreference resolution produced dubious result; using original")
    return question
