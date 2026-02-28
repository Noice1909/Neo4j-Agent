"""
LLM provider factory — plug-and-play.

The entire application references LLMs only through `get_llm()`.
To switch from Ollama to OpenAI or Anthropic, update this file alone.

LLM response caching is configured globally via LangChain's
`set_llm_cache()`.  The cache is initialised in `app/main.py` lifespan
and backed by Redis so all workers share the same cache.

Cache key = (prompt text, model identifier).
Identical prompts against the same model return instantly.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from langchain_ollama import ChatOllama

from app.core.config import Settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm(
    base_url: str,
    model: str,
    temperature: float,
) -> ChatOllama:
    """
    Return a cached `ChatOllama` instance.

    Parameters are explicit (not a Settings object) so ``@lru_cache`` can
    hash them correctly.  Use :func:`get_llm_from_settings` in production.

    Parameters
    ----------
    base_url:
        Ollama server base URL, e.g. ``http://localhost:11434``.
    model:
        Model tag, e.g. ``mistral:latest``.
    temperature:
        Sampling temperature — ``0.0`` is deterministic (best for Cypher).
    """
    logger.info("Creating LLM: model=%s base_url=%s temperature=%s", model, base_url, temperature)
    return ChatOllama(
        base_url=base_url,
        model=model,
        temperature=temperature,
    )


def get_llm_from_settings(settings: Settings) -> ChatOllama:
    """Return an LLM instance configured from application settings."""
    return get_llm(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.ollama_temperature,
    )
