"""
Synonym mapping for Neo4j entity labels.

Generates synonyms automatically from live schema labels — nothing is
hardcoded for any specific domain.  Three layers, highest priority wins:

  Layer A — Pattern-based auto-generation (``auto_generate_synonyms``)
  Layer B — LLM-powered enrichment (``llm_generate_synonyms``, one-time at startup)
  Layer C — Env-var overrides (``ENTITY_SYNONYM_OVERRIDES`` JSON string)

The synonym map is keyed by **lowercase alias** → **canonical label**.
"""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


# ── Pattern-based helpers ─────────────────────────────────────────────────────

def _plural(word: str) -> str:
    """Return a naive plural of *word*."""
    if word.endswith("y"):
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def _camel_parts(label: str) -> list[str]:
    """Split a CamelCase or PascalCase label into its component words."""
    return re.findall(r"[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)", label)


def _synonyms_for_label(label: str) -> Dict[str, str]:
    """
    Return all pattern-derived synonyms for a single *label*.

    Covers: lowercase, plural, CamelCase split (space + underscore),
    individual meaningful words, underscore-delimited natural form,
    uppercase acronym, and short abbreviation.
    """
    result: Dict[str, str] = {}
    lower = label.lower()
    result[lower] = label
    result[_plural(lower)] = label

    parts = _camel_parts(label)
    if len(parts) > 1:
        result[" ".join(p.lower() for p in parts)] = label
        result["_".join(p.lower() for p in parts)] = label
        for p in parts:
            if len(p) >= 3:
                result[p.lower()] = label

    if "_" in label:
        natural = label.replace("_", " ").lower()
        result[natural] = label
        for word in natural.split():
            if len(word) >= 3:
                result[word] = label

    acronym = "".join(c for c in label if c.isupper()).lower()
    if len(acronym) >= 2:
        result[acronym] = label

    if len(label) >= 6:
        result[label[:3].lower()] = label

    return result


# ── Pattern-based auto-generation ────────────────────────────────────────────


def auto_generate_synonyms(schema_labels: list[str]) -> Dict[str, str]:
    """
    Auto-generate synonyms from actual schema labels using pattern rules.

    Parameters
    ----------
    schema_labels:
        Node labels and/or relationship type strings from the live schema.

    Returns
    -------
    dict
        Lowercase-alias → canonical-label mapping.
    """
    synonyms: Dict[str, str] = {}
    for label in schema_labels:
        synonyms.update(_synonyms_for_label(label))
    return synonyms


# ── LLM-powered enrichment ────────────────────────────────────────────────────

_LLM_SYNONYM_PROMPT = """\
You are a database assistant. For each Neo4j node label below, list 3-5 short
natural-language aliases that a non-technical user might type in a chat interface.

Labels: {labels}

Respond with ONLY a JSON object where every key is a lowercase alias and the
value is the exact label it maps to.  No explanation, no markdown fences.
Example output: {{"app": "Application", "domain area": "Domain"}}"""


async def llm_generate_synonyms(
    label_names: list[str],
    llm: "BaseChatModel",
) -> Dict[str, str]:
    """
    Ask the LLM to generate natural-language aliases for each label.

    Called once at startup and cached with the topology.  Returns an empty
    dict on failure so the caller can always proceed safely.

    Parameters
    ----------
    label_names:
        Canonical Neo4j node labels from the live schema.
    llm:
        Any LangChain ``BaseChatModel`` instance.
    """
    if not label_names:
        return {}

    prompt = _LLM_SYNONYM_PROMPT.format(labels=", ".join(label_names))
    try:
        response = await llm.ainvoke(prompt)
        raw: str = str(response.content) if hasattr(response, "content") else str(response)

        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")

        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object.")

        result: Dict[str, str] = {
            k.lower().strip(): str(v)
            for k, v in parsed.items()
            if isinstance(v, str)
        }
        logger.info("LLM synonym enrichment: %d aliases generated.", len(result))
        return result
    except Exception as exc:
        logger.warning("LLM synonym enrichment failed (non-fatal): %s", exc)
        return {}


# ── Final map builder ──────────────────────────────────────────────────────────


def build_synonym_map(
    schema_labels: list[str],
    overrides_json: Optional[str] = None,
    llm_synonyms: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Build the final synonym map by merging all three layers.

    Priority (highest wins): env-var overrides > LLM-generated > pattern-based.

    Parameters
    ----------
    schema_labels:
        Actual Neo4j node labels from the schema.
    overrides_json:
        Optional JSON string of custom synonym overrides
        (``ENTITY_SYNONYM_OVERRIDES`` env var).
    llm_synonyms:
        Pre-generated LLM synonyms dict (from ``llm_generate_synonyms``).

    Returns
    -------
    dict
        Complete lowercase-alias → canonical-label mapping.
    """
    # Layer A: pattern-based
    result: Dict[str, str] = auto_generate_synonyms(schema_labels)

    # Layer B: LLM-generated (only accepted for known labels)
    if llm_synonyms:
        valid_labels = set(schema_labels)
        for alias, label in llm_synonyms.items():
            if label in valid_labels:
                result[alias.lower()] = label
        logger.debug("Applied %d LLM-generated synonyms.", len(llm_synonyms))

    # Layer C: env-var overrides (highest priority, no label validation)
    if overrides_json:
        try:
            overrides: object = json.loads(overrides_json)
            if isinstance(overrides, dict):
                result.update({str(k).lower(): str(v) for k, v in overrides.items()})
                logger.info("Loaded %d custom synonym override(s).", len(overrides))
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse ENTITY_SYNONYM_OVERRIDES: %s", exc)

    logger.debug("Synonym map built with %d entries.", len(result))
    return result
