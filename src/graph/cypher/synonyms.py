"""
Synonym mapping for Neo4j entity labels.

Generates synonyms automatically from live schema labels — nothing is
hardcoded for any specific domain.  Three layers, highest priority wins:

  Layer A — Pattern-based auto-generation (``auto_generate_synonyms``)
  Layer B — Concept node nlp_terms (domain-expert curated, from Neo4j)
  Layer C — Env-var overrides (``ENTITY_SYNONYM_OVERRIDES`` JSON string)

The synonym map is keyed by **lowercase alias** → **canonical label**.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, Optional

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


# ── Final map builder ──────────────────────────────────────────────────────────


def build_synonym_map(
    schema_labels: list[str],
    overrides_json: Optional[str] = None,
    concept_nlp_terms: Optional[Dict[str, list[str]]] = None,
) -> Dict[str, str]:
    """
    Build the final synonym map by merging all three layers.

    Priority (highest wins): env-var overrides > Concept nlp_terms > pattern-based.

    Parameters
    ----------
    schema_labels:
        Actual Neo4j node labels from the schema.
    overrides_json:
        Optional JSON string of custom synonym overrides
        (``ENTITY_SYNONYM_OVERRIDES`` env var).
    concept_nlp_terms:
        Per-label nlp_terms from Concept nodes (``topology.nlp_terms_by_label``).

    Returns
    -------
    dict
        Complete lowercase-alias → canonical-label mapping.
    """
    # Layer A: pattern-based
    result: Dict[str, str] = auto_generate_synonyms(schema_labels)

    # Layer B: Concept node nlp_terms (domain-expert curated)
    if concept_nlp_terms:
        valid_labels = set(schema_labels)
        for label, terms in concept_nlp_terms.items():
            if label in valid_labels:
                for term in terms:
                    result[term.lower().strip()] = label
        logger.debug("Applied Concept nlp_terms as synonyms.")

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
