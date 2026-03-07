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


# ── Property synonym map ─────────────────────────────────────────────────────


def _property_pattern_synonyms(prop_name: str) -> list[str]:
    """Generate pattern-based synonyms for a property name."""
    synonyms: list[str] = [prop_name.lower()]

    # Underscore split: movie_type → "movie type", "type", "movie"
    if "_" in prop_name:
        natural = prop_name.replace("_", " ").lower()
        synonyms.append(natural)
        for word in natural.split():
            if len(word) >= 3:
                synonyms.append(word)

    # CamelCase split: releaseDate → "release date", "release", "date"
    parts = _camel_parts(prop_name)
    if len(parts) > 1:
        synonyms.append(" ".join(p.lower() for p in parts))
        for p in parts:
            if len(p) >= 3:
                synonyms.append(p.lower())

    # Plural
    synonyms.append(_plural(prop_name.lower()))

    return list(dict.fromkeys(synonyms))  # deduplicate preserving order


def build_property_synonym_map(
    topology: "GraphTopology",
    semantic_layer: "SchemaSemanticLayer | None" = None,
) -> "dict[str, tuple[str, str]]":
    """
    Build a synonym map for properties: nl_term → (label, property_name).

    Layers (highest priority wins):
      A. Pattern-based: underscore_split, camelCase_split, plural
      B. Semantic layer (LLM-generated at startup)
      C. Property nlp_terms from topology (enriched by semantic layer)

    Parameters
    ----------
    topology:
        The live graph topology with label + property info.
    semantic_layer:
        Optional SchemaSemanticLayer from the startup LLM call.

    Returns
    -------
    dict
        Lowercase NL term → (label, property_name) mapping.
    """
    from src.graph.topology import GraphTopology
    if semantic_layer is not None:
        from src.graph.semantic_layer import SchemaSemanticLayer

    result: dict[str, tuple[str, str]] = {}

    # Layer A: Pattern-based
    for li in topology.labels:
        for prop in li.properties:
            for syn in _property_pattern_synonyms(prop):
                if syn and len(syn) >= 3:
                    result[syn] = (li.label, prop)

    # Layer B: Semantic layer NL terms (LLM-generated)
    if semantic_layer is not None:
        for label, props in semantic_layer.property_semantics.items():
            for ps in props:
                for nl_name in ps.natural_names:
                    key = nl_name.lower().strip()
                    if key and len(key) >= 2:
                        result[key] = (ps.label, ps.property_name)

    # Layer C: Property nlp_terms from topology
    for li in topology.labels:
        for prop, terms in li.property_nlp_terms.items():
            for term in terms:
                key = term.lower().strip()
                if key:
                    result[key] = (li.label, prop)

    logger.debug("Property synonym map built with %d entries.", len(result))
    return result


# ── Relationship synonym map ────────────────────────────────────────────────


def _rel_pattern_synonyms(rel_type: str) -> list[str]:
    """Generate pattern-based synonyms for a relationship type."""
    synonyms: list[str] = [rel_type.lower()]

    # Underscore to space: ACTED_IN → "acted in"
    if "_" in rel_type:
        natural = rel_type.replace("_", " ").lower()
        synonyms.append(natural)
        for word in natural.split():
            if len(word) >= 3:
                synonyms.append(word)

    # Remove common prefixes: HAS_GENRE → "genre"
    for prefix in ("has_", "is_", "was_", "belongs_to_"):
        if rel_type.lower().startswith(prefix):
            stripped = rel_type[len(prefix):].replace("_", " ").lower().strip()
            if stripped:
                synonyms.append(stripped)

    return list(dict.fromkeys(synonyms))  # deduplicate


def build_relationship_synonym_map(
    topology: "GraphTopology",
    semantic_layer: "SchemaSemanticLayer | None" = None,
) -> "dict[str, str]":
    """
    Build a synonym map for relationships: nl_phrase → rel_type.

    Layers:
      A. Pattern-based: underscore_to_space, strip common prefixes
      B. Semantic layer (LLM-generated at startup)

    Parameters
    ----------
    topology:
        The live graph topology with relationship triples.
    semantic_layer:
        Optional SchemaSemanticLayer from the startup LLM call.

    Returns
    -------
    dict
        Lowercase NL phrase → rel_type mapping.
    """
    result: dict[str, str] = {}

    # Layer A: Pattern-based
    for t in topology.triples:
        for syn in _rel_pattern_synonyms(t.rel_type):
            if syn and len(syn) >= 3:
                result[syn] = t.rel_type

    # Layer B: Semantic layer NL phrases
    if semantic_layer is not None:
        for rs in semantic_layer.relationship_semantics:
            for phrase in rs.natural_phrases:
                key = phrase.lower().strip()
                if key:
                    result[key] = rs.rel_type

    logger.debug("Relationship synonym map built with %d entries.", len(result))
    return result
