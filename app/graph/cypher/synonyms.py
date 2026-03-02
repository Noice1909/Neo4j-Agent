"""
Synonym mapping for Neo4j entity labels.

Provides both a static default synonym map and auto-generation of common
synonyms from schema labels.  Custom overrides can be injected via the
``ENTITY_SYNONYM_OVERRIDES`` env var (JSON string).

The synonym map is keyed by **lowercase alias** → **canonical label**.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)

# ── Default synonym map ──────────────────────────────────────────────────────
# Common aliases that users might type instead of actual Neo4j labels.
# Keys are lowercase; values are the canonical Neo4j label.
DEFAULT_SYNONYMS: Dict[str, str] = {
    # Movie domain
    "film": "Movie",
    "films": "Movie",
    "movies": "Movie",
    "flick": "Movie",
    "flicks": "Movie",
    "picture": "Movie",
    "pictures": "Movie",
    # Person domain
    "actor": "Person",
    "actors": "Person",
    "actress": "Person",
    "actresses": "Person",
    "director": "Person",
    "directors": "Person",
    "people": "Person",
    "persons": "Person",
    # Generic aliases
    "app": "Application",
    "apps": "Application",
    "application": "Application",
    "applications": "Application",
    "sor": "Application",
    "sors": "Application",
    "system": "Application",
    "systems": "Application",
    "service": "Application",
    "services": "Application",
}


def auto_generate_synonyms(schema_labels: list[str]) -> Dict[str, str]:
    """
    Auto-generate obvious synonyms from actual schema labels.

    For each label, generates:
    - Lowercase form
    - Plural/singular forms (naive heuristic)
    - Underscore/camelCase splits

    Parameters
    ----------
    schema_labels:
        List of actual Neo4j node labels (e.g. ["Movie", "Person"]).

    Returns
    -------
    dict
        Mapping of lowercase alias → canonical label.
    """
    synonyms: Dict[str, str] = {}

    for label in schema_labels:
        lower = label.lower()
        synonyms[lower] = label

        # Naive plural: "Movie" → "movies"
        if lower.endswith("y"):
            synonyms[lower[:-1] + "ies"] = label
        elif lower.endswith(("s", "x", "z", "ch", "sh")):
            synonyms[lower + "es"] = label
        else:
            synonyms[lower + "s"] = label

        # CamelCase split: "MovieGenre" → "movie genre", "movie_genre"
        parts = re.findall(r"[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)", label)
        if len(parts) > 1:
            joined_space = " ".join(p.lower() for p in parts)
            joined_underscore = "_".join(p.lower() for p in parts)
            synonyms[joined_space] = label
            synonyms[joined_underscore] = label

    return synonyms


def build_synonym_map(
    schema_labels: list[str],
    overrides_json: str = "",
) -> Dict[str, str]:
    """
    Build the final synonym map by merging defaults, auto-generated, and overrides.

    Priority (highest wins): overrides > auto-generated > defaults.

    Parameters
    ----------
    schema_labels:
        Actual Neo4j node labels from the schema.
    overrides_json:
        Optional JSON string of custom synonym overrides
        (e.g. ``'{"sor": "Application", "db": "Database"}'``).

    Returns
    -------
    dict
        Complete lowercase-alias → canonical-label mapping.
    """
    # Start with defaults
    result: Dict[str, str] = dict(DEFAULT_SYNONYMS)

    # Layer auto-generated synonyms on top
    result.update(auto_generate_synonyms(schema_labels))

    # Layer user overrides on top (highest priority)
    if overrides_json and overrides_json.strip():
        try:
            overrides = json.loads(overrides_json)
            if isinstance(overrides, dict):
                # Normalise keys to lowercase
                result.update({k.lower(): v for k, v in overrides.items()})
                logger.info(
                    "Loaded %d custom synonym override(s).", len(overrides),
                )
        except json.JSONDecodeError as exc:
            logger.warning(
                "Failed to parse ENTITY_SYNONYM_OVERRIDES: %s", exc,
            )

    logger.debug("Synonym map built with %d entries.", len(result))
    return result
