"""
Semantic schema validation for LLM-generated Cypher.

``validate_cypher_schema()`` parses MATCH patterns from generated Cypher and
cross-checks label names, relationship types, and arrow directions against the
live ``GraphTopology``.  It returns human-readable error strings that are fed
directly into the correction prompt so the LLM gets structural guidance instead
of opaque Neo4j runtime errors.
"""
from __future__ import annotations

import difflib
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.graph.topology import GraphTopology

# Matches directed MATCH patterns: (var:Label)-[:REL]->(var:Label)
# Groups: (src_label, rel_type, tgt_label)
_DIRECTED_PATTERN_RE = re.compile(
    r"\(\s*\w*\s*:?\s*(\w+)\s*[^)]*\)\s*-\[.*?:(\w+).*?\]->\s*\(\s*\w*\s*:?\s*(\w+)\s*[^)]*\)",
)


def validate_cypher_schema(
    cypher: str,
    topology: "GraphTopology",
) -> list[str]:
    """
    Parse directed MATCH patterns from *cypher* and cross-check them against
    *topology*.  Returns a list of actionable error strings (empty = no issues).

    Error categories:
    - Direction reversed: the triple exists but arrow is backwards.
    - Wrong endpoint: rel_type exists but not between these two labels.
    - Unknown rel_type: not in the schema at all (suggests closest by name).
    - Unknown label: label not in schema (suggests aliases if applicable).
    """
    if not topology.triples:
        return []

    triple_set = {
        (t.source_label, t.rel_type, t.target_label) for t in topology.triples
    }
    reverse_set = {
        (t.target_label, t.rel_type, t.source_label) for t in topology.triples
    }
    rel_to_triples: dict[str, list[tuple[str, str]]] = {}
    for t in topology.triples:
        rel_to_triples.setdefault(t.rel_type, []).append((t.source_label, t.target_label))

    label_name_set = set(topology.label_names)
    # Flatten aliases for reverse-lookup: alias → primary
    alias_to_primary: dict[str, str] = {}
    for primary, aliases in topology.label_aliases.items():
        for alias in aliases:
            alias_to_primary[alias] = primary

    errors: list[str] = []
    for match in _DIRECTED_PATTERN_RE.finditer(cypher):
        src_label, rel_type, tgt_label = match.group(1), match.group(2), match.group(3)
        _check_pattern(
            src_label, rel_type, tgt_label,
            triple_set, reverse_set, rel_to_triples,
            label_name_set, alias_to_primary, topology,
            errors,
        )
    return errors


def _check_pattern(
    src: str,
    rel: str,
    tgt: str,
    triple_set: set[tuple[str, str, str]],
    reverse_set: set[tuple[str, str, str]],
    rel_to_triples: dict[str, list[tuple[str, str]]],
    label_name_set: set[str],
    alias_to_primary: dict[str, str],
    topology: "GraphTopology",
    errors: list[str],
) -> None:
    """Validate one (src)-[:rel]->(tgt) pattern; append errors in-place."""
    # Normalise via alias map before checking
    src_norm = alias_to_primary.get(src, src)
    tgt_norm = alias_to_primary.get(tgt, tgt)

    if (src_norm, rel, tgt_norm) in triple_set:
        return  # valid

    if (src_norm, rel, tgt_norm) in reverse_set:
        errors.append(
            f"Direction reversed: used ({src})-[:{rel}]->({tgt}) "
            f"but schema has ({tgt})-[:{rel}]->({src}). Swap the arrow direction."
        )
        return

    if rel in rel_to_triples:
        valid_pairs = rel_to_triples[rel]
        pairs_str = ", ".join(f"({s})->[:{rel}]->({t})" for s, t in valid_pairs[:3])
        errors.append(
            f"Relationship [:{rel}] exists but not between ({src}) and ({tgt}). "
            f"Valid path(s): {pairs_str}."
        )
        return

    # Unknown rel_type — suggest closest
    closest = _closest_rel(rel, set(rel_to_triples.keys()), topology)
    errors.append(
        f"Unknown relationship [:{rel}] — not in schema.{closest}"
    )

    # Also flag unknown labels (independent of rel check)
    for label in (src, tgt):
        if label not in label_name_set and label not in alias_to_primary:
            suggestions = difflib.get_close_matches(label, label_name_set, n=2, cutoff=0.6)
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            errors.append(f"Unknown label '{label}' — not in schema.{hint}")


def _closest_rel(
    rel: str,
    valid_rels: set[str],
    topology: "GraphTopology",
) -> str:
    """Return a 'Did you mean X (N uses)?' hint string, or empty string."""
    matches = difflib.get_close_matches(rel, valid_rels, n=1, cutoff=0.6)
    if not matches:
        return ""
    best = matches[0]
    count = next(
        (t.count for t in topology.triples if t.rel_type == best), 0,
    )
    hint = f" Did you mean [:{best}]"
    hint += f" ({count} uses)?" if count else "?"
    return hint
