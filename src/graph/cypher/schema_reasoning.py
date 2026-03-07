"""
Schema reasoning — ambiguity detection and LLM-based resolution.

Detects when a user NL term could map to multiple schema elements
(e.g., "genre" could be a property on Movie OR a separate Genre node)
and uses a lightweight LLM call to resolve the ambiguity.

When no ambiguity exists (the common case), no LLM call is made.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from src.graph.topology import GraphTopology
    from src.graph.semantic_layer import SchemaSemanticLayer

logger = logging.getLogger(__name__)


@dataclass
class SchemaCandidate:
    """One possible schema mapping for an ambiguous term."""

    kind: str           # "property" | "node_label" | "relationship"
    label: str          # "Movie" or "Genre"
    element_name: str   # "movie_type" or "Genre" or "HAS_GENRE"
    confidence: float = 0.8
    context: str = ""   # extra info (e.g., sample values)


@dataclass
class SchemaAmbiguity:
    """A detected ambiguity in how a user term maps to schema."""

    user_term: str
    candidates: list[SchemaCandidate] = field(default_factory=list)


@dataclass
class ResolvedMapping:
    """A resolved NL term → schema element mapping."""

    user_term: str
    kind: str           # "property" | "node_label" | "relationship"
    label: str
    element_name: str
    description: str = ""


def detect_ambiguities(
    question: str,
    topology: "GraphTopology",
    label_synonym_map: dict[str, str],
    property_synonym_map: dict[str, tuple[str, str]],
    relationship_synonym_map: dict[str, str] | None = None,
) -> list[SchemaAmbiguity]:
    """
    Detect terms in the question that map to multiple schema elements.

    Only returns ambiguities where a term matches BOTH a label AND a property
    (or a relationship). Single-match terms are not ambiguous.
    """
    ambiguities: list[SchemaAmbiguity] = []
    seen_terms: set[str] = set()

    for word in question.lower().split():
        candidate = word.strip("?.,!'\"-")
        if len(candidate) < 3 or candidate in seen_terms:
            continue
        seen_terms.add(candidate)

        candidates: list[SchemaCandidate] = []

        # Check label match
        if candidate in label_synonym_map:
            matched_label = label_synonym_map[candidate]
            candidates.append(SchemaCandidate(
                kind="node_label",
                label=matched_label,
                element_name=matched_label,
                context=f"Node label: {matched_label}",
            ))

        # Check property match
        if candidate in property_synonym_map:
            label, prop_name = property_synonym_map[candidate]
            # Get sample value for context
            sample = ""
            for li in topology.labels:
                if li.label == label:
                    sample = li.sample_values.get(prop_name, "")
                    break
            candidates.append(SchemaCandidate(
                kind="property",
                label=label,
                element_name=prop_name,
                context=f"Property {label}.{prop_name}" + (f" (e.g., '{sample}')" if sample else ""),
            ))

        # Check relationship match
        if relationship_synonym_map and candidate in relationship_synonym_map:
            rel_type = relationship_synonym_map[candidate]
            for t in topology.triples:
                if t.rel_type == rel_type:
                    candidates.append(SchemaCandidate(
                        kind="relationship",
                        label=t.source_label,
                        element_name=rel_type,
                        context=f"({t.source_label})-[:{rel_type}]->({t.target_label})",
                    ))
                    break

        # Only flag as ambiguous if there are 2+ different kinds of candidates
        if len(candidates) >= 2:
            kinds = set(c.kind for c in candidates)
            if len(kinds) >= 2:
                ambiguities.append(SchemaAmbiguity(
                    user_term=candidate,
                    candidates=candidates,
                ))

    return ambiguities


async def resolve_ambiguity(
    question: str,
    ambiguities: list[SchemaAmbiguity],
    topology: "GraphTopology",
    llm: "BaseChatModel",
) -> list[ResolvedMapping]:
    """
    Use LLM to resolve schema ambiguities.

    Makes ONE lightweight LLM call with all ambiguities batched together.
    """
    if not ambiguities:
        return []

    prompt_parts = [
        "Given this user question and database schema, resolve each ambiguous term.",
        f"\nQuestion: \"{question}\"",
    ]

    for i, amb in enumerate(ambiguities):
        prompt_parts.append(f"\nAmbiguous term #{i+1}: \"{amb.user_term}\"")
        prompt_parts.append("Possible interpretations:")
        for j, c in enumerate(amb.candidates):
            letter = chr(65 + j)  # A, B, C, ...
            prompt_parts.append(f"  {letter}) {c.kind}: {c.context}")

    prompt_parts.append(
        "\nFor each term, reply with ONLY the term and the letter of the correct "
        "interpretation, one per line. Format: term=LETTER"
    )

    prompt = "\n".join(prompt_parts)

    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, lambda: llm.invoke(prompt),
        )
        raw = str(response.content).strip()
    except Exception as exc:
        logger.warning("Schema ambiguity resolution LLM call failed: %s", exc)
        # Default: pick the first candidate (usually the node label)
        return [
            ResolvedMapping(
                user_term=amb.user_term,
                kind=amb.candidates[0].kind,
                label=amb.candidates[0].label,
                element_name=amb.candidates[0].element_name,
            )
            for amb in ambiguities
        ]

    # Parse response
    resolved: list[ResolvedMapping] = []
    for amb in ambiguities:
        # Try to find the resolution in the LLM response
        chosen = amb.candidates[0]  # default
        for line in raw.split("\n"):
            line = line.strip().lower()
            if amb.user_term.lower() in line:
                # Extract letter
                for j, c in enumerate(amb.candidates):
                    letter = chr(65 + j).lower()
                    if f"={letter}" in line or f": {letter}" in line or line.endswith(letter):
                        chosen = c
                        break
                break

        resolved.append(ResolvedMapping(
            user_term=amb.user_term,
            kind=chosen.kind,
            label=chosen.label,
            element_name=chosen.element_name,
        ))

    return resolved


def build_schema_context(
    property_mappings: list[dict],
    resolved_ambiguities: list[ResolvedMapping] | None = None,
    topology: "GraphTopology | None" = None,
) -> str:
    """
    Build a schema_context string for the Cypher generation prompt.

    Contains explicit NL-to-schema mappings that guide the LLM.
    """
    lines = []

    if property_mappings or resolved_ambiguities:
        lines.append("Schema mapping hints for this question:")

    # Property mappings (from entity resolution Layer 1.5)
    for pm in property_mappings:
        nl_term = pm.get("nl_term", "")
        label = pm.get("label", "")
        prop = pm.get("property", "")
        if nl_term and label and prop:
            # Get sample value for extra context
            sample_info = ""
            if topology:
                for li in topology.labels:
                    if li.label == label:
                        sample = li.sample_values.get(prop, "")
                        if sample:
                            sample_info = f" (e.g., '{sample}')"
                        break
            lines.append(f"- \"{nl_term}\" -> {label}.{prop}{sample_info}")

    # Resolved ambiguities
    if resolved_ambiguities:
        for rm in resolved_ambiguities:
            if rm.kind == "property":
                lines.append(f"- \"{rm.user_term}\" -> {rm.label}.{rm.element_name} (property)")
            elif rm.kind == "node_label":
                lines.append(f"- \"{rm.user_term}\" -> :{rm.label} (node label)")
            elif rm.kind == "relationship":
                lines.append(f"- \"{rm.user_term}\" -> [:{rm.element_name}] (relationship)")

    return "\n".join(lines) if lines else ""
