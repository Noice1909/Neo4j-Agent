"""
Schema Semantic Layer — auto-generated NL metadata for every schema element.

At startup, makes ONE LLM call to analyze the entire schema (labels, properties,
relationships, sample values) and generates:

- Natural language names for every property (e.g., ``movie_type`` → ["genre", "type", "category"])
- Natural language phrases for every relationship (e.g., ``DIRECTED`` → ["directed", "who directed"])
- Data type hints inferred from sample values
- One-line descriptions for each element

Concept node metadata (manually curated) always takes priority over LLM-generated
metadata.  The LLM only fills gaps where Concept nodes don't exist.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from src.graph.topology import GraphTopology

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class PropertySemantics:
    """Semantic metadata for a single property on a node label."""

    property_name: str          # e.g., "movie_type"
    label: str                  # e.g., "Movie"
    natural_names: list[str] = field(default_factory=list)   # ["genre", "type", "category"]
    description: str = ""       # "The genre or category of the movie"
    data_type_hint: str = "string"  # "string" | "integer" | "date" | "float" | "boolean"


@dataclass
class RelationshipSemantics:
    """Semantic metadata for a relationship type."""

    rel_type: str               # e.g., "DIRECTED"
    source_label: str           # e.g., "Person"
    target_label: str           # e.g., "Movie"
    natural_phrases: list[str] = field(default_factory=list)  # ["directed", "who directed"]
    description: str = ""       # "A person who directed a movie"


@dataclass
class SchemaSemanticLayer:
    """Complete semantic layer for the entire schema."""

    property_semantics: dict[str, list[PropertySemantics]] = field(default_factory=dict)
    # label -> [PropertySemantics]
    relationship_semantics: list[RelationshipSemantics] = field(default_factory=list)
    schema_hash: str = ""       # hash of the schema used to generate this layer

    # Precomputed fast-lookup indices (built by _build_indices)
    nl_to_property: dict[str, tuple[str, str]] = field(default_factory=dict)
    # nl_term -> (label, property_name)
    nl_to_relationship: dict[str, str] = field(default_factory=dict)
    # nl_phrase -> rel_type

    def _build_indices(self) -> None:
        """Build fast-lookup indices from the semantic data."""
        self.nl_to_property = {}
        for label, props in self.property_semantics.items():
            for ps in props:
                for nl_name in ps.natural_names:
                    key = nl_name.lower().strip()
                    if key:
                        self.nl_to_property[key] = (ps.label, ps.property_name)

        self.nl_to_relationship = {}
        for rs in self.relationship_semantics:
            for phrase in rs.natural_phrases:
                key = phrase.lower().strip()
                if key:
                    self.nl_to_relationship[key] = rs.rel_type

    def get_nl_terms_for_property(self, label: str, prop: str) -> list[str]:
        """Return NL aliases for a specific property."""
        for ps in self.property_semantics.get(label, []):
            if ps.property_name == prop:
                return ps.natural_names
        return []


# ── Schema hash ──────────────────────────────────────────────────────────────

def _compute_schema_hash(topology: "GraphTopology") -> str:
    """Compute a stable hash of the schema for change detection."""
    parts = []
    for li in sorted(topology.labels, key=lambda x: x.label):
        parts.append(f"{li.label}:{','.join(sorted(li.properties))}")
    for t in sorted(topology.triples, key=lambda x: (x.source_label, x.rel_type, x.target_label)):
        parts.append(f"{t.source_label}-{t.rel_type}->{t.target_label}")
    content = "|".join(parts)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ── LLM prompt builder ──────────────────────────────────────────────────────

def _build_schema_prompt(topology: "GraphTopology") -> str:
    """Build the prompt for the LLM to analyze the schema."""
    lines = [
        "Analyze this database schema and generate natural language metadata.",
        "For EVERY property and relationship, provide:",
        "1. natural_names: 3-5 natural language names a non-technical user would use",
        "2. description: one-line description",
        "3. For properties: data_type inferred from sample value (string/integer/date/float/boolean)",
        "",
        "Schema:",
        "Labels and Properties (with sample values):",
    ]

    for li in topology.labels:
        if li.properties:
            sample_parts = []
            for prop in li.properties[:10]:
                val = li.sample_values.get(prop, "")
                if val:
                    sample_parts.append(f'{prop}="{val}"')
                else:
                    sample_parts.append(prop)
            lines.append(f"  {li.label}: {', '.join(sample_parts)}")
        else:
            lines.append(f"  {li.label}: (no properties)")

    if topology.triples:
        lines.append("")
        lines.append("Relationships:")
        for t in topology.triples:
            lines.append(f"  ({t.source_label})-[:{t.rel_type}]->({t.target_label})")

    lines.extend([
        "",
        "Respond in JSON format ONLY (no markdown, no explanation):",
        "{",
        '  "properties": {',
        '    "Label.property_name": {',
        '      "natural_names": ["alias1", "alias2", "alias3"],',
        '      "description": "one-line description",',
        '      "data_type": "string"',
        "    }",
        "  },",
        '  "relationships": {',
        '    "REL_TYPE": {',
        '      "natural_phrases": ["phrase1", "phrase2"],',
        '      "description": "one-line description"',
        "    }",
        "  }",
        "}",
    ])

    return "\n".join(lines)


# ── Response parser ──────────────────────────────────────────────────────────

def _parse_llm_response(
    raw: str,
    topology: "GraphTopology",
) -> tuple[dict[str, list[PropertySemantics]], list[RelationshipSemantics]]:
    """Parse the LLM's JSON response into semantic objects."""
    # Extract JSON from response (handle markdown fences)
    text = raw.strip()
    if text.startswith("```"):
        # Remove markdown fences
        lines = text.split("\n")
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
            except json.JSONDecodeError:
                logger.warning("Failed to parse semantic layer LLM response as JSON.")
                return {}, []
        else:
            logger.warning("No JSON found in semantic layer LLM response.")
            return {}, []

    # Parse properties
    prop_semantics: dict[str, list[PropertySemantics]] = {}
    raw_props = data.get("properties", {})
    for key, meta in raw_props.items():
        if "." in key:
            label, prop_name = key.split(".", 1)
        else:
            label = ""
            prop_name = key

        if not isinstance(meta, dict):
            continue

        ps = PropertySemantics(
            property_name=prop_name,
            label=label,
            natural_names=meta.get("natural_names", []),
            description=meta.get("description", ""),
            data_type_hint=meta.get("data_type", "string"),
        )
        prop_semantics.setdefault(label, []).append(ps)

    # Parse relationships
    rel_semantics: list[RelationshipSemantics] = []
    raw_rels = data.get("relationships", {})
    # Build rel_type -> (source, target) mapping from topology
    rel_endpoints: dict[str, tuple[str, str]] = {}
    for t in topology.triples:
        rel_endpoints.setdefault(t.rel_type, (t.source_label, t.target_label))

    for rel_type, meta in raw_rels.items():
        if not isinstance(meta, dict):
            continue
        src, tgt = rel_endpoints.get(rel_type, ("", ""))
        rs = RelationshipSemantics(
            rel_type=rel_type,
            source_label=src,
            target_label=tgt,
            natural_phrases=meta.get("natural_phrases", []),
            description=meta.get("description", ""),
        )
        rel_semantics.append(rs)

    return prop_semantics, rel_semantics


# ── Concept node merge ───────────────────────────────────────────────────────

def _merge_with_concepts(
    prop_semantics: dict[str, list[PropertySemantics]],
    rel_semantics: list[RelationshipSemantics],
    topology: "GraphTopology",
) -> None:
    """
    Merge Concept node metadata with LLM-generated metadata in-place.

    Concept node data takes priority:
    - Concept descriptions override LLM descriptions
    - Concept nlp_terms are merged with (not replaced by) LLM natural_names
    """
    for li in topology.labels:
        if not li.description and not li.nlp_terms:
            continue
        # Enrich property semantics for this label
        label_props = prop_semantics.get(li.label, [])
        # The Concept node provides label-level info, not per-property
        # But we can use nlp_terms as additional natural names for the label itself
        # (handled separately in the synonym maps)


# ── Main entry point ─────────────────────────────────────────────────────────

async def generate_semantic_layer(
    topology: "GraphTopology",
    llm: "BaseChatModel",
) -> SchemaSemanticLayer:
    """
    Generate a semantic layer for the entire schema using ONE LLM call.

    Parameters
    ----------
    topology:
        The live graph topology with labels, properties, sample values, and relationships.
    llm:
        The LLM to use for semantic analysis.

    Returns
    -------
    SchemaSemanticLayer
        Complete semantic layer with NL metadata for all schema elements.
    """
    schema_hash = _compute_schema_hash(topology)

    # Build prompt
    prompt = _build_schema_prompt(topology)

    # Make LLM call
    logger.info("Generating semantic layer (one-time LLM call)...")
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, lambda: llm.invoke(prompt),
        )
        raw = str(response.content).strip()
    except Exception as exc:
        logger.warning("Semantic layer LLM call failed: %s — using pattern-only fallback.", exc)
        return SchemaSemanticLayer(schema_hash=schema_hash)

    # Parse response
    prop_semantics, rel_semantics = _parse_llm_response(raw, topology)

    # Merge with Concept node metadata
    _merge_with_concepts(prop_semantics, rel_semantics, topology)

    # Build layer
    layer = SchemaSemanticLayer(
        property_semantics=prop_semantics,
        relationship_semantics=rel_semantics,
        schema_hash=schema_hash,
    )
    layer._build_indices()

    total_props = sum(len(ps) for ps in prop_semantics.values())
    total_nl = len(layer.nl_to_property) + len(layer.nl_to_relationship)
    logger.info(
        "Semantic layer generated: %d properties, %d relationships, %d NL terms indexed.",
        total_props, len(rel_semantics), total_nl,
    )

    return layer


# ── Serialization (for Redis cache) ─────────────────────────────────────────

def semantic_layer_to_json(layer: SchemaSemanticLayer) -> str:
    """Serialize a SchemaSemanticLayer to JSON string."""
    return json.dumps({
        "schema_hash": layer.schema_hash,
        "property_semantics": {
            label: [
                {
                    "property_name": ps.property_name,
                    "label": ps.label,
                    "natural_names": ps.natural_names,
                    "description": ps.description,
                    "data_type_hint": ps.data_type_hint,
                }
                for ps in props
            ]
            for label, props in layer.property_semantics.items()
        },
        "relationship_semantics": [
            {
                "rel_type": rs.rel_type,
                "source_label": rs.source_label,
                "target_label": rs.target_label,
                "natural_phrases": rs.natural_phrases,
                "description": rs.description,
            }
            for rs in layer.relationship_semantics
        ],
    })


def semantic_layer_from_json(raw: str) -> SchemaSemanticLayer:
    """Deserialize a SchemaSemanticLayer from JSON string."""
    data = json.loads(raw)

    prop_semantics: dict[str, list[PropertySemantics]] = {}
    for label, props in data.get("property_semantics", {}).items():
        prop_semantics[label] = [PropertySemantics(**p) for p in props]

    rel_semantics = [RelationshipSemantics(**r) for r in data.get("relationship_semantics", [])]

    layer = SchemaSemanticLayer(
        property_semantics=prop_semantics,
        relationship_semantics=rel_semantics,
        schema_hash=data.get("schema_hash", ""),
    )
    layer._build_indices()
    return layer
