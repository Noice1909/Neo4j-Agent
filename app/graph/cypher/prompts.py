"""
Dynamic Cypher prompt builder (Strategy #3).

Replaces the static movie-domain few-shot examples with a topology-derived
prompt that works with any Neo4j database.  The topology section and few-shot
examples are generated at startup from the live schema; the PromptTemplate
itself is rebuilt whenever the topology is refreshed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.prompts import PromptTemplate

if TYPE_CHECKING:
    from app.graph.topology import GraphTopology, RelationshipTriple

# ── Universal Cypher rules (domain-agnostic) ──────────────────────────────────

UNIVERSAL_CYPHER_RULES = """\
- Output ONLY the Cypher statement — no explanations, no markdown fences.
- Always include a RETURN clause.
- RETURN full nodes (e.g. RETURN n) unless the question targets a specific property.
- Use single quotes for string literals: {name: 'Value'}.
- DIRECTION RULE: The topology shows (Source)-[:REL]->(Target). Always keep this arrow
  direction. When filtering by the Target node, place the filter ON the target but keep
  the arrow pointing the same way:
    CORRECT:  MATCH (a:Source)-[:REL]->(b:Target {name: 'X'}) RETURN a
    WRONG:    MATCH (b:Target {name: 'X'})-[:REL]->(a:Source)  <- reversed arrow
- For "related to" / "connected to" / "belongs to" queries with no clear direction,
  use undirected: MATCH (a)-[r]-(b).
- For "downstream" / "upstream" / "flows into" queries, follow the arrow from topology.
- For multi-hop queries, use the traversal paths shown in the topology section.
- For property-specific questions (e.g. "Disposition_Status of", "Code_Repository of"),
  use RETURN n.PropertyName — do NOT create a relationship to the property.
- Never invent labels or relationship types not listed in the topology below.
- Never refuse — always produce your best attempt.
- If the question contains pronouns like 'those', 'these', 'them', interpret them \
using the most relevant entities in the schema."""

# ── Topology section builder ──────────────────────────────────────────────────


def _render_chain(chain: "list[RelationshipTriple]") -> str:
    """Render a multi-hop chain as a single traversal path string."""
    if not chain:
        return ""
    parts = [f"({chain[0].source_label})"]
    for t in chain:
        parts.append(f"-[:{t.rel_type}]->({t.target_label})")
    return "".join(parts)


def build_topology_section(topology: "GraphTopology") -> str:
    """
    Render *topology* as an enriched block including:

    - Per-triple ``(Source)-[:REL]->(Target)`` with a filter-by-target example
    - Multi-hop chain paths (for "connected to" / indirect queries)
    - Per-label property hints (for property-specific RETURN queries)

    Returns an empty string when the topology has no triples.
    """
    if not topology.triples:
        return ""

    lines: list[str] = ["Relationship topology (use ONLY these types):"]
    for t in topology.triples:
        lines.append(f"  {t}")
        lines.append(
            f"    filter by {t.target_label}: "
            f"MATCH (a:{t.source_label})-[:{t.rel_type}]->"
            f"(b:{t.target_label} {{name:'X'}}) RETURN a"
        )

    # Multi-hop chain paths
    if topology.chains:
        lines.append("")
        lines.append("Multi-hop traversal paths (for 'connected to' / indirect queries):")
        for chain in topology.chains:
            rendered = _render_chain(chain)
            if rendered:
                lines.append(f"  {rendered}")

    # Per-label property hints
    prop_lines: list[str] = []
    for li in topology.labels:
        if len(li.properties) > 1:
            prop_list = ", ".join(li.properties[:8])  # cap at 8 for prompt brevity
            prop_lines.append(f"  {li.label}: {prop_list}")
    if prop_lines:
        lines.append("")
        lines.append(
            "Node properties (use n.PropertyName in RETURN for property-specific queries):"
        )
        lines.extend(prop_lines)

    return "\n".join(lines)


# ── Dynamic prompt builder ────────────────────────────────────────────────────


def build_cypher_prompt(topology: "GraphTopology", few_shot: str) -> PromptTemplate:
    """
    Build a ``PromptTemplate`` baked with *topology* and *few_shot* examples.

    ``GraphCypherQAChain`` only interpolates ``{schema}`` and ``{question}``,
    so the topology section and rules are embedded as literal text in the
    template string at build time.

    Parameters
    ----------
    topology:
        The live graph topology used to render the topology section.
    few_shot:
        Pre-rendered few-shot examples string (from ``generate_few_shot_examples``).
    """
    topology_section = build_topology_section(topology)

    parts: list[str] = [
        "Task: Generate a Cypher statement to query a graph database.",
        "Instructions: Use ONLY the provided relationship types and properties.",
        "",
        "Schema:\n{schema}",
    ]

    if topology_section:
        parts += ["", topology_section]

    parts += [
        "",
        "Rules:",
        UNIVERSAL_CYPHER_RULES,
    ]

    if few_shot:
        parts += ["", "Few-shot examples:", few_shot]

    parts += ["", "The question is:\n{question}"]

    template = "\n".join(parts)
    return PromptTemplate(input_variables=["schema", "question"], template=template)


# ── Fallback static prompt (used before topology is available) ─────────────────

_FALLBACK_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=(
        "Task: Generate a Cypher statement to query a graph database.\n"
        "Instructions: Use only the provided relationship types and properties.\n"
        "\n"
        "Schema:\n{schema}\n\n"
        "Rules:\n"
        + UNIVERSAL_CYPHER_RULES
        + "\n\nThe question is:\n{question}"
    ),
)
