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
    from src.graph.topology import GraphTopology

# ── Universal Cypher rules (domain-agnostic) ──────────────────────────────────

UNIVERSAL_CYPHER_RULES = """\
- Output ONLY the Cypher statement — no explanations, no markdown fences.
- Always include a RETURN clause.
- RETURN full nodes (e.g. RETURN n) unless the question targets a specific property.
- Use single quotes for string literals: {name: 'Value'}.
- For "related to" / "connected to" queries, use undirected: MATCH (a)-[r]-(b).
- For "downstream" / "depends on" queries, follow the arrow direction from the topology.
- Never invent labels or relationship types not listed in the topology below.
- Never refuse — always produce your best attempt.
- If the question contains pronouns like 'those', 'these', 'them', interpret them \
using the most relevant entities in the schema."""

# ── Topology section builder ──────────────────────────────────────────────────


def build_topology_section(topology: "GraphTopology") -> str:
    """
    Render *topology* as a compact ``(A)-[:REL]->(B)`` block.

    Returns an empty string when the topology has no triples so the caller
    can decide whether to include the section at all.
    """
    if not topology.triples:
        return ""
    lines = ["Relationship topology (use ONLY these types):"]
    for t in topology.triples:
        lines.append(f"  {t}")
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
