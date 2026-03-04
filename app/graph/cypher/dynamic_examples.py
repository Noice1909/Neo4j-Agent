"""
Auto-generated Cypher few-shot examples derived from the live graph topology.

``generate_few_shot_examples()`` produces up to 8 canonical Cypher patterns
filled with real label names, relationship types, and sample property values
from ``GraphTopology``.  Nothing is hardcoded — regenerate by passing any
``GraphTopology`` instance.

The 8 pattern categories
─────────────────────────
1. List all nodes of a type
2. Forward traversal (A)-[:R]->(B) with property filter
3. Reverse traversal: find A given B
4. Multi-hop (A)-[:R1]->(B)-[:R2]->(C)
5. Generic undirected (a)-[]-(b) — "connected to" queries
6. Property-specific RETURN (not whole node)
7. Aggregation / count
8. Property filter with WHERE
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.graph.topology import GraphTopology, LabelInfo, RelationshipTriple

logger = logging.getLogger(__name__)


def _display(li: "LabelInfo") -> str:
    """Return `n.<display_property>` or `n` for a label's canonical return."""
    if li.display_property:
        return f"n.{li.display_property}"
    return "n"


def _sample(li: "LabelInfo") -> str:
    """Return a realistic sample value string for the display property, or 'Value'."""
    if li.display_property and li.display_property in li.sample_values:
        raw = li.sample_values[li.display_property]
        # Truncate very long values
        return raw[:40] if len(raw) <= 40 else raw[:37] + "..."
    return "Example"


def _second_prop(li: "LabelInfo") -> str | None:
    """Return a non-display property name, or None."""
    for p in li.properties:
        if p != li.display_property:
            return p
    return None


def generate_few_shot_examples(
    topology: "GraphTopology",
    max_examples: int = 8,
    manual_overrides: list[dict] | None = None,
) -> str:
    """
    Build a few-shot block from *topology*.

    Parameters
    ----------
    topology:
        The live graph topology to derive examples from.
    max_examples:
        Hard cap on the number of examples (default 8).
    manual_overrides:
        Optional list of dicts with ``question`` and ``cypher`` keys that are
        prepended verbatim and count against *max_examples*.

    Returns
    -------
    str
        A multi-line string with numbered Cypher examples ready for inclusion
        in a prompt template.
    """
    examples: list[tuple[str, str]] = []

    # ── Prepend manual overrides ───────────────────────────────────────────────
    if manual_overrides:
        for override in manual_overrides[:max_examples]:
            q = override.get("question", "")
            c = override.get("cypher", "")
            if q and c:
                examples.append((q, c))

    labels = topology.labels
    triples = topology.triples
    chains = topology.chains

    # Convenience lookups
    label_map = {li.label: li for li in labels}

    def _li(name: str) -> "LabelInfo | None":
        return label_map.get(name)

    # ── Pattern 1: List all nodes of a type ───────────────────────────────────
    if labels and len(examples) < max_examples:
        li = labels[0]
        examples.append((
            f"List all {li.label} nodes.",
            f"MATCH (n:{li.label}) RETURN n",
        ))

    # ── Pattern 2: Forward traversal with property filter ─────────────────────
    if triples and len(examples) < max_examples:
        t = triples[0]
        src = _li(t.source_label)
        tgt = _li(t.target_label)
        if src and tgt:
            sample_val = _sample(src)
            examples.append((
                f"Which {t.target_label} nodes are related to the "
                f"{t.source_label} named '{sample_val}'?",
                f"MATCH (a:{t.source_label} {{{src.display_property or 'name'}: "
                f"'{sample_val}'}})-[:{t.rel_type}]->(b:{t.target_label}) RETURN b",
            ))

    # ── Pattern 3: Reverse traversal ──────────────────────────────────────────
    if triples and len(examples) < max_examples:
        t = triples[0]
        src = _li(t.source_label)
        tgt = _li(t.target_label)
        if src and tgt:
            sample_val = _sample(tgt)
            examples.append((
                f"Which {t.source_label} nodes point to the "
                f"{t.target_label} named '{sample_val}'?",
                f"MATCH (a:{t.source_label})-[:{t.rel_type}]->(b:{t.target_label} "
                f"{{{tgt.display_property or 'name'}: '{sample_val}'}}) RETURN a",
            ))

    # ── Pattern 4: Multi-hop chain ─────────────────────────────────────────────
    if chains and len(examples) < max_examples:
        chain = chains[0][:2]  # take at most 2 hops
        t1 = chain[0]
        t2 = chain[1]
        src = _li(t1.source_label)
        tgt = _li(t2.target_label)
        if src and tgt:
            sample_val = _sample(src)
            examples.append((
                f"Find all {t2.target_label} reachable from the "
                f"{t1.source_label} named '{sample_val}' via "
                f"{t1.target_label}.",
                f"MATCH (a:{t1.source_label} {{{src.display_property or 'name'}: "
                f"'{sample_val}'}})-[:{t1.rel_type}]->(mid:{t1.target_label})"
                f"-[:{t2.rel_type}]->(c:{t2.target_label}) RETURN c",
            ))

    # ── Pattern 5: Generic undirected "connected to" ───────────────────────────
    if len(labels) >= 2 and len(examples) < max_examples:
        li_a = labels[0]
        li_b = labels[1]
        examples.append((
            f"What is directly connected to a specific {li_a.label}?",
            f"MATCH (a:{li_a.label})-[r]-(b) RETURN b, type(r) AS relationship",
        ))

    # ── Pattern 6: Property-specific RETURN ───────────────────────────────────
    if labels and len(examples) < max_examples:
        li = next(
            (l for l in labels if len(l.properties) > 1),
            labels[0],
        )
        second = _second_prop(li)
        if second:
            examples.append((
                f"Show the {second} of each {li.label}.",
                f"MATCH (n:{li.label}) RETURN n.{li.display_property or 'name'} "
                f"AS name, n.{second} AS {second}",
            ))

    # ── Pattern 7: Aggregation / count ────────────────────────────────────────
    if triples and len(examples) < max_examples:
        t = triples[0]
        tgt = _li(t.target_label)
        if tgt:
            dp = tgt.display_property or "name"
            examples.append((
                f"How many {t.source_label} nodes does each {t.target_label} have?",
                f"MATCH (a:{t.source_label})-[:{t.rel_type}]->(b:{t.target_label}) "
                f"RETURN b.{dp} AS {t.target_label.lower()}, "
                f"count(a) AS total_{t.source_label.lower()}s "
                f"ORDER BY total_{t.source_label.lower()}s DESC",
            ))

    # ── Pattern 8: Property filter with WHERE ─────────────────────────────────
    if labels and len(examples) < max_examples:
        li = next(
            (l for l in labels if l.display_property and l.sample_values),
            labels[0] if labels else None,
        )
        if li and li.display_property:
            sample_val = _sample(li)
            examples.append((
                f"Find {li.label} nodes where "
                f"{li.display_property} is '{sample_val}'.",
                f"MATCH (n:{li.label}) "
                f"WHERE n.{li.display_property} = '{sample_val}' RETURN n",
            ))

    if not examples:
        logger.warning("dynamic_examples: topology is empty — no examples generated.")
        return ""

    lines: list[str] = []
    for i, (question, cypher) in enumerate(examples[:max_examples], start=1):
        lines.append(f"Example {i}:")
        lines.append(f"  Question: {question}")
        lines.append(f"  Cypher:   {cypher}")
        lines.append("")

    result = "\n".join(lines).rstrip()
    logger.debug("dynamic_examples: generated %d examples.", len(examples[:max_examples]))
    return result
