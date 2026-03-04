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

_Example = tuple[str, str]


# ── Property helpers ───────────────────────────────────────────────────────────


def _sample(li: "LabelInfo") -> str:
    """Return a realistic sample value string for the display property, or 'Example'."""
    if li.display_property and li.display_property in li.sample_values:
        raw = li.sample_values[li.display_property]
        return raw[:40] if len(raw) <= 40 else raw[:37] + "..."
    return "Example"


def _second_prop(li: "LabelInfo") -> str | None:
    """Return a non-display property name, or None."""
    for p in li.properties:
        if p != li.display_property:
            return p
    return None


# ── Pattern builders ───────────────────────────────────────────────────────────


def _p1_list_all(li: "LabelInfo") -> _Example:
    return (
        f"List all {li.label} nodes.",
        f"MATCH (n:{li.label}) RETURN n",
    )


def _p2_forward(t: "RelationshipTriple", src: "LabelInfo") -> _Example:
    sample_val = _sample(src)
    return (
        f"Which {t.target_label} nodes are related to the "
        f"{t.source_label} named '{sample_val}'?",
        f"MATCH (a:{t.source_label} {{{src.display_property or 'name'}: "
        f"'{sample_val}'}})-[:{t.rel_type}]->(b:{t.target_label}) RETURN b",
    )


def _p3_reverse(t: "RelationshipTriple", tgt: "LabelInfo") -> _Example:
    sample_val = _sample(tgt)
    return (
        f"Which {t.source_label} nodes point to the "
        f"{t.target_label} named '{sample_val}'?",
        f"MATCH (a:{t.source_label})-[:{t.rel_type}]->(b:{t.target_label} "
        f"{{{tgt.display_property or 'name'}: '{sample_val}'}}) RETURN a",
    )


def _p4_multihop(
    t1: "RelationshipTriple", t2: "RelationshipTriple", src: "LabelInfo",
) -> _Example:
    sample_val = _sample(src)
    return (
        f"Find all {t2.target_label} reachable from the "
        f"{t1.source_label} named '{sample_val}' via {t1.target_label}.",
        f"MATCH (a:{t1.source_label} {{{src.display_property or 'name'}: "
        f"'{sample_val}'}})-[:{t1.rel_type}]->(mid:{t1.target_label})"
        f"-[:{t2.rel_type}]->(c:{t2.target_label}) RETURN c",
    )


def _p5_undirected(li_a: "LabelInfo") -> _Example:
    return (
        f"What is directly connected to a specific {li_a.label}?",
        f"MATCH (a:{li_a.label})-[r]-(b) RETURN b, type(r) AS relationship",
    )


def _p6_property_return(li: "LabelInfo") -> "_Example | None":
    second = _second_prop(li)
    if not second:
        return None
    return (
        f"Show the {second} of each {li.label}.",
        f"MATCH (n:{li.label}) RETURN n.{li.display_property or 'name'} "
        f"AS name, n.{second} AS {second}",
    )


def _p7_aggregation(t: "RelationshipTriple", tgt: "LabelInfo") -> _Example:
    dp = tgt.display_property or "name"
    return (
        f"How many {t.source_label} nodes does each {t.target_label} have?",
        f"MATCH (a:{t.source_label})-[:{t.rel_type}]->(b:{t.target_label}) "
        f"RETURN b.{dp} AS {t.target_label.lower()}, "
        f"count(a) AS total_{t.source_label.lower()}s "
        f"ORDER BY total_{t.source_label.lower()}s DESC",
    )


def _p8_where_filter(li: "LabelInfo") -> "_Example | None":
    if not li.display_property:
        return None
    sample_val = _sample(li)
    return (
        f"Find {li.label} nodes where {li.display_property} is '{sample_val}'.",
        f"MATCH (n:{li.label}) "
        f"WHERE n.{li.display_property} = '{sample_val}' RETURN n",
    )


def _p9_reverse_filter(t: "RelationshipTriple", tgt: "LabelInfo") -> _Example:
    """Filter by the target node — arrow direction stays the same as topology."""
    sample_val = _sample(tgt)
    return (
        f"Which {t.source_label} nodes are connected to the "
        f"{t.target_label} named '{sample_val}'?",
        f"MATCH (a:{t.source_label})-[:{t.rel_type}]->"
        f"(b:{t.target_label} {{{tgt.display_property or 'name'}: '{sample_val}'}}) RETURN a",
    )


def _p10_connected_via(
    t1: "RelationshipTriple", t2: "RelationshipTriple", src: "LabelInfo",
) -> _Example:
    """Multi-hop 'connected to' query using a two-hop chain from topology."""
    sample_val = _sample(src)
    return (
        f"Which {t2.target_label} nodes are connected to the "
        f"{t1.source_label} named '{sample_val}'?",
        f"MATCH (a:{t1.source_label} {{{src.display_property or 'name'}: '{sample_val}'}})"
        f"-[:{t1.rel_type}]->(mid:{t1.target_label})"
        f"-[:{t2.rel_type}]->(c:{t2.target_label}) RETURN c",
    )


# ── Collection helpers (break up complexity) ───────────────────────────────────


def _add_if_room(examples: "list[_Example]", pair: "_Example | None", max_n: int) -> None:
    """Append *pair* to *examples* only when within *max_n* capacity."""
    if pair and len(examples) < max_n:
        examples.append(pair)


def _collect_traversal(
    triples: "list[RelationshipTriple]",
    label_map: "dict[str, LabelInfo]",
    examples: "list[_Example]",
    max_n: int,
) -> None:
    """Patterns 2 & 3: forward and reverse traversal."""
    if not triples:
        return
    t = triples[0]
    src = label_map.get(t.source_label)
    tgt = label_map.get(t.target_label)
    if src and tgt:
        _add_if_room(examples, _p2_forward(t, src), max_n)
        _add_if_room(examples, _p3_reverse(t, tgt), max_n)


def _collect_chain(
    chains: "list[list[RelationshipTriple]]",
    label_map: "dict[str, LabelInfo]",
    examples: "list[_Example]",
    max_n: int,
) -> None:
    """Pattern 4: multi-hop chain."""
    if not chains or len(chains[0]) < 2:
        return
    t1, t2 = chains[0][0], chains[0][1]
    src = label_map.get(t1.source_label)
    if src:
        _add_if_room(examples, _p4_multihop(t1, t2, src), max_n)


def _collect_remaining(
    labels: "list[LabelInfo]",
    triples: "list[RelationshipTriple]",
    chains: "list[list[RelationshipTriple]]",
    label_map: "dict[str, LabelInfo]",
    examples: "list[_Example]",
    max_n: int,
) -> None:
    """Patterns 5–10: undirected, property-specific, aggregation, WHERE filter,
    reverse-filter direction demo, multi-hop connected."""
    if len(labels) >= 2:
        _add_if_room(examples, _p5_undirected(labels[0]), max_n)

    if labels:
        li = next((l for l in labels if len(l.properties) > 1), labels[0])
        _add_if_room(examples, _p6_property_return(li), max_n)

    if triples:
        t = triples[0]
        tgt = label_map.get(t.target_label)
        if tgt:
            _add_if_room(examples, _p7_aggregation(t, tgt), max_n)

    if labels:
        li = next(
            (l for l in labels if l.display_property and l.sample_values),
            labels[0],
        )
        _add_if_room(examples, _p8_where_filter(li), max_n)

    # Pattern 9 — reverse-filter: keep arrow direction, filter by target
    if triples:
        t = triples[0]
        tgt = label_map.get(t.target_label)
        if tgt:
            _add_if_room(examples, _p9_reverse_filter(t, tgt), max_n)

    # Pattern 10 — multi-hop "connected to" via chain
    if chains and len(chains[0]) >= 2:
        t1, t2 = chains[0][0], chains[0][1]
        src = label_map.get(t1.source_label)
        if src:
            _add_if_room(examples, _p10_connected_via(t1, t2, src), max_n)


# ── Public API ─────────────────────────────────────────────────────────────────


def generate_few_shot_examples(
    topology: "GraphTopology",
    max_examples: int = 10,
    manual_overrides: list[dict] | None = None,
) -> str:
    """
    Build a few-shot block from *topology*.

    Parameters
    ----------
    topology:
        The live graph topology to derive examples from.
    max_examples:
        Hard cap on the number of examples (default 10).
    manual_overrides:
        Optional list of dicts with ``question`` and ``cypher`` keys that are
        prepended verbatim and count against *max_examples*.

    Returns
    -------
    str
        A multi-line string with numbered Cypher examples ready for inclusion
        in a prompt template.
    """
    examples: list[_Example] = []

    if manual_overrides:
        for override in manual_overrides[:max_examples]:
            q = override.get("question", "")
            c = override.get("cypher", "")
            if q and c:
                examples.append((q, c))

    labels = topology.labels
    triples = topology.triples
    label_map: dict[str, LabelInfo] = {li.label: li for li in labels}  # type: ignore[type-arg]

    if labels:
        _add_if_room(examples, _p1_list_all(labels[0]), max_examples)

    _collect_traversal(triples, label_map, examples, max_examples)
    _collect_chain(topology.chains, label_map, examples, max_examples)
    _collect_remaining(labels, triples, topology.chains, label_map, examples, max_examples)

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
