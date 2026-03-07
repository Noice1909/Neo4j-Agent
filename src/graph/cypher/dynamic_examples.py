"""
Auto-generated Cypher few-shot examples derived from the live graph topology.

``generate_few_shot_examples()`` produces up to 40 comprehensive Cypher patterns
filled with real label names, relationship types, and sample property values
from ``GraphTopology``.  Nothing is hardcoded — regenerate by passing any
``GraphTopology`` instance.

Comprehensive pattern coverage (40 patterns)
────────────────────────────────────────────
1. List all nodes of a type
2. Forward traversal (A)-[:R]->(B) with property filter
3. Reverse traversal: find A given B
4. Multi-hop (A)-[:R1]->(B)-[:R2]->(C)
5. Generic undirected (a)-[]-(b) — "connected to" queries
6. Property-specific RETURN (not whole node)
7. Aggregation / count with ORDER BY and LIMIT
8. Property filter with WHERE
9. Reverse-filter direction demonstration
10. Multi-hop "connected to" via chain

Advanced Cypher Keywords & Syntax
──────────────────────────────────
11. DISTINCT — remove duplicates
12. UNION — combine multiple query results
13. WITH — intermediate processing and pipelines
14. CASE — conditional logic
15. COLLECT — aggregate into lists
16. UNWIND — expand lists to rows
17. OPTIONAL MATCH — LEFT JOIN equivalent
18. EXISTS — subquery pattern existence checks
19. NOT — negation logic
20. IN — list membership checks
21. String operations — CONTAINS, STARTS WITH, ENDS WITH
22. Mathematical aggregations — sum, avg, min, max
23. List comprehension — [x IN list | expression]
24. Variable length paths — [:REL*min..max]
25. Complex WHERE — AND, OR, NOT combinations
26. APOC path expansion — apoc.path.expand()
27. Date/time operations — datetime(), date(), duration()
28. Shortest path — shortestPath()
29. Multiple MATCH clauses — separate pattern matching
30. Range queries — numeric/date comparisons
31. ORDER BY multiple fields — complex sorting
32. SKIP and LIMIT — pagination
33. Pattern comprehension — [(pattern) | expression]
34. Map projection — return custom maps
35. String functions — toLower, toUpper, trim, substring
36. Relationship properties — access rel attributes
37. APOC aggregation — apoc.agg.* functions
38. CALL subquery — isolated subquery execution
39. Multiple labels — :Label1:Label2 syntax
40. Null checks — IS NULL, IS NOT NULL
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.graph.topology import GraphTopology, LabelInfo, RelationshipTriple

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


def _p7_aggregation(t: "RelationshipTriple", src: "LabelInfo") -> _Example:
    """Aggregation with GROUP BY, COUNT, ORDER BY, and LIMIT for ranking queries."""
    dp_src = src.display_property or "name"
    return (
        f"List top {t.source_label} by number of {t.target_label} connections (ranked by count).",
        f"MATCH (a:{t.source_label})-[:{t.rel_type}]->(b:{t.target_label}) "
        f"RETURN a.{dp_src} AS {t.source_label.lower()}, "
        f"count(b) AS total_{t.target_label.lower()}s "
        f"ORDER BY total_{t.target_label.lower()}s DESC "
        f"LIMIT 10",
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


def _p11_distinct(li: "LabelInfo") -> "._Example | None":
    """DISTINCT keyword — remove duplicate values from property results."""
    if not li.display_property:
        return None
    second = _second_prop(li)
    if not second:
        return None
    return (
        f"What are all the unique {second} values for {li.label}?",
        f"MATCH (n:{li.label}) RETURN DISTINCT n.{second} AS {second}",
    )


def _p12_union(t: "RelationshipTriple", src: "LabelInfo", tgt: "LabelInfo") -> _Example:
    """UNION keyword — combine results from multiple queries."""
    return (
        f"Find all {t.source_label} or {t.target_label} nodes (combined results).",
        f"MATCH (a:{t.source_label}) RETURN a.{src.display_property or 'name'} AS name "
        f"UNION "
        f"MATCH (b:{t.target_label}) RETURN b.{tgt.display_property or 'name'} AS name",
    )


def _p13_with_clause(t: "RelationshipTriple", src: "LabelInfo") -> _Example:
    """WITH clause — intermediate processing and pipeline queries."""
    dp = src.display_property or "name"
    return (
        f"Find {t.source_label} with more than 2 {t.target_label} connections.",
        f"MATCH (a:{t.source_label})-[:{t.rel_type}]->(b:{t.target_label}) "
        f"WITH a, count(b) AS connection_count "
        f"WHERE connection_count > 2 "
        f"RETURN a.{dp} AS {t.source_label.lower()}, connection_count",
    )


def _p14_case_conditional(li: "LabelInfo") -> "_Example | None":
    """CASE keyword — conditional logic in RETURN clause."""
    second = _second_prop(li)
    if not second:
        return None
    return (
        f"Categorize {li.label} based on {second} value.",
        f"MATCH (n:{li.label}) "
        f"RETURN n.{li.display_property or 'name'} AS name, "
        f"CASE WHEN n.{second} IS NULL THEN 'unknown' "
        f"ELSE 'known' END AS category",
    )


def _p15_collect_aggregation(t: "RelationshipTriple", src: "LabelInfo", tgt: "LabelInfo") -> _Example:
    """COLLECT function — aggregate related nodes into lists."""
    dp_src = src.display_property or "name"
    dp_tgt = tgt.display_property or "name"
    return (
        f"For each {t.source_label}, list all related {t.target_label} names.",
        f"MATCH (a:{t.source_label})-[:{t.rel_type}]->(b:{t.target_label}) "
        f"RETURN a.{dp_src} AS {t.source_label.lower()}, "
        f"collect(b.{dp_tgt}) AS {t.target_label.lower()}_list",
    )


def _p16_unwind(li: "LabelInfo") -> _Example:
    """UNWIND keyword — expand lists to rows."""
    return (
        f"Expand a list of values and match {li.label} nodes.",
        f"UNWIND ['value1', 'value2', 'value3'] AS search_term "
        f"MATCH (n:{li.label}) "
        f"WHERE n.{li.display_property or 'name'} = search_term "
        f"RETURN n",
    )


def _p17_optional_match(t: "RelationshipTriple", src: "LabelInfo") -> _Example:
    """OPTIONAL MATCH — LEFT JOIN equivalent, returns nulls for missing relationships."""
    dp = src.display_property or "name"
    return (
        f"Find all {t.source_label} and their {t.target_label} connections (including those with none).",
        f"MATCH (a:{t.source_label}) "
        f"OPTIONAL MATCH (a)-[:{t.rel_type}]->(b:{t.target_label}) "
        f"RETURN a.{dp} AS {t.source_label.lower()}, "
        f"collect(b.{dp}) AS {t.target_label.lower()}s",
    )


def _p18_exists_subquery(t: "RelationshipTriple", src: "LabelInfo") -> _Example:
    """EXISTS keyword — check if a pattern exists."""
    dp = src.display_property or "name"
    return (
        f"Find {t.source_label} that have at least one {t.target_label} connection.",
        f"MATCH (a:{t.source_label}) "
        f"WHERE EXISTS {{ MATCH (a)-[:{t.rel_type}]->(:  {t.target_label}) }} "
        f"RETURN a.{dp} AS {t.source_label.lower()}",
    )


def _p19_not_negation(t: "RelationshipTriple", src: "LabelInfo") -> _Example:
    """NOT keyword — negation logic."""
    dp = src.display_property or "name"
    return (
        f"Find {t.source_label} that do NOT have {t.target_label} connections.",
        f"MATCH (a:{t.source_label}) "
        f"WHERE NOT EXISTS {{ MATCH (a)-[:{t.rel_type}]->(:  {t.target_label}) }} "
        f"RETURN a.{dp} AS {t.source_label.lower()}",
    )


def _p20_in_list(li: "LabelInfo") -> "_Example | None":
    """IN keyword — list membership checks."""
    if not li.display_property:
        return None
    return (
        f"Find {li.label} where {li.display_property} is in a specific list.",
        f"MATCH (n:{li.label}) "
        f"WHERE n.{li.display_property} IN ['value1', 'value2', 'value3'] "
        f"RETURN n",
    )


def _p21_string_operations(li: "LabelInfo") -> "_Example | None":
    """String operations — CONTAINS, STARTS WITH, ENDS WITH."""
    if not li.display_property:
        return None
    return (
        f"Find {li.label} where {li.display_property} contains a substring.",
        f"MATCH (n:{li.label}) "
        f"WHERE n.{li.display_property} CONTAINS 'search' "
        f"OR n.{li.display_property} STARTS WITH 'prefix' "
        f"OR n.{li.display_property} ENDS WITH 'suffix' "
        f"RETURN n",
    )


def _p22_math_aggregations(t: "RelationshipTriple") -> "_Example | None":
    """Mathematical aggregations — sum, avg, min, max on numeric properties."""
    # Find a numeric property if available
    return (
        f"Calculate statistics on {t.target_label} connections.",
        f"MATCH (a:{t.source_label})-[:{t.rel_type}]->(b:{t.target_label}) "
        f"RETURN count(b) AS total, "
        f"min(id(b)) AS min_id, "
        f"max(id(b)) AS max_id, "
        f"avg(id(b)) AS avg_id",
    )


def _p23_list_comprehension(t: "RelationshipTriple", tgt: "LabelInfo") -> _Example:
    """List comprehension — [x IN list | expression]."""
    dp_tgt = tgt.display_property or "name"
    return (
        f"Transform list of {t.target_label} names to uppercase.",
        f"MATCH (a:{t.source_label})-[:{t.rel_type}]->(b:{t.target_label}) "
        f"RETURN a, [x IN collect(b.{dp_tgt}) | toUpper(x)] AS names_upper",
    )


def _p24_variable_length_path(t: "RelationshipTriple", src: "LabelInfo") -> _Example:
    """Variable length paths — [:REL*min..max] for graph traversal."""
    dp = src.display_property or "name"
    sample_val = _sample(src)
    return (
        f"Find all {t.target_label} reachable within 1-3 hops from {t.source_label} '{sample_val}'.",
        f"MATCH (a:{t.source_label} {{{dp}: '{sample_val}'}})-[:{t.rel_type}*1..3]->(b:{t.target_label}) "
        f"RETURN DISTINCT b",
    )


def _p25_complex_where(t: "RelationshipTriple", src: "LabelInfo", tgt: "LabelInfo") -> _Example:
    """Complex WHERE clause — AND, OR, NOT combinations."""
    dp_src = src.display_property or "name"
    dp_tgt = tgt.display_property or "name"
    return (
        f"Find {t.source_label} and {t.target_label} with complex filter conditions.",
        f"MATCH (a:{t.source_label})-[:{t.rel_type}]->(b:{t.target_label}) "
        f"WHERE (a.{dp_src} IS NOT NULL AND b.{dp_tgt} IS NOT NULL) "
        f"OR (a.{dp_src} =~ '.*pattern.*') "
        f"RETURN a, b",
    )


def _p26_apoc_path_expand(t: "RelationshipTriple", src: "LabelInfo") -> _Example:
    """APOC path expansion — apoc.path.expand for flexible graph traversal."""
    dp = src.display_property or "name"
    sample_val = _sample(src)
    return (
        f"Use APOC to expand from {t.source_label} '{sample_val}' following relationships.",
        f"MATCH (a:{t.source_label} {{{dp}: '{sample_val}'}}) "
        f"CALL apoc.path.expand(a, '{t.rel_type}>', '+{t.target_label}', 1, 3) YIELD path "
        f"RETURN path",
    )


def _p27_datetime_operations(li: "LabelInfo") -> _Example:
    """Date/time operations — datetime(), date(), duration()."""
    dp = li.display_property or "name"
    return (
        f"Work with date/time properties on {li.label}.",
        f"MATCH (n:{li.label}) "
        f"RETURN n.{dp} AS name, "
        f"datetime() AS current_time, "
        f"date() AS current_date",
    )


def _p28_shortest_path(t: "RelationshipTriple", src: "LabelInfo", tgt: "LabelInfo") -> _Example:
    """Shortest path — shortestPath() for finding minimal paths."""
    dp_src = src.display_property or "name"
    dp_tgt = tgt.display_property or "name"
    sample_src = _sample(src)
    sample_tgt = _sample(tgt)
    return (
        f"Find shortest path between {t.source_label} '{sample_src}' and {t.target_label} '{sample_tgt}'.",
        f"MATCH (a:{t.source_label} {{{dp_src}: '{sample_src}'}}), "
        f"(b:{t.target_label} {{{dp_tgt}: '{sample_tgt}'}}) "
        f"MATCH p = shortestPath((a)-[:{t.rel_type}*]-(b)) "
        f"RETURN p",
    )


def _p29_multiple_match(t1: "RelationshipTriple", t2: "RelationshipTriple") -> "_Example | None":
    """Multiple MATCH clauses — separate pattern matching."""
    if t1.source_label != t2.source_label:
        return None
    return (
        f"Use multiple MATCH clauses to find {t1.source_label} with both {t1.target_label} and {t2.target_label} connections.",
        f"MATCH (a:{t1.source_label})-[:{t1.rel_type}]->(b:{t1.target_label}) "
        f"MATCH (a)-[:{t2.rel_type}]->(c:{t2.target_label}) "
        f"RETURN a, b, c",
    )


def _p30_range_query(li: "LabelInfo") -> _Example:
    """Range queries — numeric/date comparisons with <, >, <=, >=."""
    dp = li.display_property or "name"
    return (
        f"Find {li.label} within a numeric or date range.",
        f"MATCH (n:{li.label}) "
        f"WHERE id(n) >= 100 AND id(n) <= 200 "
        f"RETURN n.{dp} AS name, id(n) AS node_id",
    )


def _p31_order_by_multiple(t: "RelationshipTriple", src: "LabelInfo") -> _Example:
    """ORDER BY multiple fields — complex sorting."""
    dp = src.display_property or "name"
    return (
        f"Sort {t.source_label} by multiple criteria.",
        f"MATCH (a:{t.source_label})-[:{t.rel_type}]->(b:{t.target_label}) "
        f"WITH a, count(b) AS connection_count "
        f"RETURN a.{dp} AS name, connection_count "
        f"ORDER BY connection_count DESC, name ASC",
    )


def _p32_skip_limit_pagination(li: "LabelInfo") -> _Example:
    """SKIP and LIMIT — pagination of results."""
    dp = li.display_property or "name"
    return (
        f"Paginate {li.label} results (skip first 10, return next 5).",
        f"MATCH (n:{li.label}) "
        f"RETURN n.{dp} AS name "
        f"ORDER BY name ASC "
        f"SKIP 10 LIMIT 5",
    )


def _p33_pattern_comprehension(t: "RelationshipTriple", src: "LabelInfo", tgt: "LabelInfo") -> _Example:
    """Pattern comprehension — [(pattern) | expression] for inline traversal."""
    dp_src = src.display_property or "name"
    dp_tgt = tgt.display_property or "name"
    return (
        f"Use pattern comprehension to inline-collect {t.target_label} names.",
        f"MATCH (a:{t.source_label}) "
        f"RETURN a.{dp_src} AS name, "
        f"[(a)-[:{t.rel_type}]->(b:{t.target_label}) | b.{dp_tgt}] AS related_names",
    )


def _p34_map_projection(li: "LabelInfo") -> "_Example | None":
    """Map projection — return custom maps/objects."""
    second = _second_prop(li)
    if not second:
        return None
    dp = li.display_property or "name"
    return (
        f"Return {li.label} as a custom map with selected properties.",
        f"MATCH (n:{li.label}) "
        f"RETURN n {{ .{dp}, .{second} }} AS node_map",
    )


def _p35_string_functions(li: "LabelInfo") -> "_Example | None":
    """String functions — toLower, toUpper, trim, substring, replace."""
    if not li.display_property:
        return None
    dp = li.display_property
    return (
        f"Apply string transformations to {li.label}.{dp}.",
        f"MATCH (n:{li.label}) "
        f"RETURN toLower(n.{dp}) AS lower, "
        f"toUpper(n.{dp}) AS upper, "
        f"trim(n.{dp}) AS trimmed, "
        f"substring(n.{dp}, 0, 5) AS first_five",
    )


def _p36_relationship_properties(t: "RelationshipTriple", src: "LabelInfo") -> _Example:
    """Relationship properties — access and filter on relationship attributes."""
    dp = src.display_property or "name"
    return (
        f"Access properties of the relationship between {t.source_label} and {t.target_label}.",
        f"MATCH (a:{t.source_label})-[r:{t.rel_type}]->(b:{t.target_label}) "
        f"RETURN a.{dp} AS source, "
        f"type(r) AS rel_type, "
        f"properties(r) AS rel_properties, "
        f"b.{dp} AS target",
    )


def _p37_apoc_aggregation(t: "RelationshipTriple", tgt: "LabelInfo") -> _Example:
    """APOC aggregation functions — apoc.agg.* for advanced aggregations."""
    dp_tgt = tgt.display_property or "name"
    return (
        f"Use APOC to aggregate {t.target_label} statistics.",
        f"MATCH (a:{t.source_label})-[:{t.rel_type}]->(b:{t.target_label}) "
        f"WITH a, collect(b.{dp_tgt}) AS names "
        f"RETURN a, apoc.agg.statistics(size(names)) AS stats",
    )


def _p38_call_subquery(t: "RelationshipTriple", src: "LabelInfo") -> _Example:
    """CALL subquery — isolated subquery execution."""
    dp = src.display_property or "name"
    return (
        f"Use CALL subquery to compute intermediate results for {t.source_label}.",
        f"MATCH (a:{t.source_label}) "
        f"CALL {{ "
        f"WITH a "
        f"MATCH (a)-[:{t.rel_type}]->(b:{t.target_label}) "
        f"RETURN count(b) AS connection_count "
        f"}} "
        f"RETURN a.{dp} AS name, connection_count",
    )


def _p39_multiple_labels(li: "LabelInfo") -> "_Example | None":
    """Multiple labels on a node — :Label1:Label2 syntax."""
    if not li.display_property:
        return None
    dp = li.display_property
    return (
        f"Match nodes with {li.label} label (potentially with multiple labels).",
        f"MATCH (n:{li.label}) "
        f"RETURN n.{dp} AS name, labels(n) AS all_labels",
    )


def _p40_null_checks(li: "LabelInfo") -> "_Example | None":
    """Null checks — IS NULL, IS NOT NULL."""
    second = _second_prop(li)
    if not second:
        return None
    dp = li.display_property or "name"
    return (
        f"Filter {li.label} by null/non-null properties.",
        f"MATCH (n:{li.label}) "
        f"WHERE n.{second} IS NOT NULL "
        f"RETURN n.{dp} AS name, n.{second} AS value",
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
    """Patterns 5–40: all remaining Cypher syntax patterns."""
    # Pattern 5: Undirected
    if len(labels) >= 2:
        _add_if_room(examples, _p5_undirected(labels[0]), max_n)

    # Pattern 6: Property-specific RETURN
    if labels:
        li = next((l for l in labels if len(l.properties) > 1), labels[0])
        _add_if_room(examples, _p6_property_return(li), max_n)

    # Pattern 7: Aggregation with COUNT/ORDER/LIMIT
    if triples:
        t = triples[0]
        src = label_map.get(t.source_label)
        if src:
            _add_if_room(examples, _p7_aggregation(t, src), max_n)

    # Pattern 8: WHERE filter
    if labels:
        li = next(
            (l for l in labels if l.display_property and l.sample_values),
            labels[0],
        )
        _add_if_room(examples, _p8_where_filter(li), max_n)

    # Pattern 9: Reverse-filter (keep arrow direction, filter by target)
    if triples:
        t = triples[0]
        tgt = label_map.get(t.target_label)
        if tgt:
            _add_if_room(examples, _p9_reverse_filter(t, tgt), max_n)

    # Pattern 10: Multi-hop "connected to" via chain
    if chains and len(chains[0]) >= 2:
        t1, t2 = chains[0][0], chains[0][1]
        src = label_map.get(t1.source_label)
        if src:
            _add_if_room(examples, _p10_connected_via(t1, t2, src), max_n)

    # Pattern 11: DISTINCT
    if labels:
        li = next((l for l in labels if l.display_property and len(l.properties) > 1), labels[0] if labels else None)
        if li:
            _add_if_room(examples, _p11_distinct(li), max_n)

    # Pattern 12: UNION
    if triples:
        t = triples[0]
        src = label_map.get(t.source_label)
        tgt = label_map.get(t.target_label)
        if src and tgt:
            _add_if_room(examples, _p12_union(t, src, tgt), max_n)

    # Pattern 13: WITH clause
    if triples:
        t = triples[0]
        src = label_map.get(t.source_label)
        if src:
            _add_if_room(examples, _p13_with_clause(t, src), max_n)

    # Pattern 14: CASE conditional
    if labels:
        li = next((l for l in labels if len(l.properties) > 1), labels[0] if labels else None)
        if li:
            _add_if_room(examples, _p14_case_conditional(li), max_n)

    # Pattern 15: COLLECT aggregation
    if triples:
        t = triples[0]
        src = label_map.get(t.source_label)
        tgt = label_map.get(t.target_label)
        if src and tgt:
            _add_if_room(examples, _p15_collect_aggregation(t, src, tgt), max_n)

    # Pattern 16: UNWIND
    if labels:
        _add_if_room(examples, _p16_unwind(labels[0]), max_n)

    # Pattern 17: OPTIONAL MATCH
    if triples:
        t = triples[0]
        src = label_map.get(t.source_label)
        if src:
            _add_if_room(examples, _p17_optional_match(t, src), max_n)

    # Pattern 18: EXISTS subquery
    if triples:
        t = triples[0]
        src = label_map.get(t.source_label)
        if src:
            _add_if_room(examples, _p18_exists_subquery(t, src), max_n)

    # Pattern 19: NOT negation
    if triples:
        t = triples[0]
        src = label_map.get(t.source_label)
        if src:
            _add_if_room(examples, _p19_not_negation(t, src), max_n)

    # Pattern 20: IN list membership
    if labels:
        li = next((l for l in labels if l.display_property), labels[0] if labels else None)
        if li:
            _add_if_room(examples, _p20_in_list(li), max_n)

    # Pattern 21: String operations (CONTAINS, STARTS WITH, ENDS WITH)
    if labels:
        li = next((l for l in labels if l.display_property), labels[0] if labels else None)
        if li:
            _add_if_room(examples, _p21_string_operations(li), max_n)

    # Pattern 22: Mathematical aggregations (sum, avg, min, max)
    if triples:
        t = triples[0]
        _add_if_room(examples, _p22_math_aggregations(t), max_n)

    # Pattern 23: List comprehension
    if triples:
        t = triples[0]
        tgt = label_map.get(t.target_label)
        if tgt:
            _add_if_room(examples, _p23_list_comprehension(t, tgt), max_n)

    # Pattern 24: Variable length paths
    if triples:
        t = triples[0]
        src = label_map.get(t.source_label)
        if src:
            _add_if_room(examples, _p24_variable_length_path(t, src), max_n)

    # Pattern 25: Complex WHERE (AND, OR, NOT)
    if triples:
        t = triples[0]
        src = label_map.get(t.source_label)
        tgt = label_map.get(t.target_label)
        if src and tgt:
            _add_if_room(examples, _p25_complex_where(t, src, tgt), max_n)

    # Pattern 26: APOC path expansion
    if triples:
        t = triples[0]
        src = label_map.get(t.source_label)
        if src:
            _add_if_room(examples, _p26_apoc_path_expand(t, src), max_n)

    # Pattern 27: Date/time operations
    if labels:
        _add_if_room(examples, _p27_datetime_operations(labels[0]), max_n)

    # Pattern 28: Shortest path
    if triples:
        t = triples[0]
        src = label_map.get(t.source_label)
        tgt = label_map.get(t.target_label)
        if src and tgt:
            _add_if_room(examples, _p28_shortest_path(t, src, tgt), max_n)

    # Pattern 29: Multiple MATCH clauses
    if len(triples) >= 2:
        t1, t2 = triples[0], triples[1]
        _add_if_room(examples, _p29_multiple_match(t1, t2), max_n)

    # Pattern 30: Range queries
    if labels:
        _add_if_room(examples, _p30_range_query(labels[0]), max_n)

    # Pattern 31: ORDER BY multiple fields
    if triples:
        t = triples[0]
        src = label_map.get(t.source_label)
        if src:
            _add_if_room(examples, _p31_order_by_multiple(t, src), max_n)

    # Pattern 32: SKIP and LIMIT pagination
    if labels:
        _add_if_room(examples, _p32_skip_limit_pagination(labels[0]), max_n)

    # Pattern 33: Pattern comprehension
    if triples:
        t = triples[0]
        src = label_map.get(t.source_label)
        tgt = label_map.get(t.target_label)
        if src and tgt:
            _add_if_room(examples, _p33_pattern_comprehension(t, src, tgt), max_n)

    # Pattern 34: Map projection
    if labels:
        li = next((l for l in labels if len(l.properties) > 1), labels[0] if labels else None)
        if li:
            _add_if_room(examples, _p34_map_projection(li), max_n)

    # Pattern 35: String functions (toLower, toUpper, trim, substring)
    if labels:
        li = next((l for l in labels if l.display_property), labels[0] if labels else None)
        if li:
            _add_if_room(examples, _p35_string_functions(li), max_n)

    # Pattern 36: Relationship properties
    if triples:
        t = triples[0]
        src = label_map.get(t.source_label)
        if src:
            _add_if_room(examples, _p36_relationship_properties(t, src), max_n)

    # Pattern 37: APOC aggregation
    if triples:
        t = triples[0]
        tgt = label_map.get(t.target_label)
        if tgt:
            _add_if_room(examples, _p37_apoc_aggregation(t, tgt), max_n)

    # Pattern 38: CALL subquery
    if triples:
        t = triples[0]
        src = label_map.get(t.source_label)
        if src:
            _add_if_room(examples, _p38_call_subquery(t, src), max_n)

    # Pattern 39: Multiple labels
    if labels:
        li = next((l for l in labels if l.display_property), labels[0] if labels else None)
        if li:
            _add_if_room(examples, _p39_multiple_labels(li), max_n)

    # Pattern 40: Null checks (IS NULL, IS NOT NULL)
    if labels:
        li = next((l for l in labels if len(l.properties) > 1), labels[0] if labels else None)
        if li:
            _add_if_room(examples, _p40_null_checks(li), max_n)


# ── Public API ─────────────────────────────────────────────────────────────────


def generate_few_shot_examples(
    topology: "GraphTopology",
    max_examples: int = 50,
    manual_overrides: list[dict] | None = None,
    question: str | None = None,
) -> str:
    """
    Build a few-shot block from *topology*.

    Parameters
    ----------
    topology:
        The live graph topology to derive examples from.
    max_examples:
        Hard cap on the number of examples (default 50).
    manual_overrides:
        Optional list of dicts with ``question`` and ``cypher`` keys that are
        prepended verbatim and count against *max_examples*.
    question:
        Optional user question; when provided, triples whose source/target
        label appears in the question are prioritised first.

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

    # Prioritise triples whose labels appear in the question
    triples = topology.triples
    if question:
        q_lower = question.lower()
        relevant = [
            t for t in triples
            if t.source_label.lower() in q_lower or t.target_label.lower() in q_lower
        ]
        rest = [t for t in triples if t not in relevant]
        triples = relevant + rest

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
    for i, (q, cypher) in enumerate(examples[:max_examples], start=1):
        lines.append(f"Example {i}:")
        lines.append(f"  Question: {q}")
        lines.append(f"  Cypher:   {cypher}")
        lines.append("")

    result = "\n".join(lines).rstrip()
    logger.debug("dynamic_examples: generated %d examples.", len(examples[:max_examples]))
    return result
