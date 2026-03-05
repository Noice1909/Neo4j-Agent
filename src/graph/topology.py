"""
Dynamic Neo4j graph topology extraction.

Derives the full schema topology (labels, relationship triples, property info,
multi-hop chains) from a live Neo4j instance using read-only Cypher queries.
Nothing is hardcoded — point at any database and the topology is rebuilt.

Data model
──────────
RelationshipTriple  — one (A)-[:R]->(B) fact
LabelInfo           — one node label with its properties, display property,
                      and one sample value per property
GraphTopology       — the full extracted topology + derived multi-hop chains
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_neo4j import Neo4jGraph

logger = logging.getLogger(__name__)

# ── Data model ────────────────────────────────────────────────────────────────

_DISPLAY_PROPERTY_PRIORITY = ("name", "title", "label", "id", "code", "key")


@dataclass(frozen=True)
class RelationshipTriple:
    """One directed relationship fact: (source)-[:rel_type]->(target)."""

    source_label: str
    rel_type: str
    target_label: str
    count: int = 0          # relationship instance count from apoc.meta.schema()
    bidirectional: bool = False  # True if the same rel_type exists in both directions

    def __str__(self) -> str:  # noqa: D105
        return f"(:{self.source_label})-[:{self.rel_type}]->(:{self.target_label})"


@dataclass
class LabelInfo:
    """Metadata for one node label."""

    label: str
    properties: list[str] = field(default_factory=list)
    display_property: str | None = None  # best property for human-readable returns
    sample_values: dict[str, str] = field(default_factory=dict)


@dataclass
class GraphTopology:
    """Full extracted topology of the live Neo4j graph."""

    labels: list[LabelInfo] = field(default_factory=list)
    triples: list[RelationshipTriple] = field(default_factory=list)
    chains: list[list[RelationshipTriple]] = field(default_factory=list)
    # primary_label → [other labels that co-occur on the same physical nodes]
    label_aliases: dict[str, list[str]] = field(default_factory=dict)

    @property
    def label_names(self) -> list[str]:
        """Sorted list of all node label strings."""
        return sorted(li.label for li in self.labels)

    @property
    def valid_rel_types(self) -> set[str]:
        """Set of all relationship type strings present in the topology."""
        return {t.rel_type for t in self.triples}

    @property
    def adjacency(self) -> dict[str, set[str]]:
        """label → set of directly connected labels (one hop, either direction)."""
        adj: dict[str, set[str]] = {}
        for t in self.triples:
            adj.setdefault(t.source_label, set()).add(t.target_label)
            adj.setdefault(t.target_label, set()).add(t.source_label)
        return adj

    @property
    def display_properties(self) -> list[str]:
        """Unique ordered display properties across all labels (for COALESCE)."""
        seen: set[str] = set()
        result: list[str] = []
        for li in self.labels:
            if li.display_property and li.display_property not in seen:
                seen.add(li.display_property)
                result.append(li.display_property)
        # Always include fallbacks even if not detected
        for fallback in _DISPLAY_PROPERTY_PRIORITY:
            if fallback not in seen:
                seen.add(fallback)
                result.append(fallback)
        return result


# ── Cypher helpers ────────────────────────────────────────────────────────────

_TRIPLES_QUERY = """\
MATCH (a)-[r]->(b)
WHERE labels(a) <> [] AND labels(b) <> []
RETURN DISTINCT labels(a)[0] AS src, type(r) AS rel, labels(b)[0] AS tgt
LIMIT 500
"""

# Detects nodes that carry more than one label so we can build an alias map.
_MULTI_LABEL_QUERY = """\
MATCH (n) WHERE size(labels(n)) > 1
RETURN DISTINCT labels(n) AS label_group LIMIT 300
"""

_LABELS_QUERY = "CALL db.labels() YIELD label RETURN label ORDER BY label"


def _sample_query(label: str) -> str:
    """Return Cypher to fetch one node's properties for *label*."""
    safe = re.sub(r"\W", "", label)
    return f"MATCH (n:`{safe}`) RETURN properties(n) AS props LIMIT 1"


def _build_label_aliases(
    multi_label_rows: list[dict],
    known_labels: set[str],
) -> dict[str, list[str]]:
    """
    Build a primary_label → [alias_labels] map from multi-label node groups.

    The "primary" label is whichever member of the group already appears as a
    source or target in the triples query (i.e. is in *known_labels*), chosen
    alphabetically.  Unknown co-labels are listed as aliases.
    """
    aliases: dict[str, list[str]] = {}
    for row in multi_label_rows:
        _merge_group(row.get("label_group") or [], known_labels, aliases)
    return aliases


def _merge_group(
    group: list[str],
    known_labels: set[str],
    aliases: dict[str, list[str]],
) -> None:
    """Merge one label group into the alias map (mutates *aliases*)."""
    if len(group) < 2:
        return
    known_in_group = sorted(lbl for lbl in group if lbl in known_labels)
    if not known_in_group:
        return
    primary = known_in_group[0]
    others = known_in_group[1:] + [lbl for lbl in group if lbl not in known_labels]
    current = aliases.setdefault(primary, [])
    current.extend(alias for alias in others if alias not in current)


def _detect_display_property(props: list[str]) -> str | None:
    """Pick the best display property using a priority list."""
    lower = {p.lower(): p for p in props}
    for candidate in _DISPLAY_PROPERTY_PRIORITY:
        if candidate in lower:
            return lower[candidate]
    return props[0] if props else None


async def _fetch_label_infos(
    loop: asyncio.AbstractEventLoop,
    graph: "Neo4jGraph",
    label_names: list[str],
) -> list[LabelInfo]:
    """Fetch properties and sample values for every label (one query each)."""
    label_infos: list[LabelInfo] = []
    for label in label_names:
        try:
            rows = await loop.run_in_executor(None, graph.query, _sample_query(label))
            props_map: dict = rows[0]["props"] if rows and rows[0].get("props") else {}
            props = list(props_map.keys())
            sample_values = {k: str(v) for k, v in props_map.items() if v is not None}
        except Exception as exc:
            logger.debug("Topology: property sample for %s failed: %s", label, exc)
            props = []
            sample_values = {}
        label_infos.append(LabelInfo(
            label=label,
            properties=props,
            display_property=_detect_display_property(props),
            sample_values=sample_values,
        ))
    return label_infos


# ── Chain detection (DFS) ─────────────────────────────────────────────────────

def _find_chains(
    triples: list[RelationshipTriple],
    max_depth: int = 5,
    max_chains: int = 40,
) -> list[list[RelationshipTriple]]:
    """
    Find multi-hop paths up to *max_depth* hops via depth-first search.

    Each returned chain is a list of consecutive RelationshipTriple objects
    where triple[i].target_label == triple[i+1].source_label.
    Only chains of length >= 2 are returned (single triples are already in
    GraphTopology.triples).
    """
    # Build adjacency: source_label -> list[RelationshipTriple]
    adj: dict[str, list[RelationshipTriple]] = {}
    for t in triples:
        adj.setdefault(t.source_label, []).append(t)

    chains: list[list[RelationshipTriple]] = []

    def _dfs(path: list[RelationshipTriple], visited_labels: set[str]) -> None:
        if len(chains) >= max_chains:
            return
        if len(path) >= 2:
            chains.append(list(path))
        if len(path) >= max_depth:
            return
        current_label = path[-1].target_label
        for next_triple in adj.get(current_label, []):
            if next_triple.target_label not in visited_labels:
                path.append(next_triple)
                visited_labels.add(next_triple.target_label)
                _dfs(path, visited_labels)
                path.pop()
                visited_labels.discard(next_triple.target_label)

    for start_triple in triples:
        if len(chains) >= max_chains:
            break
        _dfs([start_triple], {start_triple.source_label, start_triple.target_label})

    return chains


# ── APOC meta helper ─────────────────────────────────────────────────────────

def _parse_apoc_meta(
    meta: dict,
) -> dict[tuple[str, str, str], dict]:
    """
    Parse ``apoc.meta.schema()`` output into a lookup keyed by
    ``(source_label, rel_type, target_label)`` with ``{"count": int}`` values.
    """
    result: dict[tuple[str, str, str], dict] = {}
    for src_label, label_data in meta.items():
        if not isinstance(label_data, dict):
            continue
        for rel_type, rel_data in label_data.get("relationships", {}).items():
            if not isinstance(rel_data, dict):
                continue
            count = int(rel_data.get("count", 0))
            for tgt_label in rel_data.get("labels", []):
                result[(src_label, rel_type, str(tgt_label))] = {"count": count}
    return result


def _enrich_triples(
    triples: list[RelationshipTriple],
    apoc_map: dict[tuple[str, str, str], dict],
) -> list[RelationshipTriple]:
    """Rebuild *triples* with APOC-derived count and bidirectionality flags."""
    reverse_set = {(t.target_label, t.rel_type, t.source_label) for t in triples}
    return [
        RelationshipTriple(
            t.source_label, t.rel_type, t.target_label,
            count=apoc_map.get((t.source_label, t.rel_type, t.target_label), {}).get("count", 0),
            bidirectional=(t.target_label, t.rel_type, t.source_label) in reverse_set,
        )
        for t in triples
    ]


# ── Main extraction function ──────────────────────────────────────────────────

async def extract_topology(graph: "Neo4jGraph") -> GraphTopology:
    """
    Derive the full ``GraphTopology`` from a live Neo4j instance.

    Runs three read-only Cypher queries in the thread-pool executor (since
    ``Neo4jGraph.query()`` is synchronous) and builds the topology model.
    Does not raise — on partial failure it logs a warning and returns whatever
    it managed to extract.
    """
    loop = asyncio.get_running_loop()

    # ── 1. Relationship triples ────────────────────────────────────────────────
    try:
        rows = await loop.run_in_executor(None, graph.query, _TRIPLES_QUERY)
        triples = [
            RelationshipTriple(
                source_label=r["src"],
                rel_type=r["rel"],
                target_label=r["tgt"],
            )
            for r in rows
            if r.get("src") and r.get("rel") and r.get("tgt")
        ]
        logger.info("Topology: extracted %d relationship triples.", len(triples))
    except Exception as exc:
        logger.warning("Topology: relationship triple query failed: %s", exc)
        triples = []

    # ── 1b. Optional APOC enrichment (counts + bidirectionality) ──────────────
    if triples:
        try:
            apoc_rows = await loop.run_in_executor(
                None, graph.query, "CALL apoc.meta.schema() YIELD value RETURN value",
            )
            if apoc_rows and apoc_rows[0].get("value"):
                apoc_map = _parse_apoc_meta(apoc_rows[0]["value"])
                triples = _enrich_triples(triples, apoc_map)
                logger.info("Topology: APOC meta enrichment complete.")
        except Exception:
            logger.debug("APOC meta.schema() unavailable — using unenriched triples.")

    # ── 1c. Multi-label alias detection ───────────────────────────────────────
    label_aliases: dict[str, list[str]] = {}
    try:
        known_labels = {t.source_label for t in triples} | {t.target_label for t in triples}
        multi_rows = await loop.run_in_executor(None, graph.query, _MULTI_LABEL_QUERY)
        label_aliases = _build_label_aliases(multi_rows, known_labels)
        if label_aliases:
            logger.info(
                "Topology: detected %d multi-label alias group(s): %s",
                len(label_aliases),
                label_aliases,
            )
    except Exception as exc:
        logger.debug("Topology: multi-label alias detection failed: %s", exc)

    # ── 2. Node labels ─────────────────────────────────────────────────────────
    label_names: list[str] = []
    try:
        rows = await loop.run_in_executor(None, graph.query, _LABELS_QUERY)
        label_names = [r["label"] for r in rows if r.get("label")]
        logger.info("Topology: found %d node labels.", len(label_names))
    except Exception as exc:
        logger.warning("Topology: label query failed: %s — deriving from triples.", exc)
        # Fall back to labels inferred from triples
        seen: set[str] = set()
        for t in triples:
            seen.add(t.source_label)
            seen.add(t.target_label)
        label_names = sorted(seen)

    # ── 3. Properties + sample values per label ────────────────────────────────
    label_infos = await _fetch_label_infos(loop, graph, label_names)

    # ── 4. Multi-hop chain detection ───────────────────────────────────────────
    chains = _find_chains(triples)
    logger.info("Topology: detected %d multi-hop chains.", len(chains))

    topology = GraphTopology(
        labels=label_infos,
        triples=triples,
        chains=chains,
        label_aliases=label_aliases,
    )
    logger.info(
        "Topology extraction complete: %d labels, %d triples, %d chains.",
        len(topology.labels),
        len(topology.triples),
        len(topology.chains),
    )
    return topology
