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

    @property
    def label_names(self) -> list[str]:
        """Sorted list of all node label strings."""
        return sorted(li.label for li in self.labels)

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

_LABELS_QUERY = "CALL db.labels() YIELD label RETURN label ORDER BY label"


def _sample_query(label: str) -> str:
    """Return Cypher to fetch one node's properties for *label*."""
    safe = re.sub(r"[^A-Za-z0-9_]", "", label)
    return f"MATCH (n:`{safe}`) RETURN properties(n) AS props LIMIT 1"


def _detect_display_property(props: list[str]) -> str | None:
    """Pick the best display property using a priority list."""
    lower = {p.lower(): p for p in props}
    for candidate in _DISPLAY_PROPERTY_PRIORITY:
        if candidate in lower:
            return lower[candidate]
    return props[0] if props else None


# ── Chain detection (DFS) ─────────────────────────────────────────────────────

def _find_chains(
    triples: list[RelationshipTriple],
    max_depth: int = 3,
    max_chains: int = 20,
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

        display = _detect_display_property(props)
        label_infos.append(
            LabelInfo(
                label=label,
                properties=props,
                display_property=display,
                sample_values=sample_values,
            )
        )

    # ── 4. Multi-hop chain detection ───────────────────────────────────────────
    chains = _find_chains(triples)
    logger.info("Topology: detected %d multi-hop chains.", len(chains))

    topology = GraphTopology(labels=label_infos, triples=triples, chains=chains)
    logger.info(
        "Topology extraction complete: %d labels, %d triples, %d chains.",
        len(topology.labels),
        len(topology.triples),
        len(topology.chains),
    )
    return topology
