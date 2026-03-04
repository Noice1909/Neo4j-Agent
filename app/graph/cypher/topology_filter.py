"""
Query-aware topology filtering (Strategy #6).

``filter_topology()`` narrows the full graph topology down to only the labels
and relationships relevant to a specific user question, reducing prompt bloat
and keeping the LLM focused on the right schema subset.

If no labels from the question are recognised, the full topology is returned
unchanged so that the query can still proceed.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.graph.topology import GraphTopology
    from app.graph.cypher.entity_resolution.models import ResolutionResult

logger = logging.getLogger(__name__)


def _question_labels(question: str, label_names: list[str]) -> set[str]:
    """Return topology labels whose name appears (case-insensitive) in *question*."""
    q_lower = question.lower()
    return {label for label in label_names if label.lower() in q_lower}


def filter_topology(
    question: str,
    topology: "GraphTopology",
    resolution: "ResolutionResult | None" = None,
) -> "GraphTopology":
    """
    Return a topology subset containing only question-relevant labels and
    their one-hop neighbours.  Falls back to the full topology if nothing matches.

    Parameters
    ----------
    question:
        The (possibly coreference-resolved) user question.
    topology:
        Full graph topology from the schema cache.
    resolution:
        Optional entity-resolution result; its resolved question is also
        scanned for label mentions to widen the match.
    """
    from app.graph.topology import GraphTopology, _find_chains  # noqa: PLC0415

    # ── 1. Collect matched labels ─────────────────────────────────────────────
    label_names = topology.label_names
    matched = _question_labels(question, label_names)
    if resolution:
        matched |= _question_labels(resolution.resolved_question, label_names)

    if not matched:
        logger.debug("topology_filter: no label match in question — using full topology.")
        return topology

    # ── 2. One-hop expansion ──────────────────────────────────────────────────
    adj = topology.adjacency
    expanded: set[str] = set(matched)
    for label in matched:
        expanded.update(adj.get(label, set()))

    # ── 3. Filter triples and labels ─────────────────────────────────────────
    filtered_triples = [
        t for t in topology.triples
        if t.source_label in expanded and t.target_label in expanded
    ]
    if not filtered_triples:
        logger.debug("topology_filter: no matching triples — using full topology.")
        return topology

    filtered_labels = [li for li in topology.labels if li.label in expanded]
    filtered_chains = _find_chains(filtered_triples)

    logger.debug(
        "topology_filter: %d→%d triples, %d→%d labels (matched: %s).",
        len(topology.triples), len(filtered_triples),
        len(topology.labels), len(filtered_labels),
        ", ".join(sorted(matched)),
    )
    return GraphTopology(
        labels=filtered_labels,
        triples=filtered_triples,
        chains=filtered_chains,
    )
