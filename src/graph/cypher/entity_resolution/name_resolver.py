"""
Layer 2 — Entity Name Resolution (data-aware DB lookup).

Strategies (fastest → slowest): 2a) Full-text index (~1 ms),
2b) APOC multi-signal (~50-500 ms), 2c) APOC phonetic (~50-500 ms).
Read-only — no indexes created, no data written.
"""
from __future__ import annotations

import asyncio
import difflib
import logging
import re
from functools import lru_cache
from typing import Any

from src.graph.cypher.entity_resolution.capabilities import (
    check_apoc_available,
    detect_fulltext_index,
)
from src.graph.cypher.entity_resolution.label_resolver import LabelResolver
from src.graph.cypher.entity_resolution.models import (
    Correction,
    FULLTEXT_INDEX_NAME,
    _LUCENE_SPECIAL_RE,
)

logger = logging.getLogger(__name__)


# ── EntityNameResolver ───────────────────────────────────────────────────────


class EntityNameResolver:
    """
    Resolve typos in entity names via Neo4j lookups.

    Designed for **read-only accounts** and **3M+ node databases**:

    - **2a — Full-text index** (if admin created ``entityNameIndex``):
      Uses Lucene fuzzy ``~`` operator.  Sub-millisecond, zero RAM.
    - **2b — APOC multi-signal** (no index, APOC available):
      Label-scoped query using averaged Levenshtein + Sørensen-Dice
      + Jaro-Winkler scores.
    - **2c — APOC phonetic** (sounds-alike fallback):
      doubleMetaphone matching for phonetically similar names.

    All queries are LRU-cached (256 entries) to avoid repeated DB hits.
    """

    def __init__(
        self,
        graph: Any,
        schema: str = "",
        fuzzy_threshold: float = 0.75,
        max_candidates: int = 5,
        fulltext_index_name: str = FULLTEXT_INDEX_NAME,
        display_properties: list[str] | None = None,
    ) -> None:
        self._graph = graph
        self._schema = schema
        self._fuzzy_threshold = fuzzy_threshold
        self._max_candidates = max_candidates
        self._index_name = fulltext_index_name
        # Build the COALESCE expression for name retrieval dynamically
        props = display_properties or ["name", "title"]
        coalesce_args = ", ".join(f"node.{p}" for p in props)
        self._name_coalesce = f"COALESCE({coalesce_args})"

        # Detect capabilities once (read-only probes)
        self._has_fulltext = (
            detect_fulltext_index(graph, fulltext_index_name) if schema else False
        )
        self._has_apoc = (
            check_apoc_available(graph)
            if (schema and not self._has_fulltext) else False
        )

        # Extract labels from schema for label-scoped APOC queries
        self._labels = LabelResolver._extract_labels(schema) if schema else []

    async def resolve(
        self, question: str, known_labels: list[str],
    ) -> tuple[str, list[Correction]]:
        """
        Extract quoted strings and capitalised phrases, then look up
        close matches in the database.
        """
        corrections: list[Correction] = []
        candidates = self._extract_candidates(question)

        if not candidates:
            return question, corrections

        # Use labels from Layer 1 if available, otherwise from schema
        target_labels = known_labels or self._labels

        resolved_question = question
        for candidate in candidates:
            match = await self._find_closest_match(candidate, target_labels)
            if match and match.lower() != candidate.lower():
                similarity = difflib.SequenceMatcher(
                    None, candidate.lower(), match.lower(),
                ).ratio()
                if similarity >= self._fuzzy_threshold:
                    corrections.append(Correction(
                        original=candidate,
                        corrected=match,
                        layer="entity_name",
                        confidence=similarity,
                    ))
                    resolved_question = resolved_question.replace(
                        candidate, match,
                    )

        return resolved_question, corrections

    @staticmethod
    def _extract_candidates(question: str) -> list[str]:
        """Extract potential entity names from the question."""
        candidates: list[str] = []

        # Quoted strings: "Tom Hanks", 'The Matrix'
        for match in re.finditer(r"""["']([^"']+)["']""", question):
            candidates.append(match.group(1))

        # Capitalised multi-word phrases (likely entity names)
        for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", question):
            phrase = match.group(1)
            if phrase not in candidates:
                candidates.append(phrase)

        return candidates

    # ── 2a: Full-text index query ────────────────────────────────────────

    @lru_cache(maxsize=256)
    def _fulltext_query(self, candidate: str) -> list[dict]:
        """Lucene fuzzy search via pre-existing full-text index (~1 ms)."""
        escaped = _LUCENE_SPECIAL_RE.sub(r"\\\\\\1", candidate)
        search_term = escaped + "~"
        cypher = (
            "CALL db.index.fulltext.queryNodes($indexName, $term) "
            "YIELD node, score "
            "RETURN labels(node)[0] AS label, "
            f"{self._name_coalesce} AS name, score "
            "LIMIT $limit"
        )
        try:
            return self._graph.query(
                cypher,
                params={
                    "indexName": self._index_name,
                    "term": search_term,
                    "limit": self._max_candidates,
                },
            )
        except Exception as exc:
            logger.warning("Full-text entity lookup failed: %s", exc)
            return []

    # ── 2b: APOC multi-signal query (label-scoped) ──────────────────────

    @lru_cache(maxsize=256)
    def _apoc_multi_signal_query(
        self, candidate_lower: str, label: str,
    ) -> list[dict]:
        """Label-scoped APOC query combining three similarity metrics."""
        cypher = (
            f"MATCH (n:`{label}`) "
            "WHERE n.name IS NOT NULL "
            "WITH n, "
            "  apoc.text.levenshteinSimilarity(toLower(n.name), $candidate) AS levSim, "
            "  apoc.text.sorensenDiceSimilarity(toLower(n.name), $candidate) AS sdSim, "
            "  apoc.text.jaroWinklerDistance(toLower(n.name), $candidate) AS jwSim "
            "WITH n, (levSim + sdSim + jwSim) / 3.0 AS avgScore "
            "WHERE avgScore > $threshold "
            "RETURN labels(n)[0] AS label, n.name AS name, avgScore AS score "
            "ORDER BY avgScore DESC "
            "LIMIT $limit"
        )
        try:
            return self._graph.query(
                cypher,
                params={
                    "candidate": candidate_lower,
                    "threshold": self._fuzzy_threshold,
                    "limit": self._max_candidates,
                },
            )
        except Exception as exc:
            logger.warning(
                "APOC multi-signal lookup failed for label '%s': %s",
                label, exc,
            )
            return []

    # ── 2c: APOC phonetic query (label-scoped) ──────────────────────────

    @lru_cache(maxsize=256)
    def _apoc_phonetic_query(
        self, candidate_lower: str, label: str,
    ) -> list[dict]:
        """Phonetic matching via APOC doubleMetaphone."""
        cypher = (
            f"MATCH (n:`{label}`) "
            "WHERE n.name IS NOT NULL "
            "WITH n, "
            "  apoc.text.doubleMetaphone(n.name) AS nodePhonetic, "
            "  apoc.text.doubleMetaphone($candidate) AS queryPhonetic "
            "WHERE nodePhonetic = queryPhonetic "
            "RETURN labels(n)[0] AS label, n.name AS name "
            "LIMIT $limit"
        )
        try:
            return self._graph.query(
                cypher,
                params={
                    "candidate": candidate_lower,
                    "limit": self._max_candidates,
                },
            )
        except Exception as exc:
            logger.warning(
                "APOC phonetic lookup failed for label '%s': %s",
                label, exc,
            )
            return []

    # ── Resolution orchestration ─────────────────────────────────────────

    async def _try_label_scoped(
        self,
        query_fn: Any,
        candidate: str,
        labels: list[str],
    ) -> str | None:
        """Run *query_fn* for each label and return the first good match."""
        loop = asyncio.get_running_loop()
        for label in labels:
            results = await loop.run_in_executor(
                None, query_fn, candidate.lower(), label,
            )
            best = self._pick_best(results, candidate)
            if best:
                return best
        return None

    async def _find_closest_match(
        self, candidate: str, labels: list[str],
    ) -> str | None:
        """Try each sub-layer until a confident match is found."""
        # ── 2a: Full-text index (fastest) ────────────────────────────────
        if self._has_fulltext:
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                None, self._fulltext_query, candidate,
            )
            best = self._pick_best(results, candidate)
            if best:
                return best

        # ── 2b: APOC multi-signal (label-scoped) ────────────────────────
        if self._has_apoc and labels:
            best = await self._try_label_scoped(
                self._apoc_multi_signal_query, candidate, labels,
            )
            if best:
                return best

        # ── 2c: APOC phonetic (label-scoped) ─────────────────────────────
        if self._has_apoc and labels:
            best = await self._try_label_scoped(
                self._apoc_phonetic_query, candidate, labels,
            )
            if best:
                return best

        return None

    def _pick_best(
        self, results: list[dict], candidate: str,
    ) -> str | None:
        """Select the best matching name from query results."""
        if not results:
            return None

        best_match: str | None = None
        best_score = 0.0

        for row in results:
            name = str(row.get("name", ""))
            if not name:
                continue
            score = difflib.SequenceMatcher(
                None, candidate.lower(), name.lower(),
            ).ratio()
            if score > best_score:
                best_score = score
                best_match = name

        return best_match if best_score >= self._fuzzy_threshold else None
