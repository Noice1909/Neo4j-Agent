"""
Dynamic entity resolution pipeline (3-layer).

Corrects user queries *before* Cypher generation by resolving:
  - Wrong labels/categories  (Layer 1 — schema-aware fuzzy matching)
  - Typos in entity names    (Layer 2 — data-aware DB lookup)
  - Ambiguous references     (Layer 3 — LLM-assisted fallback)

The pipeline is feature-flagged via ``ENTITY_RESOLUTION_ENABLED`` and
sits between coreference resolution and the Cypher prompt.
"""
from __future__ import annotations

import asyncio
import difflib
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List

from langchain_core.language_models import BaseChatModel

from src.graph.cypher.synonyms import build_synonym_map

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class Correction:
    """A single correction applied to the user's query."""
    original: str
    corrected: str
    layer: str          # "label", "entity_name", or "llm"
    confidence: float   # 0.0–1.0


@dataclass
class ResolutionResult:
    """Result of the entity resolution pipeline."""
    original_question: str
    resolved_question: str
    corrections: List[Correction] = field(default_factory=list)

    @property
    def was_corrected(self) -> bool:
        return len(self.corrections) > 0


# ── Layer 1: Label / Category Resolution ─────────────────────────────────────


class LabelResolver:
    """
    Resolve wrong labels/categories using schema-aware fuzzy matching.

    Compares words in the user's question against actual Neo4j node labels
    and relationship types, using a synonym map and fuzzy matching.
    """

    def __init__(
        self,
        schema: str,
        synonym_overrides: str = "",
        fuzzy_threshold: float = 0.75,
    ) -> None:
        self._labels = self._extract_labels(schema)
        self._rel_types = self._extract_relationship_types(schema)
        self._all_types = self._labels + self._rel_types
        self._synonym_map = build_synonym_map(self._labels, synonym_overrides)
        self._fuzzy_threshold = fuzzy_threshold

    @staticmethod
    def _extract_labels(schema: str) -> list[str]:
        """Extract node labels from the schema string."""
        # Schema format: "Node properties: \n Movie {title: STRING, ...}"
        labels: list[str] = []
        for match in re.finditer(r"^\s*(\w+)\s*\{", schema, re.MULTILINE):
            label = match.group(1)
            if label not in ("Node", "Relationship", "The"):
                labels.append(label)
        return list(dict.fromkeys(labels))  # dedupe, preserve order

    @staticmethod
    def _extract_relationship_types(schema: str) -> list[str]:
        """Extract relationship types from the schema string."""
        types: list[str] = []
        for match in re.finditer(r"\[:(\w+)\]", schema):
            types.append(match.group(1))
        return list(dict.fromkeys(types))

    def resolve(self, question: str) -> tuple[str, list[Correction]]:
        """
        Check each word/phrase in the question for label mismatches.

        Returns the corrected question and a list of corrections made.
        """
        corrections: list[Correction] = []
        words = question.split()
        corrected_words: list[str] = []

        for word in words:
            clean = re.sub(r"[^\w]", "", word).lower()
            if not clean or len(clean) < 2:
                corrected_words.append(word)
                continue

            # Check synonym map first (exact match)
            if clean in self._synonym_map:
                canonical = self._synonym_map[clean]
                # Only correct if it's actually different from a known label
                if clean not in [l.lower() for l in self._all_types]:
                    corrections.append(Correction(
                        original=word,
                        corrected=canonical,
                        layer="label",
                        confidence=1.0,
                    ))
                    corrected_words.append(
                        word.replace(re.sub(r"[^\w]", "", word), canonical)
                    )
                    continue

            # Fuzzy match against known labels
            close = difflib.get_close_matches(
                clean,
                [l.lower() for l in self._all_types],
                n=1,
                cutoff=self._fuzzy_threshold,
            )
            if close:
                # Map back to properly-cased label
                matched_label = next(
                    l for l in self._all_types if l.lower() == close[0]
                )
                if clean != close[0]:  # Only correct if actually different
                    corrections.append(Correction(
                        original=word,
                        corrected=matched_label,
                        layer="label",
                        confidence=difflib.SequenceMatcher(
                            None, clean, close[0]
                        ).ratio(),
                    ))
                    corrected_words.append(
                        word.replace(re.sub(r"[^\w]", "", word), matched_label)
                    )
                    continue

            corrected_words.append(word)

        return " ".join(corrected_words), corrections


# ── Layer 2: Entity Name Resolution ──────────────────────────────────────────


class EntityNameResolver:
    """
    Resolve typos in entity names by querying Neo4j for close matches.

    Uses a lightweight Cypher query with ``toLower()`` and an in-memory
    LRU cache to avoid repeated DB hits.
    """

    def __init__(
        self,
        graph: Any,
        fuzzy_threshold: float = 0.75,
        max_candidates: int = 5,
    ) -> None:
        self._graph = graph
        self._fuzzy_threshold = fuzzy_threshold
        self._max_candidates = max_candidates

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

        resolved_question = question
        for candidate in candidates:
            match = await self._find_closest_match(candidate)
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

    @lru_cache(maxsize=256)
    def _cached_query(self, candidate_lower: str) -> list[dict]:
        """Synchronous Neo4j query (cached to avoid repeated lookups)."""
        cypher = (
            "MATCH (n) "
            "WHERE any(prop IN keys(n) WHERE toLower(toString(n[prop])) "
            "CONTAINS $candidate) "
            "RETURN labels(n)[0] AS label, "
            "head([prop IN keys(n) WHERE toLower(toString(n[prop])) "
            "CONTAINS $candidate | n[prop]]) AS name "
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
            logger.warning("Entity name lookup failed: %s", exc)
            return []

    async def _find_closest_match(self, candidate: str) -> str | None:
        """Find the closest matching entity name in Neo4j."""
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None, self._cached_query, candidate.lower(),
        )

        if not results:
            return None

        # Find the best match by similarity
        best_match: str | None = None
        best_score = 0.0

        for row in results:
            name = str(row.get("name", ""))
            score = difflib.SequenceMatcher(
                None, candidate.lower(), name.lower(),
            ).ratio()
            if score > best_score:
                best_score = score
                best_match = name

        return best_match if best_score >= self._fuzzy_threshold else None


# ── Layer 3: LLM-Assisted Fallback ──────────────────────────────────────────


async def llm_resolve(
    question: str,
    schema: str,
    llm: BaseChatModel,
) -> tuple[str, list[Correction]]:
    """
    Ask the LLM to interpret and correct the user's question given the schema.

    This is the expensive fallback — only used when Layers 1 & 2 fail to
    resolve any entities.
    """
    prompt = (
        "You are an entity-resolution assistant. The user asked a question "
        "that may contain typos, wrong category names, or ambiguous entity "
        "references.\n\n"
        f"Database schema:\n{schema}\n\n"
        f"User question: {question}\n\n"
        "If the question contains any entity names or category references "
        "that don't exactly match the schema, rewrite the question with "
        "corrected names/labels. If everything looks correct, return the "
        "question unchanged.\n\n"
        "Output ONLY the corrected question — no explanations."
    )

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None, lambda: llm.invoke(prompt),
    )
    resolved = str(response.content).strip()

    corrections: list[Correction] = []
    if resolved and resolved != question and len(resolved) > 5:
        corrections.append(Correction(
            original=question,
            corrected=resolved,
            layer="llm",
            confidence=0.7,  # LLM confidence is inherently lower
        ))
        return resolved, corrections

    return question, corrections


# ── Orchestrator ─────────────────────────────────────────────────────────────


async def resolve_entities(
    question: str,
    schema: str,
    graph: Any,
    llm: BaseChatModel,
    *,
    enabled: bool = True,
    fuzzy_threshold: float = 0.75,
    synonym_overrides: str = "",
    max_candidates: int = 5,
) -> ResolutionResult:
    """
    Run the 3-layer entity resolution pipeline.

    Parameters
    ----------
    question:
        The user's natural-language question.
    schema:
        The Neo4j schema string (from SchemaCache).
    graph:
        The Neo4jGraph instance for data-aware lookups.
    llm:
        The LLM for fallback resolution.
    enabled:
        Feature flag — when False, returns immediately with no corrections.
    fuzzy_threshold:
        Minimum similarity score (0.0–1.0) for fuzzy matching.
    synonym_overrides:
        JSON string of custom synonym overrides.
    max_candidates:
        Max candidates to fetch from Neo4j for name lookups.

    Returns
    -------
    ResolutionResult
        Contains the (possibly corrected) question and all corrections made.
    """
    if not enabled:
        return ResolutionResult(
            original_question=question,
            resolved_question=question,
        )

    all_corrections: list[Correction] = []
    current_question = question

    # ── Layer 1: Label resolution ────────────────────────────────────────
    try:
        label_resolver = LabelResolver(
            schema=schema,
            synonym_overrides=synonym_overrides,
            fuzzy_threshold=fuzzy_threshold,
        )
        current_question, label_corrections = label_resolver.resolve(
            current_question,
        )
        all_corrections.extend(label_corrections)
        if label_corrections:
            logger.info(
                "Layer 1 (label): %d correction(s) applied: %s",
                len(label_corrections),
                [(c.original, c.corrected) for c in label_corrections],
            )
    except Exception as exc:
        logger.warning("Layer 1 (label) resolution failed: %s", exc)

    # ── Layer 2: Entity name resolution ──────────────────────────────────
    try:
        name_resolver = EntityNameResolver(
            graph=graph,
            fuzzy_threshold=fuzzy_threshold,
            max_candidates=max_candidates,
        )
        current_question, name_corrections = await name_resolver.resolve(
            current_question,
            known_labels=label_resolver._labels if label_resolver else [],
        )
        all_corrections.extend(name_corrections)
        if name_corrections:
            logger.info(
                "Layer 2 (name): %d correction(s) applied: %s",
                len(name_corrections),
                [(c.original, c.corrected) for c in name_corrections],
            )
    except Exception as exc:
        logger.warning("Layer 2 (name) resolution failed: %s", exc)

    # ── Layer 3: LLM fallback (only if no corrections so far) ────────────
    if not all_corrections:
        try:
            current_question, llm_corrections = await llm_resolve(
                current_question, schema, llm,
            )
            all_corrections.extend(llm_corrections)
            if llm_corrections:
                logger.info(
                    "Layer 3 (LLM): correction applied: %r → %r",
                    question, current_question,
                )
        except Exception as exc:
            logger.warning("Layer 3 (LLM) resolution failed: %s", exc)

    result = ResolutionResult(
        original_question=question,
        resolved_question=current_question,
        corrections=all_corrections,
    )

    if result.was_corrected:
        logger.info(
            "Entity resolution: %r → %r (%d correction(s))",
            question, current_question, len(all_corrections),
        )

    return result
