"""
Layer 1 — Label / Category Resolution.

Resolves wrong labels and categories using schema-aware fuzzy matching.
Compares words in the user's question against actual Neo4j node labels
and relationship types, using a synonym map and ``difflib`` fuzzy matching.
"""
from __future__ import annotations

import difflib
import re

from src.graph.cypher.entity_resolution.models import (
    Correction,
    _NON_WORD_RE,
)
from src.graph.cypher.synonyms import build_synonym_map


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
        labels: list[str] = []
        for match in re.finditer(r"^\s*(\w+)\s*\{", schema, re.MULTILINE):
            label = match.group(1)
            if label not in ("Node", "Relationship", "The"):
                labels.append(label)
        return list(dict.fromkeys(labels))

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
            replacement = self._resolve_word(word)
            if replacement is not None:
                corrected_words.append(replacement[0])
                corrections.append(replacement[1])
            else:
                corrected_words.append(word)

        return " ".join(corrected_words), corrections

    def _resolve_word(
        self, word: str,
    ) -> tuple[str, Correction] | None:
        """Try to resolve a single word; returns (replacement, correction) or None."""
        raw = _NON_WORD_RE.sub("", word)
        clean = raw.lower()
        if not clean or len(clean) < 2:
            return None

        # Check synonym map first (exact match)
        if clean in self._synonym_map:
            canonical = self._synonym_map[clean]
            if clean not in [label.lower() for label in self._all_types]:
                return (
                    word.replace(raw, canonical),
                    Correction(
                        original=word,
                        corrected=canonical,
                        layer="label",
                        confidence=1.0,
                    ),
                )

        # Fuzzy match against known labels
        close = difflib.get_close_matches(
            clean,
            [label.lower() for label in self._all_types],
            n=1,
            cutoff=self._fuzzy_threshold,
        )
        if close and clean != close[0]:
            matched_label = next(
                label for label in self._all_types if label.lower() == close[0]
            )
            return (
                word.replace(raw, matched_label),
                Correction(
                    original=word,
                    corrected=matched_label,
                    layer="label",
                    confidence=difflib.SequenceMatcher(
                        None, clean, close[0],
                    ).ratio(),
                ),
            )

        return None
