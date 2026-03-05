"""
Entity resolution data models and shared constants.

Contains the ``Correction`` and ``ResolutionResult`` data classes used
across all three layers of the resolution pipeline, plus shared regex
constants.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

# Regex to strip non-word characters from a token.
_NON_WORD_RE = re.compile(r"[^\w]")

# Regex for escaping special characters in Lucene queries.
_LUCENE_SPECIAL_RE = re.compile(r'([+\-&|!(){}[\]^"~*?:\\\/ ])')

# Full-text index for name lookups (admin-created: globalNameIndex).
FULLTEXT_INDEX_NAME = "globalNameIndex"
# Full-text index for ID lookups (admin-created: globalIndex).
FULLTEXT_ID_INDEX_NAME = "globalIndex"


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
