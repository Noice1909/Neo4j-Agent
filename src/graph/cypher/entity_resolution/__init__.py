"""
Entity resolution sub-package — 3-layer pipeline for query correction.

Re-exports all public symbols for backward compatibility::

    from src.graph.cypher.entity_resolution import resolve_entities
"""
from src.graph.cypher.entity_resolution.label_resolver import LabelResolver
from src.graph.cypher.entity_resolution.models import (
    Correction,
    FULLTEXT_INDEX_NAME,
    FULLTEXT_ID_INDEX_NAME,
    ResolutionResult,
)
from src.graph.cypher.entity_resolution.capabilities import (
    detect_fulltext_index,
)
from src.graph.cypher.entity_resolution.name_resolver import (
    EntityNameResolver,
)
from src.graph.cypher.entity_resolution.orchestrator import resolve_entities

__all__ = [
    "Correction",
    "EntityNameResolver",
    "FULLTEXT_ID_INDEX_NAME",
    "FULLTEXT_INDEX_NAME",
    "LabelResolver",
    "ResolutionResult",
    "detect_fulltext_index",
    "resolve_entities",
]
