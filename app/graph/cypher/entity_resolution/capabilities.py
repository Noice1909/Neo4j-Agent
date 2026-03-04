"""
Neo4j capability detection for entity resolution Layer 2.

Read-only probes that detect whether a full-text index or APOC text
procedures are available.  Called once during ``EntityNameResolver``
initialisation.
"""
from __future__ import annotations

import logging
from typing import Any

from app.graph.cypher.entity_resolution.models import FULLTEXT_INDEX_NAME

logger = logging.getLogger(__name__)


def detect_fulltext_index(
    graph: Any, index_name: str = FULLTEXT_INDEX_NAME,
) -> bool:
    """
    Check whether a full-text index exists (read-only probe).

    If the index is missing, logs the CREATE statement the admin should run.
    """
    try:
        existing = graph.query(
            "SHOW FULLTEXT INDEXES YIELD name WHERE name = $name RETURN name",
            params={"name": index_name},
        )
        if existing:
            logger.info(
                "Full-text index '%s' detected — Layer 2a enabled.", index_name,
            )
            return True
    except Exception:
        # SHOW INDEXES may fail on older Neo4j versions — not fatal
        pass

    logger.warning(
        "Full-text index '%s' NOT found. Layer 2 will use APOC fallback "
        "(slower on large databases). Ask your Neo4j admin to run:\n"
        "  CREATE FULLTEXT INDEX %s IF NOT EXISTS\n"
        "  FOR (n) ON EACH [n.name, n.title]",
        index_name, index_name,
    )
    return False


def check_apoc_available(graph: Any) -> bool:
    """Check whether APOC text procedures are installed."""
    try:
        graph.query(
            "RETURN apoc.text.levenshteinSimilarity('a', 'b') AS sim",
        )
        return True
    except Exception:
        logger.warning(
            "APOC text functions not available. "
            "Layer 2b/2c (APOC fallback) disabled.",
        )
        return False
