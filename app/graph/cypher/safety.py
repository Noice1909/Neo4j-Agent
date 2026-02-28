"""
Cypher write-operation safety validator.

Two layers of read-only protection exist:
  1. The Neo4j DB user has only READ privilege (database-level).
  2. This module inspects every generated Cypher string *before* it is sent
     to Neo4j — defence-in-depth with zero query execution overhead.

Usage::

    from app.graph.cypher.safety import validate_read_only
    validate_read_only(cypher_string)   # raises ReadOnlyViolationError or passes
"""
from __future__ import annotations

import re

from app.core.exceptions import ReadOnlyViolationError

# Patterns that indicate mutating Cypher clauses.
# Ordered from most to least common to short-circuit early.
_WRITE_PATTERNS: list[str] = [
    r"\bCREATE\b",
    r"\bMERGE\b",
    r"\bSET\b",
    r"\bDELETE\b",
    r"\bDETACH\s+DELETE\b",
    r"\bREMOVE\b",
    r"\bDROP\b",
    r"\bFOREACH\b",
    r"\bLOAD\s+CSV\b",
    r"\bALTER\b",
    r"\bRENAME\b",
    # Subquery transactions (can embed writes)
    r"\bCALL\s*\{[^}]*\}\s*IN\s+TRANSACTIONS\b",
    # Privilege management
    r"\bGRANT\b",
    r"\bREVOKE\b",
    r"\bDENY\b",
    # APOC write procedures — block mutating APOC calls
    r"\bCALL\s+apoc\.(create|merge|refactor|trigger|nodes|rels|do)\b",
    # Admin / dangerous procedures
    r"\bCALL\s+db\.(clear|createIndex|dropIndex|createConstraint|dropConstraint)\b",
    r"\bCALL\s+dbms\.(security|setConfigValue|killQuery)\b",
]

_COMPILED: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE | re.DOTALL) for p in _WRITE_PATTERNS
]


def validate_read_only(cypher: str) -> None:
    """
    Raise :class:`ReadOnlyViolationError` if *cypher* contains any write clause.

    Parameters
    ----------
    cypher:
        The Cypher query string to validate.

    Raises
    ------
    ReadOnlyViolationError
        If a write-operation keyword is found.
    """
    for pattern in _COMPILED:
        if pattern.search(cypher):
            raise ReadOnlyViolationError(cypher)


def is_read_only(cypher: str) -> bool:
    """Return ``True`` if *cypher* contains no write operations."""
    try:
        validate_read_only(cypher)
        return True
    except ReadOnlyViolationError:
        return False
