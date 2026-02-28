"""
Cypher write-operation safety validator.
"""
from __future__ import annotations

import re

from src.core.exceptions import ReadOnlyViolationError

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
    r"\bCALL\s*\{[^}]*\}\s*IN\s+TRANSACTIONS\b",
    r"\bGRANT\b",
    r"\bREVOKE\b",
    r"\bDENY\b",
    r"\bCALL\s+apoc\.(create|merge|refactor|trigger|nodes|rels|do)\b",
    r"\bCALL\s+db\.(clear|createIndex|dropIndex|createConstraint|dropConstraint)\b",
    r"\bCALL\s+dbms\.(security|setConfigValue|killQuery)\b",
]

_COMPILED: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE | re.DOTALL) for p in _WRITE_PATTERNS
]


def validate_read_only(cypher: str) -> None:
    """Raise :class:`ReadOnlyViolationError` if *cypher* contains any write clause."""
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
