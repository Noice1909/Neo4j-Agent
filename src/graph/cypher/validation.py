"""
Cypher syntax pre-validation — fast regex heuristics (Strategy #4).
"""
from __future__ import annotations

import re


def pre_validate_cypher(cypher: str) -> list[str]:
    """Return a list of likely syntax issues (empty list = looks OK)."""
    issues: list[str] = []
    stripped = cypher.strip()
    upper = stripped.upper()

    if len(stripped) < 10:
        issues.append("Query suspiciously short")

    if "RETURN" not in upper and not upper.startswith("CALL"):
        issues.append("Missing RETURN clause")

    if cypher.count("(") != cypher.count(")"):
        issues.append("Unbalanced parentheses")
    if cypher.count("[") != cypher.count("]"):
        issues.append("Unbalanced square brackets")
    if cypher.count("{") != cypher.count("}"):
        issues.append("Unbalanced curly braces")

    if re.search(r"\bSELECT\s", upper):
        issues.append("SQL SELECT detected — expected Cypher")

    return issues
