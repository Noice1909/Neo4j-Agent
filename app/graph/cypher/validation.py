"""
Cypher syntax pre-validation — fast regex heuristics (Strategy #4).

Catches the most common LLM mistakes before hitting Neo4j,
saving a database round-trip.
"""
from __future__ import annotations

import re


def pre_validate_cypher(cypher: str) -> list[str]:
    """
    Return a list of likely syntax issues (empty list = looks OK).

    This is a fast, cheap check before hitting Neo4j — catches the most
    common LLM mistakes without a full parser.
    """
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


def validate_relationship_types(cypher: str, valid_types: set[str]) -> list[str]:
    """
    Return relationship types used in *cypher* that are not in *valid_types*.

    Returns an empty list when all types are valid.
    """
    used = set(re.findall(r"\[:(\w+)\]", cypher))
    return sorted(used - valid_types)
