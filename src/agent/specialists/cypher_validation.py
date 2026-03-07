"""
Agent 6: Cypher Validation

Validates generated Cypher for:
- Read-only compliance (no WRITE, DELETE, CREATE, MERGE, SET, REMOVE)
- Basic syntax (no empty query, valid structure)
- Relationship types (must exist in schema)
- Schema correctness (labels, properties, relationships match topology)

Deterministic agent — no LLM calls.
"""
from __future__ import annotations

import logging

from src.agent.state import PipelineState
from src.graph.cypher.safety import validate_read_only
from src.graph.cypher.validation import pre_validate_cypher, validate_relationship_types
from src.graph.cypher.schema_validation import validate_cypher_schema
from src.core.exceptions import ReadOnlyViolationError
from src.core.tracing import trace_event

logger = logging.getLogger(__name__)


def build_cypher_validation_node():
    """
    Build the Cypher validation agent node.

    Returns
    -------
    async function
        Node function for LangGraph: `(PipelineState) -> dict`
    """
    async def cypher_validation_node(state: PipelineState) -> dict:
        """
        Validate the generated Cypher query.

        Reads
        -----
        - state["generated_cypher"]: The Cypher query to validate
        - state["filtered_topology_json"]: Filtered topology for schema validation
        - state["full_topology_json"]: Full topology for relationship types

        Returns
        -------
        dict
            State update with:
            - "validation_passed": True if all checks pass
            - "validation_errors": List of error messages (empty if passed)

        Raises
        ------
        ReadOnlyViolationError
            If the Cypher contains write operations (security boundary).
        """
        cypher = state.get("generated_cypher", "")

        trace_event("CYPHER_VALIDATION_START", "info", f"Cypher: {cypher[:100]}")

        errors = []

        try:
            # ── Validation 1: Read-only check ────────────────────────────
            try:
                validate_read_only(cypher)
            except ReadOnlyViolationError as exc:
                # Re-raise immediately — security boundary
                trace_event("CYPHER_VALIDATION_READONLY_FAIL", "fail", str(exc))
                logger.error("Read-only violation detected: %s", exc)
                raise

            # ── Validation 2: Pre-validation (syntax, structure) ────────
            syntax_issues = pre_validate_cypher(cypher)
            if syntax_issues:
                errors.extend(syntax_issues)
                trace_event(
                    "CYPHER_VALIDATION_SYNTAX_FAIL",
                    "fail",
                    f"{len(syntax_issues)} issue(s)",
                )
                logger.warning(
                    "Pre-validation failed: %s",
                    ", ".join(syntax_issues),
                )

            # ── Validation 3: Relationship types ────────────────────────
            from src.graph.schema_cache import _topology_from_json
            full_topology_json = state.get("full_topology_json")
            if full_topology_json:
                full_topology = _topology_from_json(full_topology_json)
                valid_rel_types = full_topology.valid_rel_types

                invalid_rels = validate_relationship_types(cypher, valid_rel_types)
                if invalid_rels:
                    errors.append(
                        f"Unknown relationship type(s): {', '.join(invalid_rels)}. "
                        f"Valid types: {', '.join(sorted(valid_rel_types))}"
                    )
                    trace_event(
                        "CYPHER_VALIDATION_RELTYPES_FAIL",
                        "fail",
                        f"{len(invalid_rels)} unknown type(s)",
                    )
                    logger.warning(
                        "Invalid relationship types: %s",
                        ", ".join(invalid_rels),
                    )

            # ── Validation 4: Schema validation (labels, properties) ───
            filtered_topology_json = state.get("filtered_topology_json")
            if filtered_topology_json:
                filtered_topology = _topology_from_json(filtered_topology_json)

                schema_errors = validate_cypher_schema(cypher, filtered_topology)
                if schema_errors:
                    errors.extend(schema_errors)
                    trace_event(
                        "CYPHER_VALIDATION_SCHEMA_FAIL",
                        "fail",
                        f"{len(schema_errors)} schema issue(s)",
                    )
                    logger.warning(
                        "Schema validation failed: %s",
                        "; ".join(schema_errors[:3]),
                    )

            # ── Final result ─────────────────────────────────────────────
            if errors:
                trace_event(
                    "CYPHER_VALIDATION_FAIL",
                    "fail",
                    f"{len(errors)} error(s) total",
                )
                logger.warning(
                    "Cypher validation failed with %d error(s)",
                    len(errors),
                )
                return {
                    "validation_passed": False,
                    "validation_errors": errors,
                }
            else:
                trace_event("CYPHER_VALIDATION_PASS", "ok", "All checks passed")
                logger.info("Cypher validation passed")
                return {
                    "validation_passed": True,
                    "validation_errors": [],
                }

        except ReadOnlyViolationError:
            # Re-raise security violations
            raise
        except Exception as exc:
            logger.error(
                "Cypher validation encountered unexpected error: %s",
                exc,
                exc_info=True,
            )
            trace_event("CYPHER_VALIDATION_ERROR", "fail", str(exc)[:120])
            # Treat as validation failure
            return {
                "validation_passed": False,
                "validation_errors": [f"Validation error: {exc}"],
            }

    return cypher_validation_node
