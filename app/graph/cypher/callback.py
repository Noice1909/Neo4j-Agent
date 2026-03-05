"""
CypherSafetyCallback — LangChain callback interceptor.

Intercepts generated Cypher for:
  - Read-only safety validation
  - Strategy #4 syntax pre-validation
  - Capturing the last generated Cypher for the retry loop
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import BaseCallbackHandler

from app.graph.cypher.safety import validate_read_only
from app.graph.cypher.schema_validation import validate_cypher_schema
from app.graph.cypher.validation import pre_validate_cypher, validate_relationship_types
from app.core.tracing import trace_event

if TYPE_CHECKING:
    from app.graph.topology import GraphTopology

logger = logging.getLogger(__name__)


class CypherSafetyCallback(BaseCallbackHandler):
    """Intercepts generated Cypher for read-only + syntax + schema pre-validation.

    Also *captures* the last generated Cypher so the retry loop can
    reference it when building a correction prompt.
    """

    def __init__(
        self,
        valid_rel_types: set[str] | None = None,
        topology: "GraphTopology | None" = None,
    ) -> None:
        super().__init__()
        self.last_generated_cypher: str | None = None
        self._valid_rel_types = valid_rel_types
        self._topology = topology

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Inspect chain outputs for generated Cypher and validate."""
        if not isinstance(outputs, dict):
            return
        query = outputs.get("query")
        if not query or not isinstance(query, str):
            return
        self.last_generated_cypher = query
        logger.debug("Pre-execution Cypher check: %s", query[:200])
        trace_event("CYPHER_GENERATED", "info", query)
        self._validate(query)

    def _validate(self, query: str) -> None:
        """Run all validation checks; raise ValueError on first failure."""
        self._check_syntax(query)
        validate_read_only(query)
        self._check_rel_types(query)
        self._check_schema(query)
        trace_event("CYPHER_VALIDATED", "ok", "Read-only \u2713  Syntax \u2713  Types \u2713  Schema \u2713")

    def _check_syntax(self, query: str) -> None:
        issues = pre_validate_cypher(query)
        if issues:
            trace_event("CYPHER_VALIDATED", "fail", f"Pre-validation: {', '.join(issues)}")
            raise ValueError(
                f"Cypher pre-validation failed ({', '.join(issues)}): {query[:200]}"
            )

    def _check_rel_types(self, query: str) -> None:
        if not self._valid_rel_types:
            return
        invalid = validate_relationship_types(query, self._valid_rel_types)
        if invalid:
            types_str = ", ".join(invalid)
            valid_str = ", ".join(sorted(self._valid_rel_types))
            trace_event("CYPHER_VALIDATED", "fail", f"Unknown rel type(s): {types_str}")
            raise ValueError(
                f"Unknown relationship type(s): {types_str}. Valid types: {valid_str}"
            )

    def _check_schema(self, query: str) -> None:
        if not self._topology:
            return
        schema_errors = validate_cypher_schema(query, self._topology)
        if schema_errors:
            errors_str = "; ".join(schema_errors)
            trace_event("CYPHER_VALIDATED", "fail", f"Schema errors: {errors_str[:120]}")
            raise ValueError(
                "Schema violations detected:\n"
                + "\n".join(f"  - {e}" for e in schema_errors)
            )
