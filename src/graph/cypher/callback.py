"""
CypherSafetyCallback — LangChain callback interceptor.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from src.graph.cypher.safety import validate_read_only
from src.graph.cypher.validation import pre_validate_cypher, validate_relationship_types
from src.core.tracing import trace_event

logger = logging.getLogger(__name__)


class CypherSafetyCallback(BaseCallbackHandler):
    """Intercepts generated Cypher for read-only + syntax pre-validation."""

    def __init__(self, valid_rel_types: set[str] | None = None) -> None:
        super().__init__()
        self.last_generated_cypher: str | None = None
        self._valid_rel_types = valid_rel_types

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Inspect chain outputs for generated Cypher and validate."""
        if isinstance(outputs, dict):
            query = outputs.get("query")
            if query and isinstance(query, str):
                self.last_generated_cypher = query
                logger.debug("Pre-execution Cypher check: %s", query[:200])
                trace_event("CYPHER_GENERATED", "info", query)

                issues = pre_validate_cypher(query)
                if issues:
                    trace_event("CYPHER_VALIDATED", "fail", f"Pre-validation: {', '.join(issues)}")
                    raise ValueError(
                        f"Cypher pre-validation failed ({', '.join(issues)}): "
                        f"{query[:200]}"
                    )

                validate_read_only(query)

                if self._valid_rel_types:
                    invalid = validate_relationship_types(query, self._valid_rel_types)
                    if invalid:
                        types_str = ", ".join(invalid)
                        valid_str = ", ".join(sorted(self._valid_rel_types))
                        trace_event("CYPHER_VALIDATED", "fail", f"Unknown rel type(s): {types_str}")
                        raise ValueError(
                            f"Unknown relationship type(s): {types_str}. "
                            f"Valid types: {valid_str}"
                        )

                trace_event("CYPHER_VALIDATED", "ok", "Read-only ✓  Syntax ✓  Types ✓")
