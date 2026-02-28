"""
CypherSafetyCallback — LangChain callback interceptor.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from src.graph.cypher.safety import validate_read_only
from src.graph.cypher.validation import pre_validate_cypher

logger = logging.getLogger(__name__)


class CypherSafetyCallback(BaseCallbackHandler):
    """Intercepts generated Cypher for read-only + syntax pre-validation."""

    def __init__(self) -> None:
        super().__init__()
        self.last_generated_cypher: str | None = None

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Inspect chain outputs for generated Cypher and validate."""
        if isinstance(outputs, dict):
            query = outputs.get("query")
            if query and isinstance(query, str):
                self.last_generated_cypher = query
                logger.debug("Pre-execution Cypher check: %s", query[:200])

                issues = pre_validate_cypher(query)
                if issues:
                    raise ValueError(
                        f"Cypher pre-validation failed ({', '.join(issues)}): "
                        f"{query[:200]}"
                    )

                validate_read_only(query)
