"""
Agent 5: Cypher Generation

Generates a Cypher query using few-shot examples and the filtered topology.
Does NOT execute the query — only generates it.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from src.agent.state import PipelineState
from src.graph.cypher.dynamic_examples import generate_few_shot_examples
from src.graph.cypher.prompts import build_cypher_prompt, build_topology_section
from src.core.tracing import trace_event

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


def build_cypher_generation_node(llm: "BaseChatModel"):
    """
    Build the Cypher generation agent node.

    Parameters
    ----------
    llm:
        Language model for Cypher generation.

    Returns
    -------
    async function
        Node function for LangGraph: `(PipelineState) -> dict`
    """
    async def cypher_generation_node(state: PipelineState) -> dict:
        """
        Generate a Cypher query from the resolved question and filtered topology.

        Reads
        -----
        - state["entity_resolved_question"]: Question after entity resolution
        - state["filtered_topology_json"]: Filtered topology
        - state["correction_guidance"]: Hints from retry decision (if retry_count > 0)
        - state["retry_count"]: Current retry attempt

        Returns
        -------
        dict
            State update with:
            - "generated_cypher": The Cypher query string
            - "cypher_prompt_text": The full prompt (for debugging/retry)
            - "topology_section": Rendered topology text
        """
        question = state.get("entity_resolved_question", state.get("user_question", ""))
        retry_count = state.get("retry_count", 0)
        correction_guidance = state.get("correction_guidance", "")

        trace_event(
            "CYPHER_GEN_START",
            "info",
            f"Attempt {retry_count}: {question[:80]}",
        )

        try:
            # Deserialize filtered topology
            from src.graph.schema_cache import _topology_from_json
            topology_json = state.get("filtered_topology_json")
            if not topology_json:
                raise ValueError("filtered_topology_json missing from state")

            filtered_topology = _topology_from_json(topology_json)

            # Generate few-shot examples
            few_shot = generate_few_shot_examples(
                filtered_topology,
                question=question,
            )
            logger.debug(
                "Generated %d few-shot examples",
                len(few_shot),
            )

            # Build Cypher prompt
            cypher_prompt = build_cypher_prompt(filtered_topology, few_shot)

            # Build topology section (for retry context)
            # Get full topology valid_rel_types from state (or use filtered)
            from src.graph.schema_cache import _topology_from_json
            full_topology_json = state.get("full_topology_json")
            if full_topology_json:
                full_topology = _topology_from_json(full_topology_json)
                valid_rel_types = full_topology.valid_rel_types
            else:
                valid_rel_types = filtered_topology.valid_rel_types

            topology_section = build_topology_section(
                filtered_topology,
                full_valid_types=valid_rel_types,
            )

            # Add schema context from semantic layer (property/relationship mappings)
            schema_context = state.get("schema_context", "")

            # Build final prompt text
            prompt_text = cypher_prompt

            if retry_count > 0 and correction_guidance:
                prompt_text += f"\n\n{correction_guidance}"

            if schema_context:
                prompt_text += f"\n\n{schema_context}"

            prompt_text += f"\n\nQuestion: {question}\nCypher:"

            # Invoke LLM to generate Cypher
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: llm.invoke(prompt_text),
            )

            generated_cypher = str(response.content).strip()

            # Remove markdown code fences if present
            from src.graph.cypher.retry import strip_code_fences
            generated_cypher = strip_code_fences(generated_cypher)

            trace_event(
                "CYPHER_GENERATED",
                "ok",
                f"Attempt {retry_count}: {generated_cypher[:100]}",
            )
            logger.info(
                "Cypher generated (attempt %d): %s",
                retry_count,
                generated_cypher,
            )

            return {
                "generated_cypher": generated_cypher,
                "cypher_prompt_text": prompt_text,
                "topology_section": topology_section,
            }

        except Exception as exc:
            logger.error(
                "Cypher generation failed (attempt %d): %s",
                retry_count,
                exc,
                exc_info=True,
            )
            trace_event("CYPHER_GEN_FAIL", "fail", str(exc)[:120])
            # Return empty Cypher — validation agent will catch this
            return {
                "generated_cypher": "",
                "cypher_prompt_text": "",
                "topology_section": "",
            }

    return cypher_generation_node
