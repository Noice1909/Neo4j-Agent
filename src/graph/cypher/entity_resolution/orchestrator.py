"""
Layer 3 + pipeline orchestrator for entity resolution.

``llm_resolve()`` is the LLM-assisted fallback (Layer 3).
``resolve_entities()`` is the main entry point that runs all three layers.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.graph.cypher.entity_resolution.label_resolver import LabelResolver
from src.graph.cypher.entity_resolution.models import (
    Correction,
    FULLTEXT_INDEX_NAME,
    FULLTEXT_ID_INDEX_NAME,
    ResolutionResult,
)
from src.graph.cypher.entity_resolution.name_resolver import EntityNameResolver
from src.core.tracing import trace_event

logger = logging.getLogger(__name__)


# ── Layer 3: LLM-Assisted Fallback ──────────────────────────────────────────


async def llm_resolve(
    question: str,
    schema: str,
    llm: BaseChatModel,
    topology_section: str = "",
) -> tuple[str, list[Correction]]:
    """
    Ask the LLM to interpret and correct the user's question given the schema.

    This is the expensive fallback — only used when Layers 1 & 2 fail to
    resolve any entities.
    """
    parts = [
        "You are an entity-resolution assistant. The user asked a question "
        "that may contain typos, wrong category names, or ambiguous entity "
        "references.",
        f"\nDatabase schema:\n{schema}",
    ]
    if topology_section:
        parts.append(f"\n{topology_section}")
    parts += [
        f"\nUser question: {question}",
        "\nIf the question contains any entity names or category references "
        "that don't exactly match the schema, rewrite the question with "
        "corrected names/labels. If everything looks correct, return the "
        "question unchanged.",
        "\nOutput ONLY the corrected question — no explanations.",
    ]
    prompt = "\n".join(parts)

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None, lambda: llm.invoke(prompt),
    )
    resolved = str(response.content).strip()

    corrections: list[Correction] = []
    if resolved and resolved != question and len(resolved) > 5:
        corrections.append(Correction(
            original=question,
            corrected=resolved,
            layer="llm",
            confidence=0.7,
        ))
        return resolved, corrections

    return question, corrections


def _log_layer_result(
    stage: str, layer_label: str,
    corrections: list[Correction], skip_msg: str,
) -> None:
    """Trace + log the outcome of a resolution layer."""
    if corrections:
        logger.info("%s: %d correction(s)", layer_label, len(corrections))
        trace_event(stage, "ok", f"{len(corrections)} correction(s)")
    else:
        trace_event(stage, "skip", skip_msg)


async def _run_llm_fallback(
    question: str,
    current_question: str,
    schema: str,
    llm: BaseChatModel,
    topology_section: str = "",
) -> tuple[str, list[Correction]]:
    """Run Layer 3 LLM entity resolution with error handling and tracing."""
    try:
        current_question, llm_corrections = await llm_resolve(
            current_question, schema, llm, topology_section=topology_section,
        )
        if llm_corrections:
            logger.info(
                "Layer 3 (LLM): correction applied: %r → %r",
                question, current_question,
            )
            trace_event(
                "ENTITY_RES_L3", "ok",
                f"{question[:40]} → {current_question[:40]}",
            )
        else:
            trace_event("ENTITY_RES_L3", "skip", "LLM found no corrections")
        return current_question, llm_corrections
    except Exception as exc:
        logger.warning("Layer 3 (LLM) resolution failed: %s", exc)
        trace_event("ENTITY_RES_L3", "fail", str(exc)[:80])
        return current_question, []


# ── Orchestrator ─────────────────────────────────────────────────────────────


async def resolve_entities(
    question: str,
    schema: str,
    graph: Any,
    llm: Any,
    *,
    enabled: bool = True,
    fuzzy_threshold: float = 0.75,
    synonym_overrides: str = "",
    max_candidates: int = 5,
    fulltext_index_name: str = FULLTEXT_INDEX_NAME,
    id_index_name: str = FULLTEXT_ID_INDEX_NAME,
    display_properties: list[str] | None = None,
    topology_section: str = "",
) -> ResolutionResult:
    """
    Run the 3-layer entity resolution pipeline.

    Parameters
    ----------
    question:
        The user's natural-language question.
    schema:
        The Neo4j schema string (from SchemaCache).
    graph:
        The Neo4jGraph instance for data-aware lookups.
    llm:
        The LLM for fallback resolution.
    enabled:
        Feature flag — when False, returns immediately with no corrections.
    fuzzy_threshold:
        Minimum similarity score (0.0–1.0) for fuzzy matching.
    synonym_overrides:
        JSON string of custom synonym overrides.
    max_candidates:
        Max candidates to fetch from Neo4j for name lookups.
    """
    if not enabled:
        return ResolutionResult(
            original_question=question,
            resolved_question=question,
        )

    all_corrections: list[Correction] = []
    current_question = question
    label_resolver: LabelResolver | None = None

    # ── Layer 1: Label resolution ────────────────────────────────────────
    try:
        label_resolver = LabelResolver(
            schema=schema,
            synonym_overrides=synonym_overrides,
            fuzzy_threshold=fuzzy_threshold,
        )
        current_question, label_corrections = label_resolver.resolve(
            current_question,
        )
        all_corrections.extend(label_corrections)
        _log_layer_result("ENTITY_RES_L1", "Layer 1 (label)", label_corrections, "No label corrections")
    except Exception as exc:
        logger.warning("Layer 1 (label) resolution failed: %s", exc)
        trace_event("ENTITY_RES_L1", "fail", str(exc)[:80])

    # ── Layer 2: Entity name resolution ──────────────────────────────────
    try:
        name_resolver = EntityNameResolver(
            graph=graph,
            schema=schema,
            fuzzy_threshold=fuzzy_threshold,
            max_candidates=max_candidates,
            fulltext_index_name=fulltext_index_name,
            id_index_name=id_index_name,
            display_properties=display_properties,
        )
        current_question, name_corrections = await name_resolver.resolve(
            current_question,
            known_labels=label_resolver._labels if label_resolver else [],
        )
        all_corrections.extend(name_corrections)
        _log_layer_result("ENTITY_RES_L2", "Layer 2 (name)", name_corrections, "No name corrections")
    except Exception as exc:
        logger.warning("Layer 2 (name) resolution failed: %s", exc)
        trace_event("ENTITY_RES_L2", "fail", str(exc)[:80])

    # ── Layer 3: LLM fallback (only if no corrections so far) ────────────
    if not all_corrections:
        current_question, llm_corr = await _run_llm_fallback(
            question, current_question, schema, llm,
            topology_section=topology_section,
        )
        all_corrections.extend(llm_corr)
    else:
        trace_event("ENTITY_RES_L3", "skip", "Skipped (L1/L2 already corrected)")

    result = ResolutionResult(
        original_question=question,
        resolved_question=current_question,
        corrections=all_corrections,
    )

    if result.was_corrected:
        logger.info(
            "Entity resolution: %r → %r (%d correction(s))",
            question, current_question, len(all_corrections),
        )

    return result
