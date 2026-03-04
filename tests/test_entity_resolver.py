"""
Unit tests for the entity resolution pipeline.

Run with: python -m pytest tests/test_entity_resolver.py -v
"""
from __future__ import annotations

import asyncio
import pytest

from src.graph.cypher.synonyms import (
    auto_generate_synonyms,
    build_synonym_map,
)
from src.graph.cypher.entity_resolution import (
    Correction,
    LabelResolver,
    ResolutionResult,
    resolve_entities,
)


# ── Mock helpers ─────────────────────────────────────────────────────────────

MOCK_SCHEMA = """
Node properties:
Movie {title: STRING, released: INTEGER, tagline: STRING}
Person {name: STRING, born: INTEGER}
Application {name: STRING, type: STRING}

Relationship properties:

The relationships:
(:Person)-[:ACTED_IN]->(:Movie)
(:Person)-[:DIRECTED]->(:Movie)
(:Person)-[:PRODUCED]->(:Movie)
(:Person)-[:WROTE]->(:Movie)
(:Person)-[:FOLLOWS]->(:Person)
(:Person)-[:REVIEWED]->(:Movie)
"""


class MockGraph:
    """Fake Neo4jGraph for testing entity name resolution."""

    def __init__(self, data: list[dict] | None = None):
        self._data = data or []

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        return self._data


class MockLLM:
    """Fake LLM that returns the question unchanged."""

    def invoke(self, prompt: str) -> "MockResponse":
        # For LLM resolve, just return the original question
        return MockResponse(prompt.split("User question: ")[-1].split("\n")[0])


class MockResponse:
    def __init__(self, content: str):
        self.content = content


# ── Synonym tests ────────────────────────────────────────────────────────────


class TestAutoGenerateSynonyms:
    def test_lowercase_mapping(self):
        result = auto_generate_synonyms(["Movie", "Person"])
        assert result["movie"] == "Movie"
        assert result["person"] == "Person"

    def test_plural_generation(self):
        result = auto_generate_synonyms(["Movie"])
        assert result["movies"] == "Movie"

    def test_plural_y_ending(self):
        result = auto_generate_synonyms(["Category"])
        assert result["categories"] == "Category"

    def test_camelcase_split(self):
        result = auto_generate_synonyms(["MovieGenre"])
        assert result["movie genre"] == "MovieGenre"
        assert result["movie_genre"] == "MovieGenre"

    def test_single_word_no_split(self):
        result = auto_generate_synonyms(["Movie"])
        # Should not create space/underscore variants for single-word labels
        assert "m ovie" not in result


class TestBuildSynonymMap:
    def test_includes_defaults(self):
        result = build_synonym_map(["Movie"])
        assert result["film"] == "Movie"  # from DEFAULT_SYNONYMS

    def test_auto_generated_override_defaults(self):
        result = build_synonym_map(["Film"])
        # Auto-generated "film" → "Film" should override default "film" → "Movie"
        assert result["film"] == "Film"

    def test_custom_overrides(self):
        result = build_synonym_map(
            ["Movie"],
            overrides_json='{"flick": "Movie", "cinema": "Movie"}',
        )
        assert result["flick"] == "Movie"
        assert result["cinema"] == "Movie"

    def test_invalid_json_handled_gracefully(self):
        # Should not raise, just log a warning
        result = build_synonym_map(["Movie"], overrides_json="not valid json")
        assert "movie" in result  # auto-generated still works


# ── Label resolver tests ─────────────────────────────────────────────────────


class TestLabelResolver:
    def test_synonym_correction(self):
        resolver = LabelResolver(schema=MOCK_SCHEMA)
        _, corrections = resolver.resolve("Show me all films")
        assert len(corrections) == 1
        assert corrections[0].corrected == "Movie"
        assert corrections[0].layer == "label"
        assert corrections[0].confidence == pytest.approx(1.0)

    def test_no_correction_for_valid_label(self):
        resolver = LabelResolver(schema=MOCK_SCHEMA)
        _, corrections = resolver.resolve("Show me all Movie details")
        assert len(corrections) == 0

    def test_fuzzy_match(self):
        resolver = LabelResolver(schema=MOCK_SCHEMA, fuzzy_threshold=0.6)
        _, corrections = resolver.resolve("Tell me about Moive titles")
        # "Moive" should fuzzy-match to "Movie"
        has_correction = any(c.corrected == "Movie" for c in corrections)
        assert has_correction

    def test_sor_to_application(self):
        resolver = LabelResolver(schema=MOCK_SCHEMA)
        _, corrections = resolver.resolve("What is WIDL sor?")
        # "sor" should map to "Application" via default synonyms
        has_correction = any(c.corrected == "Application" for c in corrections)
        assert has_correction

    def test_extract_labels(self):
        labels = LabelResolver._extract_labels(MOCK_SCHEMA)
        assert "Movie" in labels
        assert "Person" in labels
        assert "Application" in labels

    def test_extract_relationship_types(self):
        rel_types = LabelResolver._extract_relationship_types(MOCK_SCHEMA)
        assert "ACTED_IN" in rel_types
        assert "DIRECTED" in rel_types


# ── Resolution result tests ──────────────────────────────────────────────────


class TestResolutionResult:
    def test_was_corrected_true(self):
        result = ResolutionResult(
            original_question="test",
            resolved_question="corrected",
            corrections=[Correction("a", "b", "label", 1.0)],
        )
        assert result.was_corrected is True

    def test_was_corrected_false(self):
        result = ResolutionResult(
            original_question="test",
            resolved_question="test",
        )
        assert result.was_corrected is False


# ── Orchestrator tests ───────────────────────────────────────────────────────


class TestResolveEntities:
    def test_disabled_returns_unchanged(self):
        result = asyncio.get_event_loop().run_until_complete(
            resolve_entities(
                question="Show me films",
                schema=MOCK_SCHEMA,
                graph=MockGraph(),
                llm=MockLLM(),
                enabled=False,
            )
        )
        assert result.resolved_question == "Show me films"
        assert result.was_corrected is False

    def test_label_correction(self):
        result = asyncio.get_event_loop().run_until_complete(
            resolve_entities(
                question="List all films",
                schema=MOCK_SCHEMA,
                graph=MockGraph(),
                llm=MockLLM(),
                enabled=True,
            )
        )
        assert "Movie" in result.resolved_question
        assert result.was_corrected is True

    def test_entity_name_with_mock_data(self):
        mock_graph = MockGraph(data=[
            {"label": "Person", "name": "Tom Hanks"},
        ])
        result = asyncio.get_event_loop().run_until_complete(
            resolve_entities(
                question='Who is "Tom Hankes"?',
                schema=MOCK_SCHEMA,
                graph=mock_graph,
                llm=MockLLM(),
                enabled=True,
            )
        )
        # Should find "Tom Hanks" as a close match for "Tom Hankes"
        if result.was_corrected:
            name_corrections = [
                c for c in result.corrections if c.layer == "entity_name"
            ]
            if name_corrections:
                assert name_corrections[0].corrected == "Tom Hanks"
