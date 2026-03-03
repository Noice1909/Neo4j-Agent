"""Tests for src.core.masking — sensitive log masking."""
from __future__ import annotations

import sys
from io import StringIO
from unittest.mock import MagicMock

import pytest

from src.core.masking import (
    mask_value,
    build_sensitive_map,
    init_masking,
    mask_sensitive_processor,
    MaskingStream,
    install_stream_masking,
    _replace_secrets,
)


# ── mask_value ──────────────────────────────────────────────────────────────


class TestMaskValue:
    def test_short_string(self):
        assert mask_value("ab") == "***"
        assert mask_value("abc") == "***"

    def test_medium_string(self):
        assert mask_value("abcd") == "a***d"
        assert mask_value("abcdef") == "a***f"

    def test_long_string(self):
        assert mask_value("abcdefg") == "ab****fg"
        assert mask_value("mySecret123") == "my****23"

    def test_uri(self):
        assert mask_value("redis://:pass@host:6379") == "re****79"

    def test_empty(self):
        assert mask_value("") == "***"


# ── build_sensitive_map ─────────────────────────────────────────────────────


class TestBuildSensitiveMap:
    def _make_settings(self, **overrides):
        defaults = {
            "neo4j_uri": "neo4j+s://8bfe3016.databases.neo4j.io",
            "neo4j_user": "neo4j",
            "neo4j_password": "superSecretPassword",
            "neo4j_database": "8bfe3016",
            "ollama_base_url": "http://localhost:11434",
            "ollama_model": "qwen2.5:latest",
            "ollama_temperature": 0.0,
            "redis_url": "redis://localhost:6379",
            "api_key": "sk-test-1234567890abcdef",
            "app_name": "Neo4j Agent",
            "debug": False,
            "log_level": "INFO",
            "checkpointer_backend": "sqlite",
            "schema_cache_ttl_seconds": 300,
            "entity_resolution_enabled": True,
            "query_dedup_enabled": True,
        }
        defaults.update(overrides)
        mock = MagicMock()
        mock.model_dump.return_value = defaults
        return mock

    def test_includes_sensitive_strings(self):
        settings = self._make_settings()
        result = build_sensitive_map(settings)
        assert "neo4j+s://8bfe3016.databases.neo4j.io" in result
        assert "superSecretPassword" in result
        assert "sk-test-1234567890abcdef" in result

    def test_excludes_safe_fields(self):
        settings = self._make_settings()
        result = build_sensitive_map(settings)
        assert "Neo4j Agent" not in result

    def test_excludes_short_strings(self):
        settings = self._make_settings()
        result = build_sensitive_map(settings)
        assert "neo4j" not in result

    def test_sorted_longest_first(self):
        settings = self._make_settings()
        result = build_sensitive_map(settings)
        keys = list(result.keys())
        lengths = [len(k) for k in keys]
        assert lengths == sorted(lengths, reverse=True)

    def test_empty_password_not_included(self):
        settings = self._make_settings(neo4j_password="", api_key="")
        result = build_sensitive_map(settings)
        assert "" not in result


# ── MaskingStream (primary mechanism) ───────────────────────────────────────


class TestMaskingStream:
    def setup_method(self):
        mock = MagicMock()
        mock.model_dump.return_value = {
            "neo4j_uri": "neo4j+s://mydb.neo4j.io",
            "neo4j_password": "superSecret",
            "api_key": "sk-abcdef123456",
        }
        init_masking(mock)

    def test_masks_write(self):
        buf = StringIO()
        stream = MaskingStream(buf)
        stream.write("Connecting to neo4j+s://mydb.neo4j.io\n")
        output = buf.getvalue()
        assert "neo4j+s://mydb.neo4j.io" not in output
        assert "****" in output

    def test_leaves_safe_text(self):
        buf = StringIO()
        stream = MaskingStream(buf)
        stream.write("Starting server on port 8001\n")
        assert buf.getvalue() == "Starting server on port 8001\n"

    def test_masks_password(self):
        buf = StringIO()
        stream = MaskingStream(buf)
        stream.write("Auth failed for superSecret\n")
        assert "superSecret" not in buf.getvalue()

    def test_masks_multiple_values(self):
        buf = StringIO()
        stream = MaskingStream(buf)
        stream.write("URI=neo4j+s://mydb.neo4j.io key=sk-abcdef123456\n")
        output = buf.getvalue()
        assert "neo4j+s://mydb.neo4j.io" not in output
        assert "sk-abcdef123456" not in output

    def test_flush_proxied(self):
        buf = StringIO()
        stream = MaskingStream(buf)
        stream.write("test")
        stream.flush()  # Should not raise

    def test_install_wraps_streams(self):
        """install_stream_masking() wraps sys.stderr and sys.stdout."""
        old_stderr = sys.stderr
        old_stdout = sys.stdout
        try:
            install_stream_masking()
            assert isinstance(sys.stderr, MaskingStream)
            assert isinstance(sys.stdout, MaskingStream)
        finally:
            sys.stderr = old_stderr
            sys.stdout = old_stdout

    def test_install_idempotent(self):
        """Calling install twice doesn't double-wrap."""
        old_stderr = sys.stderr
        old_stdout = sys.stdout
        try:
            install_stream_masking()
            first_stderr = sys.stderr
            install_stream_masking()
            assert sys.stderr is first_stderr  # Same wrapper, not nested
        finally:
            sys.stderr = old_stderr
            sys.stdout = old_stdout


# ── structlog processor (belt-and-suspenders) ───────────────────────────────


class TestMaskSensitiveProcessor:
    def setup_method(self):
        mock = MagicMock()
        mock.model_dump.return_value = {
            "neo4j_uri": "neo4j+s://mydb.neo4j.io",
            "neo4j_password": "superSecret",
            "api_key": "sk-abcdef123456",
        }
        init_masking(mock)

    def test_masks_event_message(self):
        event_dict = {"event": "Connecting to neo4j+s://mydb.neo4j.io"}
        result = mask_sensitive_processor(None, "info", event_dict)
        assert "neo4j+s://mydb.neo4j.io" not in result["event"]
        assert "****" in result["event"]

    def test_leaves_non_sensitive(self):
        event_dict = {"event": "Starting server on port 8001"}
        result = mask_sensitive_processor(None, "info", event_dict)
        assert result["event"] == "Starting server on port 8001"
