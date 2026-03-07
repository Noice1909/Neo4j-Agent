"""Tests for conversation trimming utility."""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.trimming import trim_conversation, _estimate_tokens


# ── Token estimator tests ─────────────────────────────────────────────────────


class TestEstimateTokens:
    def test_short_message(self):
        msg = HumanMessage(content="Hello")
        tokens = _estimate_tokens(msg)
        # 4 overhead + len("Hello")//4 = 4 + 1 = 5
        assert tokens == 5

    def test_empty_message(self):
        msg = HumanMessage(content="")
        assert _estimate_tokens(msg) == 4  # Just overhead

    def test_long_message(self):
        msg = HumanMessage(content="x" * 400)
        tokens = _estimate_tokens(msg)
        assert tokens == 4 + 100  # overhead + 400//4

    def test_non_string_content(self):
        """Content that is a list (multimodal) should still work."""
        msg = HumanMessage(content=[{"type": "text", "text": "hello"}])
        tokens = _estimate_tokens(msg)
        assert tokens > 4  # overhead + stringified content


# ── Trimming logic tests ─────────────────────────────────────────────────────


class TestTrimConversation:
    @staticmethod
    def _make_history(n_pairs: int, content_size: int = 200) -> list:
        """Create a conversation with system + n human/ai pairs."""
        msgs: list = [SystemMessage(content="You are a helpful assistant.")]
        for i in range(n_pairs):
            msgs.append(HumanMessage(content=f"Question {i}: " + "x" * content_size))
            msgs.append(AIMessage(content=f"Answer {i}: " + "y" * content_size))
        return msgs

    def test_no_trimming_when_within_budget(self):
        msgs = self._make_history(3)
        result = trim_conversation(msgs, max_tokens=100_000)
        assert len(result) == len(msgs)

    def test_system_prompt_always_preserved(self):
        msgs = self._make_history(50)
        result = trim_conversation(msgs, max_tokens=500)
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "You are a helpful assistant."

    def test_first_human_message_preserved(self):
        msgs = self._make_history(50)
        result = trim_conversation(msgs, max_tokens=500)
        # Find human messages in result
        human_msgs = [m for m in result if isinstance(m, HumanMessage)]
        assert len(human_msgs) >= 1
        assert "Question 0" in human_msgs[0].content

    def test_recent_messages_kept(self):
        msgs = self._make_history(50)
        result = trim_conversation(msgs, max_tokens=500)
        # The last message in result should be the last message from input
        assert result[-1].content == msgs[-1].content

    def test_empty_messages(self):
        assert trim_conversation([], max_tokens=1000) == []

    def test_messages_are_trimmed(self):
        msgs = self._make_history(50)
        result = trim_conversation(msgs, max_tokens=500)
        assert len(result) < len(msgs)

    def test_custom_token_counter(self):
        msgs = self._make_history(5)
        # Counter that says every message is 100 tokens
        result = trim_conversation(
            msgs, max_tokens=350, token_counter=lambda m: 100
        )
        # Budget=350, each msg=100.
        # Pinned: system(100) + first_human(100) = 200.
        # Remaining 150 → 1 more message from tail.
        assert len(result) == 3

    def test_only_system_message(self):
        msgs = [SystemMessage(content="System")]
        result = trim_conversation(msgs, max_tokens=1000)
        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)

    def test_no_system_message(self):
        msgs = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
        ]
        result = trim_conversation(msgs, max_tokens=100_000)
        assert len(result) == 2

    def test_tool_messages_included(self):
        msgs = [
            SystemMessage(content="System"),
            HumanMessage(content="Hi"),
            AIMessage(
                content="Let me check",
                additional_kwargs={},
                tool_calls=[{"id": "1", "name": "t", "args": {}}],
            ),
            ToolMessage(content="result data", tool_call_id="1"),
            AIMessage(content="Here's what I found"),
        ]
        result = trim_conversation(msgs, max_tokens=100_000)
        assert len(result) == 5

    def test_pinned_messages_exceed_budget(self):
        """When even pinned messages exceed budget, return them anyway (best effort)."""
        msgs = [
            SystemMessage(content="x" * 1000),
            HumanMessage(content="y" * 1000),
            AIMessage(content="z" * 1000),
        ]
        # Budget so small even system+first_human won't fit
        result = trim_conversation(msgs, max_tokens=10)
        # Should return pinned messages (system + first human) as best effort
        assert len(result) == 2
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)

    def test_single_human_message_not_duplicated(self):
        """First human message shouldn't appear twice when it's also the most recent."""
        msgs = [
            SystemMessage(content="System"),
            HumanMessage(content="Only question"),
        ]
        result = trim_conversation(msgs, max_tokens=100_000)
        assert len(result) == 2

    def test_chronological_order_maintained(self):
        """Result messages should be in chronological order."""
        msgs = self._make_history(10)
        result = trim_conversation(msgs, max_tokens=1000)
        # All messages in result should maintain their relative order from original
        original_indices = []
        for r_msg in result:
            for i, o_msg in enumerate(msgs):
                if r_msg is o_msg or r_msg.content == o_msg.content:
                    original_indices.append(i)
                    break
        assert original_indices == sorted(original_indices)
