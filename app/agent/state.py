"""
LangGraph agent state schema.

The ``AgentState`` TypedDict is shared across all graph nodes and
represents the data flowing through each invocation step.
"""
from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    The state flowing through every node in the graph.

    ``messages`` uses the ``add_messages`` reducer — new messages are appended
    rather than overwriting the list, giving a persistent conversation history.
    """

    messages: Annotated[list[BaseMessage], add_messages]
