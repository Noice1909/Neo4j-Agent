"""
Chat API routes.

POST /api/v1/chat
    Submit a message in a session.  Session context is persisted automatically
    via the LangGraph Redis checkpointer (keyed by session_id).

GET  /api/v1/chat/stream
    SSE streaming variant — yields tokens as the LLM generates them.

The `session_id` is a client-supplied UUID string.  Different session_ids are
fully isolated.  The same session_id resumes the conversation where it left off
(across reboots, across workers).
"""
from __future__ import annotations

import logging
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from src.core.dependencies import get_agent, get_schema_cache
from src.core.exceptions import AgentError, ReadOnlyViolationError
from src.middleware.rate_limit import limiter
from src.core.config import get_settings
from src.api.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=ChatResponse,
    summary="Send a chat message and receive an agent response.",
    response_description="Agent's natural-language reply.",
)
@limiter.limit(lambda: get_settings().rate_limit_chat)
async def chat(
    request: Request,
    body: ChatRequest,
    agent=Depends(get_agent),
) -> ChatResponse:
    """
    Submit a message to the Neo4j knowledge-graph agent.

    The agent maintains full conversation history per `session_id`.
    All generated Cypher queries are executed read-only — write operations
    are blocked at both the application and database layers.
    """
    logger.info("Chat request: session_id=%s message=%r", body.session_id, body.message[:80])

    config = {"configurable": {"thread_id": body.session_id}}
    input_state = {"messages": [HumanMessage(content=body.message)]}

    try:
        result = await agent.ainvoke(input_state, config=config)
    except ReadOnlyViolationError:
        raise
    except Exception as exc:
        logger.error(
            "Agent error (session=%s): %s", body.session_id, exc, exc_info=True
        )
        raise AgentError(str(exc)) from exc

    # The last message in the result is the AI response
    messages = result.get("messages", [])
    if not messages:
        raise AgentError("Agent returned no response.")

    answer: str = messages[-1].content

    # Extract token usage if available
    tokens_used: int | None = None
    last_msg = messages[-1]
    if hasattr(last_msg, "usage_metadata") and last_msg.usage_metadata:
        tokens_used = last_msg.usage_metadata.get("total_tokens")

    logger.info(
        "Chat response: session_id=%s answer=%r (tokens=%s)",
        body.session_id,
        answer[:120],
        tokens_used,
    )

    return ChatResponse(
        session_id=body.session_id,
        message=answer,
        tokens_used=tokens_used,
    )


@router.get(
    "/stream",
    summary="Stream a chat response via Server-Sent Events.",
    response_description="SSE stream of agent response tokens.",
)
@limiter.limit(lambda: get_settings().rate_limit_chat)
async def chat_stream(
    request: Request,
    session_id: str,
    message: str,
    agent=Depends(get_agent),
) -> StreamingResponse:
    """
    Submit a message and receive the response as a Server-Sent Events stream.

    The response is streamed token-by-token as the LLM generates it.
    Use this endpoint for real-time UIs.
    """
    logger.info("Stream request: session_id=%s message=%r", session_id, message[:80])

    config = {"configurable": {"thread_id": session_id}}
    input_state = {"messages": [HumanMessage(content=message)]}

    async def _event_stream() -> AsyncIterator[str]:
        try:
            async for event in agent.astream_events(
                input_state, config=config, version="v2"
            ):
                kind = event.get("event", "")
                # Emit on-llm-stream events (token-level)
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield f"data: {chunk.content}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error("Stream error (session=%s): %s", session_id, exc, exc_info=True)
            yield "data: [ERROR] An internal error occurred. Please try again.\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        },
    )
