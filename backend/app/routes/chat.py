"""Translate the core event stream to server-sent events."""

from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Annotated

import anyio
from chatbot.types import DomainModel
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import Field, field_validator
from starlette.requests import ClientDisconnect
from starlette.types import Receive, Scope, Send

from app.auth import CurrentUser
from app.deps import RuntimeDep
from app.routes.sessions import OwnedSession

router = APIRouter(prefix="/sessions", tags=["chat"])


class ChatRequest(DomainModel):
    message: Annotated[str, Field(min_length=1, max_length=16_000)]

    @field_validator("message")
    @classmethod
    def nonblank_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message must not be blank")
        return value


class ChatStreamResponse(StreamingResponse):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        except ClientDisconnect:
            return

    async def stream_response(self, send: Send) -> None:
        try:
            await super().stream_response(send)
        finally:
            # Close on the same task that iterated the generator: logging and OTel
            # context tokens must be reset there, including after a failed send.
            close = getattr(self.body_iterator, "aclose", None)
            if close is not None:
                with anyio.CancelScope(shield=True):
                    await close()


@router.post(
    "/{session_id}/chat",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "Chat event stream"}},
)
async def chat(
    payload: ChatRequest,
    session: OwnedSession,
    user: CurrentUser,
    runtime: RuntimeDep,
) -> StreamingResponse:
    async def events() -> AsyncGenerator[str, None]:
        yield ": connected\n\n"
        async with aclosing(runtime.chatbot.respond(session.id, payload.message, user)) as stream:
            async for event in stream:
                await anyio.lowlevel.checkpoint()
                yield f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"

    return ChatStreamResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
