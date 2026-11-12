"""Owner-scoped session endpoints; persistence belongs to the chatbot store."""

from typing import Annotated
from uuid import UUID

from chatbot.store.base import Session
from chatbot.types import DomainModel, Message
from fastapi import APIRouter, Depends, HTTPException
from pydantic import StringConstraints

from app.auth import CurrentUser
from app.deps import RuntimeDep

router = APIRouter(prefix="/sessions", tags=["sessions"])


class CreateSessionRequest(DomainModel):
    title: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
        | None
    ) = None


class SessionDetail(Session):
    messages: list[Message]


async def owned_session(session_id: UUID, user: CurrentUser, runtime: RuntimeDep) -> Session:
    session = await runtime.chatbot.store.get_session(session_id, user.id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


OwnedSession = Annotated[Session, Depends(owned_session)]


@router.get("")
async def list_sessions(user: CurrentUser, runtime: RuntimeDep) -> list[Session]:
    return await runtime.chatbot.store.list_sessions(user.id)


@router.post("", status_code=201)
async def create_session(
    user: CurrentUser,
    runtime: RuntimeDep,
    payload: CreateSessionRequest | None = None,
) -> Session:
    return await runtime.chatbot.store.create_session(
        user.id, title=payload.title if payload else None
    )


@router.get("/{session_id}")
async def get_session(
    session: OwnedSession, user: CurrentUser, runtime: RuntimeDep
) -> SessionDetail:
    messages = await runtime.chatbot.store.list_messages(session.id, user.id)
    return SessionDetail(**session.model_dump(), messages=messages)
