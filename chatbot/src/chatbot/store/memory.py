"""In-memory history store with the same ownership and atomicity contract."""

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

from chatbot.errors import ConflictError, NotFoundError
from chatbot.store.base import Session, Turn, validate_turn_messages
from chatbot.types import Message, User


class MemoryStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._users: dict[UUID, User] = {}
        self._user_ids: dict[str, UUID] = {}
        self._sessions: dict[UUID, Session] = {}
        self._turns: dict[UUID, Turn] = {}
        self._messages: dict[UUID, list[Message]] = {}
        self._message_ids: set[UUID] = set()

    async def upsert_user(self, user: User) -> User:
        async with self._lock:
            if user.external_id not in self._user_ids and user.id in self._users:
                raise ConflictError("User ID already exists")
            user_id = self._user_ids.get(user.external_id, user.id)
            stored = user.model_copy(update={"id": user_id}, deep=True)
            self._users[user_id] = stored
            self._user_ids[user.external_id] = user_id
            return stored.model_copy(deep=True)

    async def create_session(self, owner_id: UUID, title: str | None = None) -> Session:
        async with self._lock:
            if owner_id not in self._users:
                raise NotFoundError("User not found")
            session = Session(owner_id=owner_id, title=title)
            self._sessions[session.id] = session
            self._messages[session.id] = []
            return session.model_copy(deep=True)

    async def get_session(self, session_id: UUID, owner_id: UUID) -> Session | None:
        session = self._sessions.get(session_id)
        if session is None or session.owner_id != owner_id:
            return None
        return session.model_copy(deep=True)

    async def list_sessions(self, owner_id: UUID) -> list[Session]:
        sessions = [s for s in self._sessions.values() if s.owner_id == owner_id]
        return [
            s.model_copy(deep=True)
            for s in sorted(sessions, key=lambda s: (s.updated_at, s.id), reverse=True)
        ]

    async def update_session(self, session: Session) -> Session:
        async with self._lock:
            stored = self._sessions.get(session.id)
            if stored is None or stored.owner_id != session.owner_id:
                raise NotFoundError("Session not found")
            updated = stored.model_copy(
                update={
                    "title": session.title,
                    "summary": session.summary,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._sessions[session.id] = updated
            return updated.model_copy(deep=True)

    async def save_turn(self, turn: Turn, messages: Sequence[Message], owner_id: UUID) -> None:
        validate_turn_messages(turn, messages)
        async with self._lock:
            session = self._sessions.get(turn.session_id)
            if session is None or session.owner_id != owner_id:
                raise NotFoundError("Session not found")
            if turn.id in self._turns:
                raise ConflictError("Turn already exists")
            existing = {message.sequence for message in self._messages[turn.session_id]}
            if existing.intersection(message.sequence for message in messages):
                raise ConflictError("Message sequence already exists")
            if self._message_ids.intersection(message.id for message in messages):
                raise ConflictError("Message ID already exists")
            self._turns[turn.id] = turn.model_copy(deep=True)
            self._messages[turn.session_id].extend(m.model_copy(deep=True) for m in messages)
            self._message_ids.update(m.id for m in messages)
            self._sessions[turn.session_id] = session.model_copy(
                update={"updated_at": datetime.now(UTC)}
            )

    async def list_turns(self, session_id: UUID, owner_id: UUID) -> list[Turn]:
        session = self._sessions.get(session_id)
        if session is None or session.owner_id != owner_id:
            return []
        turns = (t for t in self._turns.values() if t.session_id == session_id)
        return [t.model_copy(deep=True) for t in sorted(turns, key=lambda t: (t.created_at, t.id))]

    async def list_messages(self, session_id: UUID, owner_id: UUID) -> list[Message]:
        session = self._sessions.get(session_id)
        if session is None or session.owner_id != owner_id:
            return []
        messages = sorted(self._messages[session_id], key=lambda m: m.sequence)
        return [m.model_copy(deep=True) for m in messages]
