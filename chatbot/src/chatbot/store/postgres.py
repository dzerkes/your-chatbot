"""Async Postgres history store; all production history SQL lives here."""

# Cursor and transaction contexts depend on the acquired connection.
# ruff: noqa: SIM117

from collections.abc import Sequence
from typing import Self
from uuid import UUID

from psycopg import errors
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from chatbot.config import Settings
from chatbot.errors import ConflictError, NotFoundError
from chatbot.store.base import Session, Turn, validate_turn_messages
from chatbot.types import Message, User


class PostgresStore:
    """Use with ``async with``; the pool connects as chatbot_app only."""

    def __init__(self, settings: Settings) -> None:
        self._pool = AsyncConnectionPool(
            conninfo=make_conninfo(
                host=settings.postgres_host,
                port=settings.postgres_port,
                dbname=settings.postgres_db,
                user=settings.app_db_user,
                password=settings.app_db_password.get_secret_value(),
            ),
            min_size=1,
            max_size=10,
            open=False,
        )

    async def __aenter__(self) -> Self:
        await self._pool.open(wait=True)
        return self

    async def __aexit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        await self._pool.close()

    async def upsert_user(self, user: User) -> User:
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cursor:
                    await cursor.execute(
                        """
                        INSERT INTO users (id, external_id, display_name, groups)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (external_id) DO UPDATE
                        SET display_name = EXCLUDED.display_name, groups = EXCLUDED.groups
                        RETURNING id, external_id, display_name, groups
                        """,
                        (user.id, user.external_id, user.display_name, user.groups),
                    )
                    row = await cursor.fetchone()
                    assert row is not None
                    return User.model_validate(row)
        except errors.UniqueViolation as exc:
            raise ConflictError("User ID already exists") from exc

    async def create_session(self, owner_id: UUID, title: str | None = None) -> Session:
        session = Session(owner_id=owner_id, title=title)
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cursor:
                    await cursor.execute(
                        """
                        INSERT INTO sessions (id, owner_id, title, summary, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id, owner_id, title, summary, created_at, updated_at
                        """,
                        (
                            session.id,
                            session.owner_id,
                            session.title,
                            session.summary,
                            session.created_at,
                            session.updated_at,
                        ),
                    )
                    row = await cursor.fetchone()
                    assert row is not None
                    return Session.model_validate(row)
        except errors.ForeignKeyViolation as exc:
            raise NotFoundError("User not found") from exc

    async def get_session(self, session_id: UUID, owner_id: UUID) -> Session | None:
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    SELECT id, owner_id, title, summary, created_at, updated_at
                    FROM sessions WHERE id = %s AND owner_id = %s
                    """,
                    (session_id, owner_id),
                )
                row = await cursor.fetchone()
                return Session.model_validate(row) if row is not None else None

    async def list_sessions(self, owner_id: UUID) -> list[Session]:
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    SELECT id, owner_id, title, summary, created_at, updated_at
                    FROM sessions WHERE owner_id = %s ORDER BY updated_at DESC, id DESC
                    """,
                    (owner_id,),
                )
                return [Session.model_validate(row) for row in await cursor.fetchall()]

    async def update_session(self, session: Session) -> Session:
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    UPDATE sessions SET title = %s, summary = %s, updated_at = now()
                    WHERE id = %s AND owner_id = %s
                    RETURNING id, owner_id, title, summary, created_at, updated_at
                    """,
                    (session.title, session.summary, session.id, session.owner_id),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise NotFoundError("Session not found")
                return Session.model_validate(row)

    async def save_turn(self, turn: Turn, messages: Sequence[Message], owner_id: UUID) -> None:
        validate_turn_messages(turn, messages)
        try:
            async with self._pool.connection() as conn:
                # A failure anywhere below rolls back the turn and every message.
                async with conn.transaction():
                    cursor = await conn.execute(
                        """
                        INSERT INTO turns (
                            id, session_id, status, trace_id, input_tokens, output_tokens,
                            cache_read_input_tokens, cache_creation_input_tokens,
                            cost_usd, latency_ms, created_at, updated_at, completed_at
                        )
                        SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        FROM sessions WHERE id = %s AND owner_id = %s
                        RETURNING id
                        """,
                        (
                            turn.id,
                            turn.session_id,
                            turn.status,
                            turn.trace_id,
                            turn.input_tokens,
                            turn.output_tokens,
                            turn.cache_read_input_tokens,
                            turn.cache_creation_input_tokens,
                            turn.cost_usd,
                            turn.latency_ms,
                            turn.created_at,
                            turn.updated_at,
                            turn.completed_at,
                            turn.session_id,
                            owner_id,
                        ),
                    )
                    if await cursor.fetchone() is None:
                        raise NotFoundError("Session not found")
                    for message in messages:
                        await conn.execute(
                            """
                            INSERT INTO messages (
                                id, session_id, turn_id, sequence, role,
                                content, model, usage, created_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                message.id,
                                message.session_id,
                                message.turn_id,
                                message.sequence,
                                message.role,
                                Jsonb([block.model_dump(mode="json") for block in message.content]),
                                message.model,
                                Jsonb(message.usage.model_dump(mode="json"))
                                if message.usage is not None
                                else None,
                                message.created_at,
                            ),
                        )
                    await conn.execute(
                        "UPDATE sessions SET updated_at = now() WHERE id = %s",
                        (turn.session_id,),
                    )
        except errors.UniqueViolation as exc:
            raise ConflictError("Turn or message already exists") from exc

    async def list_turns(self, session_id: UUID, owner_id: UUID) -> list[Turn]:
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    SELECT t.* FROM turns AS t
                    JOIN sessions AS s ON s.id = t.session_id
                    WHERE t.session_id = %s AND s.owner_id = %s
                    ORDER BY t.created_at, t.id
                    """,
                    (session_id, owner_id),
                )
                return [Turn.model_validate(row) for row in await cursor.fetchall()]

    async def list_messages(self, session_id: UUID, owner_id: UUID) -> list[Message]:
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    """
                    SELECT m.* FROM messages AS m
                    JOIN sessions AS s ON s.id = m.session_id
                    WHERE m.session_id = %s AND s.owner_id = %s
                    ORDER BY m.sequence
                    """,
                    (session_id, owner_id),
                )
                return [Message.model_validate(row) for row in await cursor.fetchall()]
