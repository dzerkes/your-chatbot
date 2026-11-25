"""User memory: facts a user asked the assistant to remember across sessions.

Every operation takes the owner's user id, so one user can never read or
delete another's memories. Facts are added and forgotten, never edited; the
database role has no UPDATE on the table.
"""

# Cursor and transaction contexts depend on the acquired connection.
# ruff: noqa: SIM117

import asyncio
from datetime import UTC, datetime
from itertools import count
from typing import Protocol
from uuid import UUID

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import AwareDatetime, Field

from chatbot.types import DomainModel


class Memory(DomainModel):
    id: int
    fact: str = Field(min_length=1, max_length=300)
    evidence: str = Field(min_length=1, max_length=300)
    source_session_id: UUID | None = None
    created_at: AwareDatetime


class UserMemoryStore(Protocol):
    async def add(
        self, user_id: UUID, fact: str, evidence: str, session_id: UUID | None
    ) -> tuple[Memory, bool]:
        """Save a fact; returns (memory, created). An equal fact (ignoring case) is reused."""
        ...

    async def list(
        self, user_id: UUID, *, query: str | None = None, limit: int = 20, offset: int = 0
    ) -> tuple[list[Memory], int]:
        """Newest first, optionally only facts containing ``query``; returns (page, total)."""
        ...

    async def delete(self, user_id: UUID, memory_id: int) -> bool:
        """Forget one fact; False when it does not exist or belongs to someone else."""
        ...


def _matches(memory: Memory, query: str | None) -> bool:
    return query is None or query.casefold() in memory.fact.casefold()


class InMemoryUserMemoryStore:
    """For unit tests; same behaviour as the Postgres store."""

    def __init__(self) -> None:
        self._rows: dict[int, tuple[UUID, Memory]] = {}
        self._ids = count(1)
        self._lock = asyncio.Lock()

    async def add(
        self, user_id: UUID, fact: str, evidence: str, session_id: UUID | None
    ) -> tuple[Memory, bool]:
        async with self._lock:
            for owner, memory in self._rows.values():
                if owner == user_id and memory.fact.casefold() == fact.casefold():
                    return memory, False
            memory = Memory(
                id=next(self._ids),
                fact=fact,
                evidence=evidence,
                source_session_id=session_id,
                created_at=datetime.now(UTC),
            )
            self._rows[memory.id] = (user_id, memory)
            return memory, True

    async def list(
        self, user_id: UUID, *, query: str | None = None, limit: int = 20, offset: int = 0
    ) -> tuple[list[Memory], int]:
        rows = sorted(
            (m for owner, m in self._rows.values() if owner == user_id and _matches(m, query)),
            key=lambda m: (m.created_at, m.id),
            reverse=True,
        )
        return rows[offset : offset + limit], len(rows)

    async def delete(self, user_id: UUID, memory_id: int) -> bool:
        async with self._lock:
            row = self._rows.get(memory_id)
            if row is None or row[0] != user_id:
                return False
            del self._rows[memory_id]
            return True


_COLUMNS = "id, fact, evidence, source_session_id, created_at"


def _literal_pattern(value: str) -> str:
    return "%" + value.replace("!", "!!").replace("%", "!%").replace("_", "!_") + "%"


class PostgresUserMemoryStore:
    """Uses the history store's pool, which connects as the application role."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def add(
        self, user_id: UUID, fact: str, evidence: str, session_id: UUID | None
    ) -> tuple[Memory, bool]:
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    f"""
                    INSERT INTO user_memory (user_id, fact, evidence, source_session_id)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id, lower(fact)) DO NOTHING
                    RETURNING {_COLUMNS}
                    """,
                    (user_id, fact, evidence, session_id),
                )
                row = await cursor.fetchone()
                if row is not None:
                    return Memory.model_validate(row), True
                await cursor.execute(
                    f"SELECT {_COLUMNS} FROM user_memory "
                    "WHERE user_id = %s AND lower(fact) = lower(%s)",
                    (user_id, fact),
                )
                existing = await cursor.fetchone()
                assert existing is not None
                return Memory.model_validate(existing), False

    async def list(
        self, user_id: UUID, *, query: str | None = None, limit: int = 20, offset: int = 0
    ) -> tuple[list[Memory], int]:
        where, params = "user_id = %s", [user_id]
        if query is not None:
            where += " AND fact ILIKE %s ESCAPE '!'"
            params.append(_literal_pattern(query))  # type: ignore[arg-type]
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    f"SELECT count(*) AS total FROM user_memory WHERE {where}", params
                )
                total_row = await cursor.fetchone()
                assert total_row is not None
                await cursor.execute(
                    f"SELECT {_COLUMNS} FROM user_memory WHERE {where} "
                    "ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s",
                    [*params, limit, offset],
                )
                rows = await cursor.fetchall()
        return [Memory.model_validate(row) for row in rows], int(total_row["total"])

    async def delete(self, user_id: UUID, memory_id: int) -> bool:
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                "DELETE FROM user_memory WHERE id = %s AND user_id = %s", (memory_id, user_id)
            )
            return cursor.rowcount == 1
