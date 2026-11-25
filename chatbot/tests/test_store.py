"""Shared history contract for the in-memory and Postgres stores."""

import subprocess
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import psycopg
import pytest
from chatbot.config import Settings
from chatbot.errors import ConflictError, NotFoundError
from chatbot.store.base import HistoryStore, Turn
from chatbot.store.memory import MemoryStore
from chatbot.store.memory_store import (
    InMemoryUserMemoryStore,
    PostgresUserMemoryStore,
    UserMemoryStore,
)
from chatbot.store.postgres import PostgresStore
from chatbot.types import Message, Text, ToolResult, ToolUse, Usage, User
from psycopg.conninfo import make_conninfo

ROOT = Path(__file__).resolve().parents[2]


async def exercise_history(store: HistoryStore) -> None:
    external_id = f"c06-{uuid4().hex}"
    alice = await store.upsert_user(
        User(external_id=external_id, display_name="Alice", groups=["engineering"])
    )
    updated_alice = await store.upsert_user(
        User(external_id=external_id, display_name="Alice A.", groups=["engineering", "demo"])
    )
    assert updated_alice.id == alice.id
    assert updated_alice.display_name == "Alice A."
    assert updated_alice.groups == ["engineering", "demo"]
    bob = await store.upsert_user(User(external_id=f"{external_id}-bob", display_name="Bob"))

    with pytest.raises(NotFoundError):
        await store.create_session(uuid4())
    session = await store.create_session(alice.id, title="Project chat")
    assert await store.get_session(session.id, alice.id) == session
    assert await store.get_session(session.id, bob.id) is None
    assert await store.list_sessions(bob.id) == []
    assert await store.list_sessions(alice.id) == [session]

    renamed = await store.update_session(
        session.model_copy(update={"title": "Active projects", "summary": "ignored"})
    )
    assert renamed.title == "Active projects"
    assert renamed.summary is None  # summaries change only through update_summary
    summarized = await store.update_summary(session.id, alice.id, "Found one project", 5)
    assert (summarized.summary, summarized.summary_through_sequence) == ("Found one project", 5)
    assert summarized.summary_updated_at is not None
    # Coverage only moves forward: a stale summary from a concurrent turn is ignored.
    stale = await store.update_summary(session.id, alice.id, "Older", 3)
    assert (stale.summary, stale.summary_through_sequence) == ("Found one project", 5)
    newer = await store.update_summary(session.id, alice.id, "Found two projects", 9)
    assert (newer.summary, newer.summary_through_sequence) == ("Found two projects", 9)
    assert (await store.get_session(session.id, alice.id)) == newer
    with pytest.raises(NotFoundError):
        await store.update_summary(session.id, bob.id, "Not yours", 20)
    assert renamed.updated_at >= session.updated_at
    with pytest.raises(NotFoundError):
        await store.update_session(renamed.model_copy(update={"owner_id": bob.id}))

    turn = Turn(
        session_id=session.id,
        status="completed",
        trace_id="trace-c06",
        input_tokens=24,
        output_tokens=8,
        cache_read_input_tokens=4,
        cost_usd=Decimal("0.001200"),
        latency_ms=75,
    )
    messages = [
        Message(
            session_id=session.id,
            turn_id=turn.id,
            sequence=0,
            role="user",
            content=[Text(text="Which projects are active?")],
        ),
        Message(
            session_id=session.id,
            turn_id=turn.id,
            sequence=1,
            role="assistant",
            model="fake:scripted",
            content=[
                Text(text="Checking projects"),
                ToolUse(id="call-1", name="query_projects", arguments={"status": "active"}),
            ],
            usage=Usage(input_tokens=12, output_tokens=4, cache_read_input_tokens=4),
        ),
        Message(
            session_id=session.id,
            turn_id=turn.id,
            sequence=2,
            role="tool",
            content=[
                ToolResult(
                    tool_use_id="call-1",
                    content={"items": [{"id": 1, "name": "Demo"}], "total": 1, "next_cursor": None},
                )
            ],
        ),
        Message(
            session_id=session.id,
            turn_id=turn.id,
            sequence=3,
            role="assistant",
            content=[Text(text="Demo is active.")],
        ),
    ]

    with pytest.raises(NotFoundError):
        await store.save_turn(turn, messages, bob.id)
    assert await store.list_turns(session.id, alice.id) == []
    await store.save_turn(turn, messages, alice.id)
    assert await store.list_turns(session.id, alice.id) == [turn]
    assert await store.list_messages(session.id, alice.id) == messages
    assert await store.list_turns(session.id, bob.id) == []
    assert await store.list_messages(session.id, bob.id) == []
    active = await store.get_session(session.id, alice.id)
    assert active is not None
    assert active.updated_at >= renamed.updated_at

    invalid_turn = Turn(session_id=session.id)
    with pytest.raises(ValueError):
        await store.save_turn(
            invalid_turn,
            [Message(session_id=session.id, turn_id=uuid4(), sequence=4, role="user", content=[])],
            alice.id,
        )
    assert await store.list_turns(session.id, alice.id) == [turn]

    conflict_turn = Turn(session_id=session.id)
    conflict_messages = [
        Message(
            session_id=session.id, turn_id=conflict_turn.id, sequence=4, role="user", content=[]
        ),
        Message(
            session_id=session.id, turn_id=conflict_turn.id, sequence=1, role="tool", content=[]
        ),
    ]
    with pytest.raises(ConflictError):
        await store.save_turn(conflict_turn, conflict_messages, alice.id)
    assert await store.list_turns(session.id, alice.id) == [turn]
    assert await store.list_messages(session.id, alice.id) == messages

    reused_id_turn = Turn(session_id=session.id)
    with pytest.raises(ConflictError):
        await store.save_turn(
            reused_id_turn,
            [
                Message(
                    id=messages[0].id,
                    session_id=session.id,
                    turn_id=reused_id_turn.id,
                    sequence=4,
                    role="user",
                    content=[Text(text="another turn")],
                )
            ],
            alice.id,
        )
    assert await store.list_turns(session.id, alice.id) == [turn]


@pytest.mark.unit
async def test_memory_store_contract() -> None:
    store = MemoryStore()
    _static_contract: HistoryStore = store
    await exercise_history(_static_contract)


@pytest.mark.unit
async def test_memory_store_returns_copies() -> None:
    store = MemoryStore()
    user = await store.upsert_user(User(external_id="alice", display_name="Alice"))
    session = await store.create_session(user.id)
    session.title = "Changed only in caller"
    stored = await store.get_session(session.id, user.id)
    assert stored is not None
    assert stored.title is None
    turn = Turn(session_id=session.id)
    message = Message(
        session_id=session.id, turn_id=turn.id, sequence=0, role="user", content=[Text(text="hi")]
    )
    await store.save_turn(turn, [message], user.id)
    message.content.append(Text(text="not stored"))
    loaded = await store.list_messages(session.id, user.id)
    assert loaded[0].content == [Text(text="hi")]
    loaded[0].content.clear()
    assert (await store.list_messages(session.id, user.id))[0].content == [Text(text="hi")]


def compose(*args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.fixture
def temporary_database(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, int]]:
    monkeypatch.chdir(ROOT)
    database = f"c06_{uuid4().hex}"
    created = compose(
        "exec",
        "-T",
        "postgres",
        "sh",
        "-c",
        # The superuser adds pgvector, as db/init does for the main database.
        'export PGPASSWORD="$POSTGRES_PASSWORD"; '
        'createdb -h 127.0.0.1 -U postgres -T template0 -O "$OWNER_DB_USER" "$1" && '
        'exec psql -Xq -h 127.0.0.1 -U postgres -d "$1" -c "CREATE EXTENSION vector"',
        "sh",
        database,
    )
    assert created.returncode == 0, created.stderr
    try:
        migrated = compose(
            "run",
            "--rm",
            "--no-deps",
            "-e",
            f"POSTGRES_DB={database}",
            "migrate",
            timeout=180,
        )
        assert migrated.returncode == 0, migrated.stderr
        published = compose("port", "postgres", "5432")
        assert published.returncode == 0, published.stderr
        port = int(published.stdout.strip().rsplit(":", 1)[1])
        yield database, port
    finally:
        dropped = compose(
            "exec",
            "-T",
            "postgres",
            "sh",
            "-c",
            'PGPASSWORD="$POSTGRES_PASSWORD" exec dropdb -h 127.0.0.1 -U postgres --if-exists "$1"',
            "sh",
            database,
        )
        assert dropped.returncode == 0, dropped.stderr


@pytest.mark.integration
async def test_postgres_store_contract(temporary_database: tuple[str, int]) -> None:
    database, port = temporary_database
    settings = Settings(postgres_db=database, postgres_port=port)
    async with PostgresStore(settings) as store:
        _static_contract: HistoryStore = store
        await exercise_history(_static_contract)


async def exercise_memories(store: UserMemoryStore, alice: UUID, bob: UUID) -> None:
    first, created = await store.add(alice, "Prefers bullet points", "I like bullet points", None)
    again, created_again = await store.add(alice, "PREFERS bullet points", "x", None)
    assert created and not created_again and again.id == first.id
    await store.add(alice, "Works on 100%_real projects", "q", None)
    await store.add(bob, "Bob's fact", "q", None)
    page, total = await store.list(alice)
    assert total == 2 and page[0].fact == "Works on 100%_real projects"
    # LIKE wildcards in the query are literal text.
    assert [m.fact for m in (await store.list(alice, query="0%_r"))[0]] == [
        "Works on 100%_real projects"
    ]
    assert (await store.list(alice, query="%"))[1] == 1
    assert not await store.delete(bob, first.id)
    assert await store.delete(alice, first.id) and not await store.delete(alice, first.id)


def test_memory_store_contract_in_memory() -> None:
    import anyio

    anyio.run(exercise_memories, InMemoryUserMemoryStore(), uuid4(), uuid4())


@pytest.mark.integration
async def test_postgres_memory_store_and_grants(temporary_database: tuple[str, int]) -> None:
    database, port = temporary_database
    settings = Settings(postgres_db=database, postgres_port=port)
    async with PostgresStore(settings) as store:
        alice = await store.upsert_user(User(external_id="mem-alice", display_name="A"))
        bob = await store.upsert_user(User(external_id="mem-bob", display_name="B"))
        await exercise_memories(PostgresUserMemoryStore(store.pool), alice.id, bob.id)
        async with store.pool.connection() as conn:
            # Facts are added and forgotten, never rewritten in place.
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                await conn.execute("UPDATE user_memory SET fact = 'changed'")
    reader = make_conninfo(
        host=settings.postgres_host,
        port=port,
        dbname=database,
        user=settings.ro_db_user,
        password=settings.ro_db_password.get_secret_value(),
    )
    async with await psycopg.AsyncConnection.connect(reader) as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            await conn.execute("SELECT count(*) FROM user_memory")
