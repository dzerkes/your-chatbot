"""The same agent tests with the real tools in-process and over MCP.

One seeded database serves both transports. For MCP, the real data server runs
on a local port with the database-backed tools, and the agent reaches it
through McpToolset with signed identity tokens.
"""

import json
import subprocess
from collections.abc import AsyncIterator, Iterator
from contextlib import AsyncExitStack
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
import pytest_asyncio
from chatbot.agent import Agent
from chatbot.config import Settings
from chatbot.llm.base import LLMEvent, LLMStop, LLMTextDelta, LLMToolCall, LLMUsage
from chatbot.llm.fake import ScriptedProvider
from chatbot.rag.embeddings import build_embedder
from chatbot.rag.ingest import ingest_documents, load_documents
from chatbot.store.memory import MemoryStore
from chatbot.tools.local import open_local_tools
from chatbot.tools.mcp_client import McpTool, McpToolset
from chatbot.tools.registry import ToolRegistry
from chatbot.types import ToolResult, ToolUse, TurnCompleted, Usage, User
from psycopg import sql
from psycopg.conninfo import make_conninfo
from pydantic import JsonValue, SecretStr

from mcp_servers.data_server import create_app
from mcp_servers.tests.server import serve

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]
ROOT = Path(__file__).resolve().parents[1]
SECRET = SecretStr("integration-" + "x" * 32)
ALICE = User(external_id="alice", display_name="Alice", groups=["employees", "engineering"])
BOB = User(external_id="bob", display_name="Bob", groups=["employees", "finance", "leadership"])


def compose(*args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout
    )


@pytest.fixture(scope="module")
def database() -> Iterator[Settings]:
    name = f"c22_{uuid4().hex}"
    created = compose(
        "exec", "-T", "postgres", "sh", "-c",
        'export PGPASSWORD="$POSTGRES_PASSWORD"; '
        'createdb -h 127.0.0.1 -U postgres -T template0 -O "$OWNER_DB_USER" "$1" && '
        'exec psql -Xq -h 127.0.0.1 -U postgres -d "$1" -c "CREATE EXTENSION vector"',
        "sh", name,
    )  # fmt: skip
    assert created.returncode == 0, created.stderr
    try:
        migrated = compose(
            "run", "--rm", "--no-deps", "-e", f"POSTGRES_DB={name}",
            "-e", "SEED_DEMO_DATA=false", "migrate", timeout=180,
        )  # fmt: skip
        assert migrated.returncode == 0, migrated.stderr
        port = int(compose("port", "postgres", "5432").stdout.strip().rsplit(":", 1)[1])
        settings = Settings(
            postgres_db=name,
            postgres_port=port,
            embedding_cache_dir=str(ROOT / ".cache/fastembed"),
            mcp_token_secret=SECRET,
            tracing_enabled=False,
        )
        owner = make_conninfo(
            host=settings.postgres_host,
            port=port,
            dbname=name,
            user=settings.postgres_user,
            password=settings.postgres_password.get_secret_value(),
        )
        with psycopg.connect(owner, autocommit=True) as conn:
            conn.execute(
                sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                    sql.Identifier(name), sql.Identifier(settings.ro_db_user)
                )
            )
            conn.execute((ROOT / "db/seed/01_demo.sql").read_text())
        yield settings
    finally:
        compose(
            "exec", "-T", "postgres", "sh", "-c",
            'PGPASSWORD="$POSTGRES_PASSWORD" exec dropdb -h 127.0.0.1 -U postgres --if-exists "$1"',
            "sh", name,
        )  # fmt: skip


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def transports(database: Settings) -> AsyncIterator[dict[str, ToolRegistry]]:
    embedder = build_embedder(database)
    await ingest_documents(load_documents(ROOT / "db/seed/documents"), database, embedder)
    await embedder.aclose()
    with serve(create_app(database)) as base:
        remote = database.model_copy(update={"mcp_server_url": f"{base}/mcp"})
        async with AsyncExitStack() as stack:
            local = await open_local_tools(database, stack)
            mcp = (await stack.enter_async_context(McpToolset(remote))).registry
            yield {"inprocess": local, "mcp": mcp}


def call(name: str, arguments: dict[str, JsonValue]) -> list[LLMEvent | Exception]:
    return [
        LLMToolCall(tool_use=ToolUse(id=f"call-{uuid4().hex[:8]}", name=name, arguments=arguments)),
        LLMUsage(usage=Usage(input_tokens=1, output_tokens=1)),
        LLMStop(reason="tool_use"),
    ]


def answer(text: str = "Done.") -> list[LLMEvent | Exception]:
    return [
        LLMTextDelta(text=text),
        LLMUsage(usage=Usage(input_tokens=1, output_tokens=1)),
        LLMStop(reason="end_turn"),
    ]


async def run_agent(
    tools: ToolRegistry, settings: Settings, user: User, *steps: list[LLMEvent | Exception]
) -> list[ToolResult]:
    """Run one scripted turn and return the tool results the agent saved."""
    store = MemoryStore()
    agent = Agent(store, ScriptedProvider(list(steps)), tools, settings, system="Test")
    saved = await store.upsert_user(user)
    session = await store.create_session(saved.id)
    events = [event async for event in agent.respond(session.id, "Question", saved)]
    assert isinstance(events[-1], TurnCompleted), events[-1]
    messages = await store.list_messages(session.id, saved.id)
    return [block for m in messages for block in m.content if isinstance(block, ToolResult)]


TRANSPORTS = ["inprocess", "mcp"]


async def test_the_mcp_registry_lists_the_same_tools(transports: dict[str, ToolRegistry]) -> None:
    assert transports["mcp"].definitions() == transports["inprocess"].definitions()
    assert {tool.name for tool in transports["mcp"].tools()} == {"query_projects", "search_docs"}
    assert all(isinstance(tool, McpTool) for tool in transports["mcp"].tools())


@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_agent_queries_projects(
    transports: dict[str, ToolRegistry], database: Settings, transport: str
) -> None:
    [result] = await run_agent(
        transports[transport],
        database,
        ALICE,
        call("query_projects", {"status": "active", "client": "Northstar"}),
        answer(),
    )
    assert not result.is_error
    assert result.content == {
        "items": [
            {
                "id": 101,
                "name": "Research Portal",
                "client_id": 1,
                "client_name": "Northstar Labs",
                "description": "Searchable workspace for research reports.",
                "status": "active",
                "start_date": "2026-01-15",
                "end_date": "2026-12-31",
            }
        ],
        "total": 1,
        "next_cursor": None,
    }


@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_agent_searches_documents_within_the_users_groups(
    transports: dict[str, ToolRegistry], database: Settings, transport: str
) -> None:
    query: dict[str, JsonValue] = {"query": "what is our day rate for an architect", "limit": 10}
    [for_alice] = await run_agent(
        transports[transport], database, ALICE, call("search_docs", query), answer()
    )
    [for_bob] = await run_agent(
        transports[transport], database, BOB, call("search_docs", query), answer()
    )
    assert "finance/" not in json.dumps(for_alice.content)
    top = json.loads(json.dumps(for_bob.content))["items"][0]
    assert (top["source"], top["section"]) == ("finance/pricing-guidelines-2026.md", "Day rates")


@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_agent_gets_tool_errors_back_and_continues(
    transports: dict[str, ToolRegistry], database: Settings, transport: str
) -> None:
    results = await run_agent(
        transports[transport],
        database,
        ALICE,
        call("search_docs", {"query": "leave", "groups": ["finance"]}),
        call("query_projects", {"status": "unknown"}),
        answer(),
    )
    assert [r.is_error for r in results] == [True, True]
    assert "groups: Extra inputs are not permitted" in str(results[0].content)
    assert "status:" in str(results[1].content)


async def test_both_transports_return_identical_results(
    transports: dict[str, ToolRegistry], database: Settings
) -> None:
    steps: list[tuple[str, dict[str, JsonValue]]] = [
        ("query_projects", {"status": "active"}),
        ("query_projects", {"text": "portal", "pagination": {"limit": 1}}),
        ("search_docs", {"query": "how do I connect to the VPN"}),
        ("search_docs", {"query": "salary bands", "limit": 3}),
    ]
    for user in (ALICE, BOB):
        outputs = {
            transport: [
                (await run_agent(transports[transport], database, user, call(n, a), answer()))[0]
                for n, a in steps
            ]
            for transport in TRANSPORTS
        }
        local, remote = outputs["inprocess"], outputs["mcp"]
        # Cursors are bound to the user's id, which differs per stored user; compare the rest.
        for a, b in zip(local, remote, strict=True):
            assert (a.is_error, _without_cursor(a.content)) == (
                b.is_error,
                _without_cursor(b.content),
            )


def _without_cursor(content: JsonValue) -> JsonValue:
    if isinstance(content, dict):
        return {k: v for k, v in content.items() if k != "next_cursor"}
    return content
