"""The search_docs tool over the seed documents with the local models.

Exercises the real path the agent uses: tool argument validation, hybrid
retrieval on the read-only role, the local cross-encoder and pagination.
"""

import json
import subprocess
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg
import pytest
import pytest_asyncio
from chatbot.config import Settings
from chatbot.rag.embeddings import build_embedder
from chatbot.rag.ingest import ingest_documents, load_documents
from chatbot.tools.base import Tool, ToolContext
from chatbot.tools.search_docs import SearchDocs, SearchDocsInput
from chatbot.types import User
from psycopg import sql
from psycopg.conninfo import make_conninfo

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]
ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "tests/fixtures/retrieval_labeled_set.json").read_text())["cases"]
RESTRICTED = ("finance/", "hr/")
ALICE = User(external_id="alice", display_name="Alice", groups=["employees", "engineering"])
BOB = User(external_id="bob", display_name="Bob", groups=["employees", "finance", "leadership"])
NOBODY = User(external_id="nobody", display_name="Nobody", groups=[])


def compose(*args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout
    )


def owner(settings: Settings) -> str:
    return make_conninfo(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password.get_secret_value(),
    )


@pytest.fixture(scope="module")
def docs_database() -> Iterator[Settings]:
    database = f"c20_{uuid4().hex}"
    created = compose(
        "exec",
        "-T",
        "postgres",
        "sh",
        "-c",
        'export PGPASSWORD="$POSTGRES_PASSWORD"; '
        'createdb -h 127.0.0.1 -U postgres -T template0 -O "$OWNER_DB_USER" "$1" && '
        'exec psql -Xq -h 127.0.0.1 -U postgres -d "$1" -c "CREATE EXTENSION vector"',
        "sh",
        database,
    )
    assert created.returncode == 0, created.stderr
    try:
        migrated = compose(
            "run", "--rm", "--no-deps", "-e", f"POSTGRES_DB={database}",
            "-e", "SEED_DEMO_DATA=false", "migrate", timeout=180,
        )  # fmt: skip
        assert migrated.returncode == 0, migrated.stderr
        published = compose("port", "postgres", "5432")
        assert published.returncode == 0, published.stderr
        settings = Settings(
            postgres_db=database,
            postgres_port=int(published.stdout.strip().rsplit(":", 1)[1]),
            embedding_cache_dir=str(ROOT / ".cache/fastembed"),
            reranker="local",
        )
        with psycopg.connect(owner(settings), autocommit=True) as conn:
            conn.execute(
                sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(sql.Identifier(database))
            )
            conn.execute(
                sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                    sql.Identifier(database), sql.Identifier(settings.ro_db_user)
                )
            )
        yield settings
    finally:
        dropped = compose(
            "exec", "-T", "postgres", "sh", "-c",
            'PGPASSWORD="$POSTGRES_PASSWORD" exec dropdb -h 127.0.0.1 -U postgres --if-exists "$1"',
            "sh", database,
        )  # fmt: skip
        assert dropped.returncode == 0, dropped.stderr


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def search_tool(docs_database: Settings) -> AsyncIterator[Tool[SearchDocsInput]]:
    embedder = build_embedder(docs_database)
    await ingest_documents(load_documents(ROOT / "db/seed/documents"), docs_database, embedder)
    async with SearchDocs(docs_database, embedder=embedder) as docs:
        yield docs.tool()


def as_user(user: User) -> ToolContext:
    return ToolContext(user=user, session_id=uuid4(), turn_id=uuid4())


async def every_hit(tool: Tool[SearchDocsInput], user: User, query: str) -> list[dict[str, Any]]:
    """Follow next_cursor to the end, as the model would."""
    hits: list[dict[str, Any]] = []
    arguments: dict[str, Any] = {"query": query, "pagination": {"limit": 10}}
    while True:
        page = await tool.execute(arguments, as_user(user))
        hits.extend(page.items)  # type: ignore[arg-type]
        assert page.total >= len(hits)
        if page.next_cursor is None:
            assert len(hits) == page.total
            return hits
        arguments = {"query": query, "pagination": {"limit": 10, "cursor": page.next_cursor}}


async def test_user_without_access_never_sees_restricted_documents(
    search_tool: Tool[SearchDocsInput],
) -> None:
    queries = [case["query"] for case in CASES] + [
        "what is our day rate for an architect",
        "salary band for a staff engineer",
    ]
    for query in queries:
        hits = await every_hit(search_tool, ALICE, query)
        assert not [h["source"] for h in hits if h["source"].startswith(RESTRICTED)], query


async def test_permitted_user_finds_the_restricted_answer_with_citation_fields(
    search_tool: Tool[SearchDocsInput],
) -> None:
    page = await search_tool.execute(
        {"query": "what is our day rate for an architect"}, as_user(BOB)
    )
    top = page.items[0]
    assert isinstance(top, dict)
    assert (top["title"], top["section"], top["source"]) == (
        "Pricing guidelines 2026",
        "Day rates",
        "finance/pricing-guidelines-2026.md",
    )
    assert "1,150 euros" in str(top["snippet"])
    alice = await every_hit(search_tool, ALICE, "what is our day rate for an architect")
    assert alice and "1,150" not in json.dumps(alice)


async def test_user_without_groups_gets_an_empty_result(search_tool: Tool[SearchDocsInput]) -> None:
    page = await search_tool.execute({"query": "annual leave"}, as_user(NOBODY))
    assert (page.items, page.total, page.next_cursor) == ([], 0, None)


async def test_pages_are_disjoint_and_stable(search_tool: Tool[SearchDocsInput]) -> None:
    hits = await every_hit(search_tool, BOB, "incident response and deployment")
    ids = [h["chunk_id"] for h in hits]
    assert len(ids) == len(set(ids)) > 10  # more than one page
    assert ids == [
        h["chunk_id"] for h in await every_hit(search_tool, BOB, "incident response and deployment")
    ]
    first = await search_tool.execute(
        {"query": "incident response and deployment", "pagination": {"limit": 2}}, as_user(BOB)
    )
    assert [h["chunk_id"] for h in first.items] == ids[:2]  # type: ignore[index, call-overload]


async def test_filters_narrow_the_search_within_what_the_user_may_read(
    search_tool: Tool[SearchDocsInput],
) -> None:
    query = "what to do with leave or pricing"
    hits = await every_hit(search_tool, BOB, query)
    assert any(not h["source"].startswith("handbook/") for h in hits)
    page = await search_tool.execute({"query": query, "source_prefix": "handbook/"}, as_user(BOB))
    sources = [str(h["source"]) for h in page.items if isinstance(h, dict)]
    assert sources and all(source.startswith("handbook/") for source in sources)
    # A filter never widens access: Bob is not in engineering, so its folder is empty for him.
    hidden = await search_tool.execute(
        {"query": query, "source_prefix": "engineering/"}, as_user(BOB)
    )
    assert hidden.total == 0


async def test_reranked_results_keep_quality_on_the_labeled_set(
    search_tool: Tool[SearchDocsInput],
) -> None:
    """C20 chose the local cross-encoder on this set: MRR@10 0.931 against 0.915 without."""
    everyone = User(
        external_id="all",
        display_name="All",
        groups=["employees", "engineering", "finance", "hr", "leadership"],
    )
    total = 0.0
    for case in CASES:
        page = await search_tool.execute(
            {"query": case["query"], "pagination": {"limit": 10}}, as_user(everyone)
        )
        headings = [
            f"{h['title']} > {h['section']}" if h["section"] else h["title"]  # type: ignore[index, call-overload]
            for h in page.items
        ]
        total += next(
            (1 / rank for rank, h in enumerate(headings, 1) if h == case["expected"]), 0.0
        )
    assert total / len(CASES) >= 0.92
