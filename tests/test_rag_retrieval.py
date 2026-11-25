"""Hybrid retrieval over the seed documents in an isolated database.

Documents are embedded with the configured local model, because the labeled-set
comparison is only meaningful with real embeddings. The model downloads once
into EMBEDDING_CACHE_DIR.
"""

import json
import subprocess
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
import pytest_asyncio
from chatbot.config import Settings
from chatbot.rag.base import Candidate, SearchFilters
from chatbot.rag.embeddings import build_embedder
from chatbot.rag.ingest import ingest_documents, load_documents
from chatbot.rag.retriever import HybridRetriever
from psycopg import sql
from psycopg.conninfo import make_conninfo

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "tests/fixtures/retrieval_labeled_set.json").read_text())["cases"]
EVERYONE = ("employees", "engineering", "finance", "hr", "leadership")
GROUP_SETS = [("employees",), ("engineering",), ("finance",), ("hr", "employees"), EVERYONE]
RESTRICTED_QUERIES = [
    "salary bands level 4 principal director",
    "day rates discounts minimum margin pricing",
    "how much does a senior engineer earn",
]


def compose(*args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
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
def retrieval_database() -> Iterator[Settings]:
    database = f"c19_{uuid4().hex}"
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
            "run",
            "--rm",
            "--no-deps",
            "-e",
            f"POSTGRES_DB={database}",
            "-e",
            "SEED_DEMO_DATA=false",
            "migrate",
            timeout=180,
        )
        assert migrated.returncode == 0, migrated.stderr
        published = compose("port", "postgres", "5432")
        assert published.returncode == 0, published.stderr
        settings = Settings(
            postgres_db=database,
            postgres_port=int(published.stdout.strip().rsplit(":", 1)[1]),
            embedding_cache_dir=str(ROOT / ".cache/fastembed"),
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


@dataclass
class Index:
    settings: Settings
    retriever: HybridRetriever
    groups_by_chunk: dict[int, set[str]]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def index(retrieval_database: Settings) -> AsyncIterator[Index]:
    embedder = build_embedder(retrieval_database)
    try:
        await ingest_documents(
            load_documents(ROOT / "db/seed/documents"), retrieval_database, embedder
        )
        with psycopg.connect(owner(retrieval_database)) as conn:
            groups = {
                chunk_id: set(allowed)
                for chunk_id, allowed in conn.execute("SELECT id, allowed_groups FROM rag.chunks")
            }
        async with HybridRetriever(retrieval_database, embedder) as retriever:
            yield Index(retrieval_database, retriever, groups)
    finally:
        await embedder.aclose()


async def all_results(index: Index, query: str, filters: SearchFilters) -> list[Candidate]:
    retriever = index.retriever
    hybrid = [h.candidate for h in await retriever.search(query, filters)]
    lexical = await retriever.search_leg(retriever.lexical, query, filters)
    vector = await retriever.search_leg(retriever.vector, query, filters)
    return hybrid + lexical + vector


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize("groups", GROUP_SETS, ids=lambda groups: "+".join(groups))
async def test_no_leg_returns_a_chunk_outside_the_callers_groups(
    index: Index, groups: tuple[str, ...]
) -> None:
    allowed = set(groups)
    returned = 0
    for query in [case["query"] for case in CASES] + RESTRICTED_QUERIES:
        for item in await all_results(index, query, SearchFilters(groups=groups)):
            returned += 1
            assert index.groups_by_chunk[item.chunk_id] & allowed, (query, item.source)
    assert returned > 0  # every group set can see something


@pytest.mark.asyncio(loop_scope="module")
async def test_a_caller_without_groups_sees_nothing(index: Index) -> None:
    for query in RESTRICTED_QUERIES:
        assert await all_results(index, query, SearchFilters(groups=())) == []
        assert await all_results(index, query, SearchFilters(groups=("nobody",))) == []


@pytest.mark.asyncio(loop_scope="module")
async def test_access_filter_runs_before_ranking(index: Index) -> None:
    """Restricted chunks are the best matches, yet the caller still gets a full list."""
    query = RESTRICTED_QUERIES[0]
    everyone = await index.retriever.search(query, SearchFilters(groups=EVERYONE))
    assert everyone[0].candidate.source == "hr/salary-bands.md"
    employees = await index.retriever.search(query, SearchFilters(groups=("employees",)))
    visible = sum("employees" in groups for groups in index.groups_by_chunk.values())
    # The vector leg ranks every chunk, so a filter applied after ranking would
    # return fewer rows; filtering inside the query returns every visible chunk.
    nearest = await index.retriever.search_leg(
        index.retriever.vector, query, SearchFilters(groups=("employees",))
    )
    assert len(nearest) == min(visible, index.retriever.candidates)
    assert employees and all(
        not h.candidate.source.startswith(("hr/", "finance/")) for h in employees
    )


@pytest.mark.asyncio(loop_scope="module")
async def test_permitted_group_finds_its_restricted_document(index: Index) -> None:
    hits = await index.retriever.search(
        "what is our day rate for an architect", SearchFilters(groups=("finance",))
    )
    assert hits[0].candidate.heading == "Pricing guidelines 2026 > Day rates"
    assert set(hits[0].ranks) == {"lexical", "vector"}


@pytest.mark.asyncio(loop_scope="module")
async def test_metadata_filters_apply_in_both_legs(index: Index) -> None:
    filters = SearchFilters(groups=EVERYONE, source_prefix="engineering/")
    results = await all_results(index, "what to do with client data and leave", filters)
    assert results and {item.source.split("/")[0] for item in results} == {"engineering"}
    tagged = await all_results(
        index, "policy", SearchFilters(groups=EVERYONE, tags=("policy", "security"))
    )
    assert tagged and {item.source for item in tagged} == {"security/data-handling-policy.md"}


@pytest.mark.asyncio(loop_scope="module")
async def test_query_text_is_data_not_sql(index: Index) -> None:
    for query in ["'); DROP TABLE rag.chunks; --", "a & b | !c", "<-> :* \\"]:
        await all_results(index, query, SearchFilters(groups=EVERYONE))
    with psycopg.connect(owner(index.settings)) as conn:
        assert conn.execute("SELECT count(*) > 0 FROM rag.chunks").fetchone() == (True,)


def reciprocal_rank(headings: list[str], expected: str, depth: int = 10) -> float:
    return next((1 / rank for rank, h in enumerate(headings[:depth], 1) if h == expected), 0.0)


@pytest.mark.asyncio(loop_scope="module")
async def test_hybrid_on_the_labeled_set(index: Index) -> None:
    """Hybrid keeps every answer within reach of the reranker and wins on exact terms.

    On this small corpus the vector leg alone ranks paraphrases best, so hybrid
    is not required to win overall MRR (see the C19 notes in the project plan).
    The floors sit just under the measured values to catch regressions.
    """
    retriever, filters = index.retriever, SearchFilters(groups=EVERYONE)
    ranks: dict[str, list[tuple[str, float, bool]]] = {"lexical": [], "vector": [], "hybrid": []}
    for case in CASES:
        query, expected, kind = case["query"], case["expected"], case["kind"]
        results = {
            "lexical": [
                c.heading for c in await retriever.search_leg(retriever.lexical, query, filters)
            ],
            "vector": [
                c.heading for c in await retriever.search_leg(retriever.vector, query, filters)
            ],
            "hybrid": [h.candidate.heading for h in await retriever.search(query, filters)],
        }
        for leg, headings in results.items():
            ranks[leg].append(
                (kind, reciprocal_rank(headings, expected), expected in headings[:10])
            )

    def mrr(leg: str, kind: str | None = None) -> float:
        values = [rr for k, rr, _ in ranks[leg] if kind in (None, k)]
        return sum(values) / len(values)

    def recall(leg: str) -> float:
        return sum(found for _, _, found in ranks[leg]) / len(ranks[leg])

    report = {
        leg: {
            "mrr": round(mrr(leg), 3),
            "exact": round(mrr(leg, "exact"), 3),
            "paraphrase": round(mrr(leg, "paraphrase"), 3),
            "recall@10": round(recall(leg), 3),
        }
        for leg in ranks
    }
    assert recall("hybrid") >= max(recall("lexical"), recall("vector")), report
    assert mrr("hybrid", "exact") >= max(mrr("lexical", "exact"), mrr("vector", "exact")), report
    assert mrr("hybrid") > mrr("lexical"), report
    assert recall("hybrid") >= 0.95 and mrr("hybrid", "exact") >= 0.95, report
    assert mrr("hybrid") >= 0.88, report
