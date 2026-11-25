"""Document ingestion, idempotency and grants in an isolated, migrated database."""

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg
import pytest
from chatbot.config import Settings
from chatbot.errors import IngestError
from chatbot.rag.chunking import chunk_markdown
from chatbot.rag.embeddings import FakeEmbedder
from chatbot.rag.ingest import ingest_documents, load_documents
from psycopg import errors, sql
from psycopg.conninfo import make_conninfo

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "db/seed/documents"


def compose(*args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def conninfo(settings: Settings, role: str = "owner") -> str:
    user, password = {
        "owner": (settings.postgres_user, settings.postgres_password),
        "app": (settings.app_db_user, settings.app_db_password),
        "reader": (settings.ro_db_user, settings.ro_db_password),
    }[role]
    return make_conninfo(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=user,
        password=password.get_secret_value(),
    )


def rows(settings: Settings, query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    with psycopg.connect(conninfo(settings)) as conn:
        return conn.execute(query, params).fetchall()


@pytest.fixture(scope="module")
def rag_database() -> Iterator[Settings]:
    database = f"c18_{uuid4().hex}"
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
            embedding_model="fake:hash",
        )
        with psycopg.connect(conninfo(settings), autocommit=True) as conn:
            # Match the database-level grants applied by init/01_roles.sql in development.
            conn.execute(
                sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(sql.Identifier(database))
            )
            conn.execute(
                sql.SQL("GRANT CONNECT ON DATABASE {} TO {}, {}").format(
                    sql.Identifier(database),
                    sql.Identifier(settings.app_db_user),
                    sql.Identifier(settings.ro_db_user),
                )
            )
        yield settings
        downgraded = compose(
            "run",
            "--rm",
            "--no-deps",
            "-e",
            f"POSTGRES_DB={database}",
            "migrate",
            "alembic",
            "-c",
            "db/alembic.ini",
            "downgrade",
            "20261112_02_projects",
            timeout=180,
        )
        assert downgraded.returncode == 0, downgraded.stderr
        assert rows(settings, "SELECT count(*) FROM pg_namespace WHERE nspname = 'rag'") == [(0,)]
        assert rows(settings, "SELECT to_regclass('curated.projects') IS NOT NULL") == [(True,)]
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


@pytest.fixture
def db(rag_database: Settings) -> Settings:
    """Each test starts from an empty index."""
    with psycopg.connect(conninfo(rag_database), autocommit=True) as conn:
        conn.execute("TRUNCATE rag.documents CASCADE")
    return rag_database


def snapshot(settings: Settings) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    documents = rows(
        settings,
        "SELECT id, source, content_hash, allowed_groups, created_at, updated_at "
        "FROM rag.documents ORDER BY source",
    )
    chunks = rows(
        settings,
        "SELECT id, document_id, position, heading, text, embedding::text, allowed_groups "
        "FROM rag.chunks ORDER BY document_id, position",
    )
    return documents, chunks


def write(directory: Path, name: str, groups: str, body: str) -> None:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\ngroups: {groups}\n---\n{body}\n")


def run_cli(settings: Settings, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "POSTGRES_DB": settings.postgres_db,
        "POSTGRES_PORT": str(settings.postgres_port),
        "EMBEDDING_MODEL": "fake:hash",
        "TRACING_ENABLED": "false",
    }
    return subprocess.run(
        [sys.executable, "-m", "chatbot.rag.ingest", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_cli_loads_seed_documents_and_a_second_run_changes_nothing(db: Settings) -> None:
    documents = load_documents(SEED)
    expected_chunks = sum(len(chunk_markdown(d.body, title=d.title)) for d in documents)

    first = run_cli(db, "db/seed/documents")
    assert first.returncode == 0, first.stderr
    assert first.stdout.strip() == (
        f"7 documents: 7 added, 0 updated, 0 unchanged; {expected_chunks} chunks written"
    )
    before = snapshot(db)

    second = run_cli(db, "db/seed/documents")
    assert second.returncode == 0, second.stderr
    assert second.stdout.strip() == "7 documents: 0 added, 0 updated, 7 unchanged; 0 chunks written"
    assert snapshot(db) == before

    assert rows(db, "SELECT count(*) FROM rag.chunks") == [(expected_chunks,)]
    # Chunks carry their document's groups, and the generated text-search column is filled.
    assert rows(
        db,
        "SELECT count(*) FROM rag.chunks AS c JOIN rag.documents AS d ON d.id = c.document_id "
        "WHERE c.allowed_groups <> d.allowed_groups OR c.search_vector = ''::tsvector",
    ) == [(0,)]
    assert rows(
        db,
        "SELECT allowed_groups, embedding_model FROM rag.documents WHERE source = %s",
        ("finance/pricing-guidelines-2026.md",),
    ) == [(["finance", "leadership"], "fake:hash")]
    assert rows(
        db,
        "SELECT DISTINCT d.source FROM rag.chunks AS c JOIN rag.documents AS d "
        "ON d.id = c.document_id WHERE c.search_vector @@ websearch_to_tsquery('english', %s)",
        ("sick leave doctor",),
    ) == [("handbook/employee-handbook.md",)]


async def test_vectors_support_nearest_neighbour_search(db: Settings) -> None:
    embedder = FakeEmbedder(db.embedding_dim)
    await ingest_documents(load_documents(SEED), db, embedder)
    query = await embedder.embed_query("roll back the canary release deployment pipeline")
    literal = "[" + ",".join(str(x) for x in query) + "]"
    nearest = rows(
        db,
        "SELECT d.source FROM rag.chunks AS c JOIN rag.documents AS d ON d.id = c.document_id "
        "ORDER BY c.embedding <=> %s::vector LIMIT 1",
        (literal,),
    )
    assert nearest == [("engineering/deployment-runbook.md",)]


async def test_changed_document_replaces_only_its_own_chunks(db: Settings, tmp_path: Path) -> None:
    embedder = FakeEmbedder(db.embedding_dim)
    write(tmp_path, "a.md", "team", "# A\n\nFirst version.")
    write(tmp_path, "b.md", "team", "# B\n\nStays the same.")
    report = await ingest_documents(load_documents(tmp_path), db, embedder)
    assert (report.added, report.chunks_written) == (["a.md", "b.md"], 2)
    (doc_a, *_), (doc_b, *_) = rows(
        db, "SELECT id, created_at, updated_at FROM rag.documents ORDER BY source"
    )
    chunk_ids = dict(rows(db, "SELECT document_id, array_agg(id) FROM rag.chunks GROUP BY 1"))
    stamps = dict(rows(db, "SELECT source, (created_at, updated_at)::text FROM rag.documents"))

    write(tmp_path, "a.md", "team, finance", "# A\n\nSecond version.\n\n## More\n\nNew section.")
    report = await ingest_documents(load_documents(tmp_path), db, embedder)
    assert (report.updated, report.unchanged, report.chunks_written) == (["a.md"], ["b.md"], 2)

    new_ids = dict(rows(db, "SELECT document_id, array_agg(id) FROM rag.chunks GROUP BY 1"))
    assert new_ids[doc_b] == chunk_ids[doc_b]
    assert not set(new_ids[doc_a]) & set(chunk_ids[doc_a])
    assert rows(
        db,
        "SELECT position, heading, text, allowed_groups FROM rag.chunks "
        "WHERE document_id = %s ORDER BY position",
        (doc_a,),
    ) == [
        (0, "A", "Second version.", ["finance", "team"]),
        (1, "A > More", "New section.", ["finance", "team"]),
    ]
    after = dict(rows(db, "SELECT source, (created_at, updated_at)::text FROM rag.documents"))
    assert after["b.md"] == stamps["b.md"]
    assert after["a.md"] != stamps["a.md"]


async def test_prune_removes_documents_whose_file_is_gone(db: Settings, tmp_path: Path) -> None:
    embedder = FakeEmbedder(db.embedding_dim)
    write(tmp_path, "keep.md", "team", "Keep me.")
    write(tmp_path, "drop.md", "team", "Drop me.")
    await ingest_documents(load_documents(tmp_path), db, embedder)
    (tmp_path / "drop.md").unlink()

    kept = await ingest_documents(load_documents(tmp_path), db, embedder)
    assert kept.removed == []
    assert rows(db, "SELECT count(*) FROM rag.documents") == [(2,)]

    pruned = await ingest_documents(load_documents(tmp_path), db, embedder, prune=True)
    assert pruned.removed == ["drop.md"]
    assert rows(db, "SELECT source FROM rag.documents") == [("keep.md",)]
    assert rows(db, "SELECT count(*) FROM rag.chunks") == [(1,)]


async def test_embedder_dimension_must_match_the_schema(db: Settings) -> None:
    with pytest.raises(IngestError, match=f"stores {db.embedding_dim}-dimension vectors"):
        await ingest_documents(load_documents(SEED), db, FakeEmbedder(8))
    assert rows(db, "SELECT count(*) FROM rag.documents") == [(0,)]


def test_reader_can_only_read_and_app_cannot_see_rag(db: Settings) -> None:
    with psycopg.connect(conninfo(db, "reader"), autocommit=True) as conn:
        assert conn.execute("SELECT count(*) FROM rag.chunks").fetchone() == (0,)
        assert conn.execute("SELECT count(*) FROM rag.documents").fetchone() == (0,)
        for statement in (
            "INSERT INTO rag.documents (source, title, allowed_groups, content_hash, "
            "embedding_model) VALUES ('x', 'x', '{a}', 'h', 'fake:hash')",
            "UPDATE rag.documents SET title = 'x'",
            "DELETE FROM rag.chunks",
            "TRUNCATE rag.chunks",
            "CREATE TABLE rag.extra (id int)",
        ):
            with pytest.raises(errors.InsufficientPrivilege):
                conn.execute(statement)
    with (
        psycopg.connect(conninfo(db, "app"), autocommit=True) as conn,
        pytest.raises(errors.InsufficientPrivilege),
    ):
        conn.execute("SELECT count(*) FROM rag.chunks")


def test_schema_has_search_indexes_and_rejects_ungrouped_rows(db: Settings) -> None:
    indexes = dict(rows(db, "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'rag'"))
    assert "USING hnsw (embedding vector_cosine_ops)" in indexes["ix_chunks_embedding"]
    assert "USING gin (search_vector)" in indexes["ix_chunks_search_vector"]
    assert "USING gin (allowed_groups)" in indexes["ix_chunks_allowed_groups"]
    assert "USING gin (allowed_groups)" in indexes["ix_documents_allowed_groups"]
    with psycopg.connect(conninfo(db)) as conn, pytest.raises(errors.CheckViolation):
        conn.execute(
            "INSERT INTO rag.documents (source, title, allowed_groups, content_hash, "
            "embedding_model) VALUES ('x', 'x', '{}', 'h', 'fake:hash')"
        )
