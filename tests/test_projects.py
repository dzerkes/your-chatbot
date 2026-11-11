"""Real project queries and grants in an isolated, seeded database."""

import subprocess
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from chatbot.config import Settings
from chatbot.errors import ToolError
from chatbot.tools.base import ToolContext
from chatbot.tools.pagination import ToolPage
from chatbot.tools.query_projects import QueryProjects
from chatbot.tools.registry import ToolRegistry
from chatbot.types import ToolUse, User
from psycopg import errors, sql
from psycopg.conninfo import make_conninfo
from pydantic import JsonValue

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1]


def compose(*args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def conninfo(settings: Settings, *, reader: bool = False) -> str:
    return make_conninfo(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.ro_db_user if reader else settings.postgres_user,
        password=(
            settings.ro_db_password if reader else settings.postgres_password
        ).get_secret_value(),
    )


@pytest.fixture(scope="module")
def project_database() -> Iterator[Settings]:
    database = f"c10_{uuid4().hex}"
    created = compose(
        "exec",
        "-T",
        "postgres",
        "sh",
        "-c",
        'PGPASSWORD="$POSTGRES_PASSWORD" exec createdb '
        '-h 127.0.0.1 -U postgres -T template0 -O "$OWNER_DB_USER" "$1"',
        "sh",
        database,
    )
    assert created.returncode == 0, created.stderr
    try:
        migrated = compose(
            "run", "--rm", "--no-deps", "-e", f"POSTGRES_DB={database}", "migrate", timeout=180
        )
        assert migrated.returncode == 0, migrated.stderr
        published = compose("port", "postgres", "5432")
        assert published.returncode == 0, published.stderr
        settings = Settings(
            postgres_db=database, postgres_port=int(published.stdout.strip().rsplit(":", 1)[1])
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
            seed = (ROOT / "db/seed/01_demo.sql").read_text()
            conn.execute(seed)
            conn.execute(seed)
            assert conn.execute("SELECT count(*) FROM raw.clients").fetchone() == (4,)
            assert conn.execute("SELECT count(*) FROM raw.projects").fetchone() == (12,)
        yield settings
        downgraded = compose(
            "run",
            "--rm",
            "--no-deps",
            "-e",
            f"POSTGRES_DB={database}",
            "migrate",
            "uv",
            "run",
            "--locked",
            "alembic",
            "-c",
            "db/alembic.ini",
            "downgrade",
            "20261112_01_history",
            timeout=180,
        )
        assert downgraded.returncode == 0, downgraded.stderr
        with psycopg.connect(conninfo(settings)) as conn:
            assert conn.execute(
                "SELECT count(*) FROM pg_namespace WHERE nspname IN ('raw', 'curated')"
            ).fetchone() == (0,)
            assert conn.execute("SELECT to_regclass('public.users') IS NOT NULL").fetchone() == (
                True,
            )
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


def test_reader_has_only_approved_views(project_database: Settings) -> None:
    with psycopg.connect(conninfo(project_database, reader=True), autocommit=True) as conn:
        assert conn.execute("SELECT count(*) FROM curated.projects").fetchone() == (12,)
        assert conn.execute("SELECT count(*) FROM curated.clients").fetchone() == (4,)
        for view in ("curated.clients", "curated.projects"):
            assert conn.execute(
                "SELECT has_table_privilege(current_user, %s, 'INSERT, UPDATE, DELETE, TRUNCATE')",
                (view,),
            ).fetchone() == (False,)
        for statement in (
            "SELECT * FROM raw.clients",
            "SELECT * FROM raw.projects",
            "INSERT INTO raw.clients VALUES (999, 'x', 'x', '')",
            "UPDATE raw.projects SET name = 'changed'",
            "DELETE FROM raw.projects",
            "TRUNCATE raw.projects",
            "UPDATE curated.clients SET name = 'changed'",
            "INSERT INTO curated.clients VALUES (999, 'x')",
            "DELETE FROM curated.clients",
            "INSERT INTO users (external_id, display_name) VALUES ('x', 'x')",
            "CREATE TABLE raw.forbidden (id integer)",
            "CREATE TABLE curated.forbidden (id integer)",
            "CREATE TABLE public.forbidden (id integer)",
            "CREATE TEMP TABLE forbidden (id integer)",
            "CREATE SCHEMA forbidden",
            "DROP VIEW curated.projects",
        ):
            # Roll back even if a permission regression unexpectedly permits a write.
            with conn.transaction(force_rollback=True), pytest.raises(errors.InsufficientPrivilege):
                conn.execute(statement)
        # A joined view rejects updates during rewrite, before checking grants.
        with pytest.raises(errors.ObjectNotInPrerequisiteState):
            conn.execute("UPDATE curated.projects SET name = 'changed'")
        for statement in (
            "SELECT contact_email FROM curated.clients",
            "SELECT budget_usd, internal_notes FROM curated.projects",
        ):
            with pytest.raises(errors.UndefinedColumn):
                conn.execute(statement)


async def test_project_tool_filters_pagination_and_injection(project_database: Settings) -> None:
    context = ToolContext(
        user=User(external_id="alice", display_name="Alice"),
        session_id=uuid4(),
        turn_id=uuid4(),
        trace_id="projects-test",
    )
    async with QueryProjects(project_database) as projects:
        registry = ToolRegistry()
        registry.register(projects.tool())
        assert registry.definitions()[0].name == "query_projects"

        async def query(arguments: dict[str, JsonValue]) -> ToolPage:
            result = await registry.execute(
                ToolUse(id="query-1", name="query_projects", arguments=arguments), context
            )
            return ToolPage.model_validate(result.content)

        first = await query({"status": "active", "pagination": {"limit": 2}})
        assert first.total == 5
        assert first.next_cursor is not None
        pages = [first]
        while pages[-1].next_cursor:
            pages.append(
                await query(
                    {
                        "status": "active",
                        "pagination": {
                            "limit": 2,
                            "cursor": pages[-1].next_cursor,
                        },
                    }
                )
            )
        items = [item for page in pages for item in page.items]
        assert [item["id"] for item in items if isinstance(item, dict)] == [101, 104, 107, 108, 110]
        assert all(
            isinstance(item, dict)
            and set(item)
            == {
                "id",
                "name",
                "client_id",
                "client_name",
                "description",
                "status",
                "start_date",
                "end_date",
            }
            for item in items
        )
        assert first.byte_size() <= 16_384

        filters: dict[str, JsonValue] = {
            "status": "active",
            "client": "NORTHSTAR",
            "client_id": 1,
            "start_date_from": "2026-01-15",
            "start_date_to": "2026-01-15",
            "text": "research",
        }
        combined = await query(filters)
        assert combined.total == 1
        assert isinstance(combined.items[0], dict) and combined.items[0]["id"] == 101
        assert (await query({"text": "portal"})).total == 2
        assert (await query({"client": "harbor"})).total == 3
        assert (
            await query({"start_date_from": "2026-02-01", "start_date_to": "2026-06-01"})
        ).total == 4
        for text in ("%' OR TRUE; DROP TABLE raw.projects; --", "%", "_"):
            assert (await query({"text": text})).items == []
        assert (await query({})).total == 12
        with pytest.raises(ToolError, match="cursor"):
            await query({"status": "completed", "pagination": {"cursor": first.next_cursor}})
