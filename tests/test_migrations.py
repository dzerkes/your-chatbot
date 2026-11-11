"""Exercise an Alembic upgrade and downgrade in a disposable database."""

import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

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


def sql(role: str, database: str, statement: str) -> subprocess.CompletedProcess[str]:
    prefixes = {"owner": "OWNER", "app": "APP", "ro": "RO"}
    prefix = prefixes[role]
    return compose(
        "exec",
        "-T",
        "-e",
        f"POSTGRES_DB={database}",
        "postgres",
        "sh",
        "-c",
        f'PGPASSWORD="${prefix}_DB_PASSWORD" exec psql -X -h 127.0.0.1 '
        f'-U "${prefix}_DB_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 -Atqc "$1"',
        "sh",
        statement,
    )


def query(role: str, database: str, statement: str) -> str:
    result = sql(role, database, statement)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def denied(role: str, database: str, statement: str, reason: str) -> None:
    result = sql(role, database, statement)
    assert result.returncode != 0, result.stdout
    assert reason in result.stderr


@pytest.mark.integration
def test_history_migration_and_permissions() -> None:
    database = f"c05_{uuid4().hex}"
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
        upgraded = compose(
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
            "upgrade",
            "20261112_01_history",
            timeout=180,
        )
        assert upgraded.returncode == 0, upgraded.stderr
        assert (
            query("owner", database, "SELECT version_num FROM alembic_version")
            == "20261112_01_history"
        )
        assert (
            query(
                "owner",
                database,
                "SELECT string_agg(tablename, ',' ORDER BY tablename) FROM pg_tables "
                "WHERE schemaname = 'public' AND tablename IN "
                "('users', 'sessions', 'turns', 'messages')",
            )
            == "messages,sessions,turns,users"
        )

        user, session, other_session, turn = (str(uuid4()) for _ in range(4))
        query(
            "app",
            database,
            f"INSERT INTO users (id, external_id, display_name, groups) "
            f"VALUES ('{user}', 'test-user', 'Test User', ARRAY['engineering']); "
            f"INSERT INTO sessions (id, owner_id, title) "
            f"VALUES ('{session}', '{user}', 'Project chat'), ('{other_session}', '{user}', NULL); "
            f"INSERT INTO turns (id, session_id) VALUES ('{turn}', '{session}'); "
            f"INSERT INTO messages (session_id, turn_id, sequence, role, content, model) "
            f"VALUES ('{session}', '{turn}', 0, 'assistant', "
            '\'[{"type":"tool_use","id":"call1","name":"query_projects",'
            '"arguments":{"status":"active"}}]\'::jsonb, \'fake:scripted\'), '
            f"('{session}', '{turn}', 1, 'tool', "
            '\'[{"type":"tool_result","tool_use_id":"call1",'
            '"content":{"items":[{"id":1}],"total":1}}]\'::jsonb, NULL)',
        )
        assert (
            query(
                "app",
                database,
                f"SELECT string_agg(content->0->>'type', ',' ORDER BY sequence) "
                f"FROM messages WHERE session_id = '{session}'",
            )
            == "tool_use,tool_result"
        )
        assert (
            query(
                "app",
                database,
                f"UPDATE turns SET status = 'completed', input_tokens = 12, output_tokens = 4, "
                f"cost_usd = 0.0012, latency_ms = 50, trace_id = 'trace-1' WHERE id = '{turn}'; "
                "SELECT status || ':' || input_tokens || ':' || output_tokens "
                f"FROM turns WHERE id = '{turn}'",
            )
            == "completed:12:4"
        )

        denied("ro", database, "SELECT id FROM users", "permission denied")
        denied("app", database, "SELECT version_num FROM alembic_version", "permission denied")
        denied("app", database, "CREATE TABLE public.forbidden (id integer)", "permission denied")
        denied(
            "app",
            database,
            f"INSERT INTO messages (session_id, turn_id, sequence, role, content) "
            f"VALUES ('{other_session}', '{turn}', 2, 'tool', '[]'::jsonb)",
            "foreign key constraint",
        )
        denied(
            "app",
            database,
            f"INSERT INTO messages (session_id, turn_id, sequence, role, content) "
            f"VALUES ('{session}', '{turn}', 0, 'tool', '[]'::jsonb)",
            "unique constraint",
        )
        denied(
            "app",
            database,
            f"INSERT INTO messages (session_id, turn_id, sequence, role, content) "
            f"VALUES ('{session}', '{turn}', 2, 'tool', '{{}}'::jsonb)",
            "check constraint",
        )
        query("app", database, f"DELETE FROM users WHERE id = '{user}'")
        assert query("app", database, "SELECT count(*) FROM messages") == "0"

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
            "base",
            timeout=180,
        )
        assert downgraded.returncode == 0, downgraded.stderr
        assert (
            query(
                "owner",
                database,
                "SELECT count(*) FROM pg_tables WHERE schemaname = 'public' "
                "AND tablename IN ('users', 'sessions', 'turns', 'messages')",
            )
            == "0"
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
