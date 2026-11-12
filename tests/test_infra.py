"""Compose configuration checks and opt-in tests against `make up`."""

import json
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[1]


def compose(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.unit
def test_compose_configuration() -> None:
    if not shutil.which("docker"):
        pytest.skip("Docker CLI is required to validate Compose configuration")
    result = compose("--env-file", ".env.example", "config", "--format", "json")
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    postgres = config["services"]["postgres"]
    env = postgres["environment"]
    assert postgres["image"] == "pgvector/pgvector:0.8.2-pg17"
    assert env["POSTGRES_USER"] == "postgres"
    assert env["OWNER_DB_USER"] == "chatbot_owner"
    assert env["APP_DB_USER"] == "chatbot_app"
    assert env["RO_DB_USER"] == "chatbot_ro"
    assert postgres["ports"][0]["host_ip"] == "127.0.0.1"
    assert any(v["type"] == "volume" for v in postgres["volumes"])
    assert any(
        v["target"] == "/docker-entrypoint-initdb.d" and v["read_only"] for v in postgres["volumes"]
    )
    assert "pg_isready -h 127.0.0.1" in postgres["healthcheck"]["test"][1]
    migrate = config["services"]["migrate"]
    backend = config["services"]["backend"]
    frontend = config["services"]["frontend"]
    assert migrate["build"]["target"] == "migrate" and not migrate.get("profiles")
    assert migrate["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert backend["depends_on"]["migrate"]["condition"] == "service_completed_successfully"
    assert frontend["depends_on"]["backend"]["condition"] == "service_healthy"
    assert backend["environment"]["POSTGRES_HOST"] == "postgres"
    assert backend["environment"]["POSTGRES_PORT"] == "5432"
    assert "POSTGRES_PASSWORD" not in backend["environment"]
    assert "POSTGRES_ADMIN_PASSWORD" not in backend["environment"]
    assert frontend["environment"] == {
        "BACKEND_URL": "http://backend:8000",
        "FRONTEND_PORT": "8501",
    }
    assert all(service["ports"][0]["host_ip"] == "127.0.0.1" for service in (backend, frontend))
    assert all(
        service.get("healthcheck") and not service.get("volumes") for service in (backend, frontend)
    )


@pytest.mark.unit
def test_compose_backend_environment_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compose's dotenv keeps inline comments on empty keys, which Settings must still accept."""
    if not shutil.which("docker"):
        pytest.skip("Docker CLI is required to validate Compose configuration")
    result = compose("--env-file", ".env.example", "config", "--format", "json")
    assert result.returncode == 0, result.stderr
    environment = json.loads(result.stdout)["services"]["backend"]["environment"]
    for name, value in environment.items():
        monkeypatch.setenv(name, value or "")
    from chatbot.config import Settings

    Settings(_env_file=None)  # type: ignore[call-arg]


def sql(role: str, statement: str) -> subprocess.CompletedProcess[str]:
    # Fixed role selectors only; credentials stay inside the container and out of argv.
    prefixes = {"owner": "OWNER", "app": "APP", "ro": "RO"}
    prefix = prefixes[role]
    return compose(
        "exec",
        "-T",
        "postgres",
        "sh",
        "-c",
        f'PGPASSWORD="${prefix}_DB_PASSWORD" exec psql -X -h 127.0.0.1 '
        f'-U "${prefix}_DB_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 -Atc "$1"',
        "sh",
        statement,
    )


def query(role: str, statement: str) -> str:
    result = sql(role, statement)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.integration
@pytest.mark.parametrize("role", ["owner", "app", "ro"])
def test_roles_can_authenticate_without_admin_privileges(role: str) -> None:
    assert (
        query(
            role,
            "SELECT rolcanlogin AND NOT rolsuper AND NOT rolcreatedb "
            "AND NOT rolcreaterole AND NOT rolreplication AND NOT rolbypassrls "
            "FROM pg_roles WHERE rolname = current_user",
        )
        == "t"
    )


@pytest.mark.integration
def test_vector_extension() -> None:
    assert query("owner", "SELECT '[1,2,3]'::vector <-> '[1,2,3]'::vector") == "0"


@pytest.mark.integration
def test_read_only_timeout() -> None:
    assert query("ro", "SHOW statement_timeout") == "5s"
    result = sql("ro", "SET statement_timeout = '50ms'; SELECT pg_sleep(1)")
    assert result.returncode != 0
    assert "statement timeout" in result.stderr


@pytest.mark.integration
@pytest.mark.parametrize(
    "statement",
    [
        "CREATE TABLE public.c03_forbidden (id integer)",
        "CREATE TEMP TABLE c03_forbidden (id integer)",
        "CREATE SCHEMA c03_forbidden",
    ],
)
def test_read_only_cannot_create_objects(statement: str) -> None:
    # Roll back even if a permission regression accidentally allows the operation.
    result = sql("ro", f"BEGIN; {statement}; ROLLBACK;")
    assert result.returncode != 0
    assert "permission denied" in result.stderr


@pytest.mark.integration
def test_owner_can_migrate_and_read_only_cannot_write() -> None:
    schema = f"c03_test_{uuid4().hex}"
    # Resolve configured names on the server instead of assuming default usernames.
    result = compose(
        "exec",
        "-T",
        "postgres",
        "printenv",
        "RO_DB_USER",
    )
    assert result.returncode == 0, result.stderr
    ro_identifier = '"' + result.stdout.strip().replace('"', '""') + '"'
    try:
        query(
            "owner",
            f"CREATE SCHEMA {schema}; CREATE TABLE {schema}.probe (id integer); "
            f"INSERT INTO {schema}.probe VALUES (1); "
            f"GRANT USAGE ON SCHEMA {schema} TO {ro_identifier}; "
            f"GRANT SELECT ON {schema}.probe TO {ro_identifier}",
        )
        assert query("ro", f"SELECT id FROM {schema}.probe") == "1"
        for statement in (
            f"INSERT INTO {schema}.probe VALUES (2)",
            f"UPDATE {schema}.probe SET id = 2",
            f"DELETE FROM {schema}.probe",
            f"TRUNCATE {schema}.probe",
            f"CREATE TABLE {schema}.forbidden (id integer)",
        ):
            result = sql("ro", f"BEGIN; {statement}; ROLLBACK;")
            assert result.returncode != 0
            assert "permission denied" in result.stderr
    finally:
        query("owner", f"DROP SCHEMA IF EXISTS {schema} CASCADE")
