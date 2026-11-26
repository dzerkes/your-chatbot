"""Compose startup job: upgrade the schema and optionally load synthetic fixtures."""

from pathlib import Path

import psycopg
from alembic import command
from alembic.config import Config
from chatbot.config import Settings
from psycopg.conninfo import make_conninfo


def main() -> None:
    settings = Settings()
    directory = Path(__file__).resolve().parent
    command.upgrade(Config(str(directory / "alembic.ini")), "head")
    if settings.seed_demo_data:
        with psycopg.connect(
            make_conninfo(
                host=settings.postgres_host,
                port=settings.postgres_port,
                dbname=settings.postgres_db,
                user=settings.postgres_user,
                password=settings.postgres_password.get_secret_value(),
            ),
            autocommit=True,
        ) as connection:
            connection.execute((directory / "seed/01_demo.sql").read_text(encoding="utf-8"))
        print("Loaded synthetic demo clients and projects.")


if __name__ == "__main__":
    main()
