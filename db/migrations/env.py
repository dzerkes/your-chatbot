"""Connect Alembic as the migration owner using the shared Settings class."""

from logging.config import fileConfig

from alembic import context
from chatbot.config import Settings
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.pool import NullPool

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)


def database_url(settings: Settings) -> URL:
    """URL.create handles punctuation in passwords without manual URL escaping."""
    return URL.create(
        "postgresql+psycopg",
        username=settings.postgres_user,
        password=settings.postgres_password.get_secret_value(),
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
    )


def run_migrations_online() -> None:
    settings = Settings()
    engine = create_engine(database_url(settings), poolclass=NullPool)
    try:
        with engine.connect() as connection:
            context.configure(connection=connection, target_metadata=None)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


def run_migrations_offline() -> None:
    settings = Settings()
    context.configure(
        url=database_url(settings),
        target_metadata=None,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
