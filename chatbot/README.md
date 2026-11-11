# Chatbot core

This package holds the application code shared by the future CLI, backend and
eval runner. It has no FastAPI, Streamlit or provider SDK dependency. The store
module uses psycopg for Postgres history persistence.
The agent loop and `respond()` will be implemented in C11.

## Settings and logging

From the repository root, run `uv sync` to install the workspace. Construct
settings at application startup and pass them to consumers:

```python
import logging
from uuid import uuid4

from chatbot.config import Settings
from chatbot.logging import configure_logging, log_context

settings = Settings()
configure_logging(settings)
logger = logging.getLogger(__name__)

with log_context(session_id=uuid4(), turn_id=uuid4(), trace_id="example-trace"):
    logger.info("Turn started")
```

Settings read `.env` from the working directory and support all application
variables in the root `.env.example`. Constructor arguments override environment
variables, which override `.env`, which overrides defaults. Empty environment
values use defaults. `LLM_FALLBACKS` accepts comma-separated `provider:model`
names. Compose-only values such as `POSTGRES_ADMIN_PASSWORD` are ignored.
Secrets use Pydantic's `SecretStr`; call `get_secret_value()` only when passing
one to a database client or SDK. Importing the package does not load settings.

Logging uses the Python standard library. `configure_logging(settings)` sets
the process's root logging handler and level, so call it once at startup.
`LOG_FORMAT` selects readable `console` output or one JSON object per record.
Both formats include `session_id`, `turn_id` and `trace_id` (null/None outside a
scope). `log_context` inherits omitted IDs in nested scopes, restores the previous
scope even on exceptions, and isolates concurrent asyncio tasks. Reconfiguring
replaces the previous handler instead of adding duplicate output.

## Domain objects

`chatbot.types` defines validated, provider-neutral Pydantic models:

- `User`: an internal UUID, external identity, display name and groups.
- `Message`: session/turn IDs, sequence, role, content, optional model and usage,
  and a timezone-aware timestamp. Content is a list of tagged `Text`, `ToolUse`
  and `ToolResult` blocks. A result references its call through `tool_use_id`.
- `Usage`: input/output tokens and cache-read/cache-creation token counts.
  Input tokens include cached tokens; cache counts describe subsets and are
  not added again by `total_tokens`.
- `Event`: the tagged union `TextDelta`, `ToolCallStarted`, `ToolCallFinished`,
  `TurnCompleted` and `TurnFailed`. Every event carries session/turn IDs and an
  optional trace ID. Completion includes usage and cost; failure includes a code,
  safe message and retryability.

Use `model_dump_json()` / `model_validate_json()` to round-trip models. For the
`Event` union, use `pydantic.TypeAdapter(Event)` to serialize and validate it.
This shape will become the persistence and streaming contract in later commits.

`chatbot.errors` provides `ChatbotError` and the specific `ProviderError`,
`ToolError`, `ContextOverflow` and `StepLimitExceeded` exceptions. Each has a
stable `code` and a `retryable` flag. Error messages must be safe to show users;
chain underlying SDK exceptions with `raise ... from ...` when needed.

## History stores

`chatbot.store.base` defines `SessionStore` and `MessageStore` protocols and the
`Session` and `Turn` records. `HistoryStore` combines both protocols. The two
implementations are `MemoryStore` and `PostgresStore`:

```python
from chatbot.config import Settings
from chatbot.store.base import Turn
from chatbot.store.postgres import PostgresStore
from chatbot.types import Message, Text, User


async def example() -> None:
    async with PostgresStore(Settings()) as store:
        user = await store.upsert_user(User(external_id="alice", display_name="Alice"))
        session = await store.create_session(user.id, title="New chat")
        turn = Turn(session_id=session.id, status="completed")
        message = Message(
            session_id=session.id,
            turn_id=turn.id,
            sequence=0,
            role="user",
            content=[Text(text="Hello")],
        )
        await store.save_turn(turn, [message], user.id)
```

`PostgresStore` opens an async connection pool when entered and closes it on
exit. It connects as `APP_DB_USER`, never as the migration owner. Construct
`MemoryStore()` directly for tests; it needs no database or lifecycle step.

`upsert_user` returns the existing UUID when an external identity already
exists and refreshes its display name and groups. `get_session`, `list_sessions`,
`list_turns`, `list_messages` and `save_turn` require the owner's UUID. A caller
cannot read or add a turn to someone else's session. `update_session` changes
title and summary, using the session's owner ID to scope the update.

`save_turn` inserts the turn and all messages in one transaction and updates
the session's activity time. A conflict or error rolls back the whole batch.
Message sequence numbers are unique across a session and supplied by the caller;
the repository rejects duplicate values. Tool calls and results stay as
provider-neutral content blocks in JSONB. Both implementations return the same
domain objects. `MemoryStore` makes copies so callers cannot mutate saved data.
