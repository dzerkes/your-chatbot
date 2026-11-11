# Chatbot core

This package holds the application code shared by the future CLI, backend and
eval runner. It has no FastAPI or Streamlit dependency. The store module uses
psycopg for Postgres history persistence; the Anthropic and OpenAI adapters use
their SDKs.
`build_chatbot(settings)` assembles the agent, history store, provider and tools.

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

## LLM providers

`chatbot.llm.base` defines `LLMProvider.stream(request)`, a provider-neutral
async stream. `LLMRequest` contains the system prompt, message history, tool
definitions, output token limit and optional timeout. A stream yields text
deltas and completed tool calls, followed by cumulative usage and a stop reason.
The adapters convert provider token counts into the shared `Usage` shape.

```python
from chatbot.config import Settings
from chatbot.llm.base import LLMRequest, LLMTextDelta
from chatbot.llm.registry import ProviderRegistry


async def stream_reply(request: LLMRequest) -> None:
    settings = Settings()
    provider = ProviderRegistry(settings).get_primary()
    try:
        async for event in provider.stream(request):
            if isinstance(event, LLMTextDelta):
                print(event.text, end="", flush=True)
    finally:
        await provider.aclose()
```

Set `LLM_PRIMARY=anthropic:claude-sonnet-5-5` with `ANTHROPIC_API_KEY`, or
`LLM_PRIMARY=openai:gpt-5.5` with `OPENAI_API_KEY`. `ProviderRegistry.get(name)`
also accepts an explicit `provider:model` name. Close the selected adapter after
use. The OpenAI adapter uses the Responses API with `store=False`; tool errors
are sent as JSON objects with an `error` field.

`ScriptedProvider` accepts one sequence of events or exceptions per request and
records copies of requests for assertions. It needs no API key. Adapter unit
tests use recorded stream events. To run the Anthropic live test, set
`ANTHROPIC_API_KEY` and run `uv run pytest -m llm tests/test_llm_live.py`.
The default pytest run excludes `llm` tests.

## Tools

`Tool` wraps a typed async function, a name, a description and a Pydantic
`ToolInput` model. The input model generates the JSON schema given to the LLM.
`ToolRegistry` rejects duplicate names, exposes `definitions()` for `LLMRequest`,
and dispatches a `ToolUse` through `execute(call, context)`:

```python
from pydantic import Field

from chatbot.tools.base import Tool, ToolContext, ToolInput
from chatbot.tools.pagination import PaginationParams, ToolPage, paginate
from chatbot.tools.registry import ToolRegistry


class NamesInput(ToolInput):
    pagination: PaginationParams = Field(default_factory=PaginationParams)


async def list_names(inputs: NamesInput, context: ToolContext) -> ToolPage:
    return paginate([context.user.display_name], inputs.pagination)


registry = ToolRegistry()
registry.register(Tool("list_names", "List names visible to the caller", NamesInput, list_names))
definitions = registry.definitions()
```

Application code supplies `ToolContext(user, session_id, turn_id, trace_id)`;
it is separate from model arguments. Undeclared input fields are rejected.
Execution binds the context IDs to logging for the duration of the call.
Successful dispatch returns a `ToolResult` containing `items`, `total` and
`next_cursor`. Invalid arguments, unknown tools, oversized results and execution
failures raise `ToolError`; unexpected failures expose a safe message and retain
their cause for debugging.

Pages default to 20 items, with a maximum of 100 and a 16 KiB JSON payload limit.
`paginate` handles ordered in-memory data. Database tools can use `decode_cursor`
to obtain an offset, fetch at most `limit` rows, and call `build_page` with that
slice and the total count. Byte limits include the whole envelope in UTF-8;
shrinking a page keeps the cursor at the next omitted item. A single oversized
item fails instead of being skipped.

Cursors use versioned offsets, so tools must supply a stable ordering. Pass a
stable `scope` representing the user and filters to detect reuse with a different
query. Cursors are pagination hints; tools enforce access on every call. Offset
pagination can shift when the underlying data changes between requests.

## Project queries

Run `make seed` to load the synthetic company dataset. `QueryProjects` owns a
separate connection pool using `RO_DB_USER` / `RO_DB_PASSWORD` and reads only
`curated.projects`. Register its tool while the pool is open:

```python
from chatbot.config import Settings
from chatbot.tools.query_projects import QueryProjects
from chatbot.tools.registry import ToolRegistry


async def project_tool_example() -> None:
    async with QueryProjects(Settings()) as projects:
        registry = ToolRegistry()
        registry.register(projects.tool())
        definitions = registry.definitions()
```

The tool accepts `status`, exact `client_id`, a `client` name substring,
`start_date_from` / `start_date_to` (inclusive project start dates), and `text`
matching the project name or description. Substring matching ignores case and
treats `%` and `_` literally. Filters combine with AND. Pagination uses the
standard nested `pagination` input, for example:

```json
{"status": "active", "client": "Northstar", "pagination": {"limit": 5}}
```

Results are ordered by project ID and contain the approved columns documented
in [db/README.md](../db/README.md). Dates are ISO strings; an unknown end date
is null. Count and page queries use one read-only repeatable-read transaction.
All filter values are SQL parameters. Cursors are scoped to the caller and
filters; send the same filters with `next_cursor` to fetch the next page.
All approved demo projects are visible to every caller; this tool currently
has no per-user row restrictions.

## Agent and terminal chat

Configure the selected provider's API key, run `make seed`, then start the CLI:

```bash
uv run python -m chatbot.cli chat --user alice
uv run python -m chatbot.cli chat --user alice --prompt "Which projects are active?"
uv run python -m chatbot.cli chat --user alice --session SESSION_UUID
```

The CLI prints the session UUID to stderr. Reuse it with the same `--user` to
resume a conversation. Enter `/quit`, `/exit` or EOF to leave interactive chat.
`--prompt` streams one reply to stdout, then exits with status 0 on success or
1 on failure. The CLI user option selects a local development identity.

Application code uses the same interface:

```python
from contextlib import aclosing

from chatbot.app import build_chatbot
from chatbot.config import Settings
from chatbot.types import TextDelta, User


async def ask() -> None:
    async with build_chatbot(Settings()) as chatbot:
        user = await chatbot.store.upsert_user(User(external_id="alice", display_name="Alice"))
        session = await chatbot.store.create_session(user.id)
        async with aclosing(
            chatbot.respond(session.id, "Which projects are active?", user)
        ) as events:
            async for event in events:
                if isinstance(event, TextDelta):
                    print(event.text, end="", flush=True)
```

`respond(session_id, user_message, user)` checks ownership, loads history, calls
the provider, runs all requested tools, and repeats until an answer or
`AGENT_MAX_STEPS` provider calls. Tool failures become `is_error` tool results
that the model can use to correct its request. Events carry session, turn and
trace IDs and include text deltas, tool start/finish, and one terminal completion
or failure event. Usage is summed across provider calls.

New messages and the completed/failed turn are saved together before a terminal
event is emitted. Failed streams retain partial text; interrupted tool batches
get error results for unfinished calls so future history stays paired. Consume
the stream fully or close it with `aclosing` on early exit. Calls for one session
are serialized within an agent instance; database sequence constraints detect
conflicts between separate instances.

The composition root closes provider clients and both database pools on exit.
For unit tests, construct `Agent` with `MemoryStore`, `ScriptedProvider` and a
`ToolRegistry`. This commit sends full history; configurable context strategies
arrive in C12. Cost fields remain zero until the pricing gateway in C25. Trace
IDs currently provide log correlation; OpenTelemetry spans arrive in C13.
