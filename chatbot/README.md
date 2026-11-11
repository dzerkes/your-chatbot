# Chatbot core

This package holds the application code shared by the future CLI, backend and
eval runner. It has no FastAPI, Streamlit, database or provider SDK dependency.
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
