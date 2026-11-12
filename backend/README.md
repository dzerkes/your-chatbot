# Chatbot HTTP service

FastAPI provides authentication, owner-scoped session endpoints and an SSE
transport for `chatbot.Agent.respond()`. The core package owns prompts, context,
LLM calls, tools, tracing and persistence.

## Run locally

From the repository root, configure `.env` with database credentials and the
selected LLM provider's API key, then run:

```bash
uv sync
make seed
uv run --package chatbot-backend python -m app
```

The launcher binds to `127.0.0.1` and uses `BACKEND_PORT` (default 8000).
Interactive API docs are at `http://localhost:8000/docs`. You can also run
`uv run uvicorn app.main:app --host 127.0.0.1 --port 8000`.

Settings and the chatbot's clients/pools are created during FastAPI's lifespan.
Importing `app.main` does not connect to a database or model provider. Application
shutdown closes those resources and flushes enabled tracing exporters.

## Development authentication

`AUTH_MODE=dev` uses `AUTH_DEV_TOKENS` from the shared settings:

```dotenv
AUTH_DEV_TOKENS=dev-token-alice:alice,dev-token-bob:bob
```

Each entry maps a bearer token to a server-controlled external identity. Valid
requests upsert that user and reuse its canonical UUID. The `current_user`
dependency is the replacement point for future SSO. Tokens are compared using
constant-time byte comparisons. Empty entries and duplicate tokens fail startup
without echoing token values.

All session/chat endpoints require `Authorization: Bearer TOKEN`. Identity and
groups come from the server; request bodies reject undeclared fields. Missing
or invalid credentials return 401 with a Bearer challenge. Unknown sessions and
sessions owned by another user both return 404.

## Endpoints

| Method/path | Request | Response |
|---|---|---|
| `GET /health` | Public | `{"status":"ok"}`; liveness after successful startup. |
| `GET /sessions` | Bearer token | The caller's sessions, newest activity first. |
| `POST /sessions` | Optional `{"title":"Projects"}` | 201 and the created session. Titles are trimmed and limited to 200 characters. |
| `GET /sessions/{id}` | Bearer token | Session fields plus `messages`, containing full stored history in sequence order. |
| `POST /sessions/{id}/chat` | `{"message":"Which projects are active?"}` | Server-sent events using the core event schema. |

Chat messages must contain non-whitespace text and be at most 16,000 characters.
Their original whitespace is preserved. Invalid bodies and UUIDs return 422.
Ownership is checked before the streaming response starts.

## Stream a reply

Create a session and copy its `id` into the second command:

```bash
curl -sS -X POST http://localhost:8000/sessions \
  -H 'Authorization: Bearer dev-token-alice' \
  -H 'Content-Type: application/json' \
  --data '{"title":"Project questions"}'

curl -N -X POST http://localhost:8000/sessions/SESSION_UUID/chat \
  -H 'Authorization: Bearer dev-token-alice' \
  -H 'Content-Type: application/json' \
  --data '{"message":"Which Northstar projects are active?"}'
```

The stream starts with a `: connected` comment. Each event has a named `event`
field and one JSON `data` field, separated from the next event by a blank line:

```text
event: text_delta
data: {"session_id":"...","turn_id":"...","trace_id":"...","type":"text_delta","text":"Research Portal"}
```

Event types are `text_delta`, `tool_call_started`, `tool_call_finished`,
`turn_completed` and `turn_failed`. Once headers have been sent, generation
errors are represented by `turn_failed` inside the 200 stream. A terminal event
is emitted after the core saves the turn. Stream requests use POST and do not
provide event replay; use the session detail endpoint to reload saved history.

The transport sets `Cache-Control: no-cache` and `X-Accel-Buffering: no`.
Disconnect cleanup closes the event iterator in its originating task, protecting
logging/tracing context. Interrupted generation saves a failed turn with partial
text and paired tool results before releasing the session.

## Tests

```bash
uv run pytest backend/tests
make test-integration
```

API tests inject `MemoryStore` and `ScriptedProvider` through `create_app`'s
`chatbot_factory` argument. They cover authentication, ownership, SSE framing,
history, failures, validation and both ASGI disconnect paths. Integration tests
exercise the same API against the real history store and seeded project tool.
