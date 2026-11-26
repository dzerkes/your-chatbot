# Runbook

How to run the system, check that it is healthy, and handle the usual
problems. To investigate one bad answer, see [DEBUGGING.md](DEBUGGING.md). How
the parts fit and why is in [ARCHITECTURE.md](ARCHITECTURE.md).

## From clone to a first answer

Requirements: Docker with Compose, [uv](https://docs.astral.sh/uv/), and an
API key for at least one model provider (OpenAI or Anthropic).

```bash
git clone <repository URL> your-chatbot && cd your-chatbot
cp .env.example .env
```

Edit `.env`:

- Replace every `change-me` value: `POSTGRES_PASSWORD`, `APP_DB_PASSWORD`,
  `RO_DB_PASSWORD` and `MCP_TOKEN_SECRET` (32 characters or more).
- Set the key for the provider in `LLM_PRIMARY`, `LLM_FALLBACKS` and
  `LLM_SUMMARY_MODEL`. The example uses `openai:` models and needs
  `OPENAI_API_KEY`. For Anthropic, set the three models to `anthropic:...` and
  `ANTHROPIC_API_KEY`.
- Optional: tracing (see [Tracing](#tracing)).

Then:

```bash
uv sync                      # Python environment for the CLI, tests and evals
make seed ingest             # Postgres, migrations, demo projects and documents
docker compose up -d --build # mcp, backend and frontend as well
```

Open <http://localhost:8501>, enter `dev-token-alice` and ask "Which Northstar
projects are active?". The first `ingest` downloads the local embedding and
reranking models (a few hundred MB) into `.cache/fastembed`. The first
`compose up` downloads them again into the `model_cache` volume for the MCP
server.

Without the web stack, the terminal chat runs the same agent in-process:

```bash
uv run python -m chatbot.cli chat --user alice --groups employees,engineering
```

## Services

| Service | Port (localhost) | Health check | Role |
|---|---|---|---|
| `postgres` | `POSTGRES_PORT` (5432) | `pg_isready` | History, audit, feedback, memory, company data, vectors |
| `migrate` | - | Exits 0 | Runs `alembic upgrade head` once, as the owner role |
| `mcp` | `MCP_PORT` (8001) | `GET /health` | `search_docs` and `query_projects` over MCP; only service with the read-only role and the models |
| `backend` | `BACKEND_PORT` (8000) | `GET /health` | Auth, sessions, `/chat` SSE, limits, feedback |
| `frontend` | `FRONTEND_PORT` (8501) | `GET /_stcore/health` | Streamlit UI |

All ports bind to `127.0.0.1`. `docker compose ps` shows each service's health.
Start order is enforced by health checks: Postgres, then the migration, then
MCP, then the backend, then the frontend.

| Task | Command |
|---|---|
| Start everything | `docker compose up -d --build` |
| Stop, keep data | `docker compose down` (or `make down`) |
| Stop and delete all data | `docker compose down -v` (database, model cache) |
| Follow logs | `docker compose logs -f backend mcp` |
| Apply new migrations | `make migrate` |
| Reload demo data | `make seed ingest` (both repeatable) |
| Add or change documents | Put Markdown with `groups:` front matter in a folder, then `uv run python -m chatbot.rag.ingest FOLDER [--prune]` |
| Open SQL as the owner | `make psql` |
| Back up the database | `docker compose exec -T postgres sh -c 'PGPASSWORD="$OWNER_DB_PASSWORD" pg_dump -h 127.0.0.1 -U "$OWNER_DB_USER" -d "$POSTGRES_DB" -Fc' > backup.dump` |

The backend reads settings at start. After changing `.env`, run
`docker compose up -d` so Compose recreates the containers that use it.
Database passwords are fixed when the volume is first created; see the
[root README](../README.md#getting-started) for changing them.

## Configuration that matters in operation

Every setting is documented in [`.env.example`](../.env.example). The ones you
change most:

| Setting | Effect |
|---|---|
| `LLM_PRIMARY`, `LLM_FALLBACKS` | Model chain. Transient errors retry `LLM_MAX_RETRIES` times, then the next model is tried. |
| `LLM_PRICES_FILE` | Your own price table (USD per million tokens). Without a price, cost is recorded as 0 and the cost cap cannot stop spend. |
| `RATE_LIMIT_REQUESTS_PER_MINUTE`, `COST_CAP_USD_PER_USER_PER_DAY` | Per-user admission limits in the backend. |
| `CONTEXT_STRATEGY`, `CONTEXT_*` | What history the model sees. |
| `MEMORY_ENABLED` | Cross-session facts, the memory tool and the UI panel. |
| `AUTH_DEV_TOKENS` | Development bearer tokens: `token:identity:groups`. Groups decide which documents a user can search. |
| `LOG_LEVEL`, `LOG_FORMAT` | `DEBUG` logs every model request verbatim; `json` gives one object per line. |
| `TRACING_ENABLED`, `LANGFUSE_*` | Trace export and feedback scores. |

A change to a prompt, tool, retrieval or model setting changes answers. Run
`make evals` before merging it; CI runs the same gate (see
[evals/README.md](../evals/README.md#baseline-and-ci)).

## When something goes wrong

Each failed turn ends with a `turn_failed` event whose `code` says where it
failed. The UI shows its message; the backend log has the stack trace.

| Symptom or code | Likely cause | What to do |
|---|---|---|
| Backend never healthy | Database or MCP not reachable, invalid settings | `docker compose logs backend migrate mcp`. Settings errors name the variable. |
| `provider_error` on every turn | Missing or invalid API key, model name, or no credit | Check the key for each model in the chain. `uv run pytest -m llm tests/test_llm_live.py` checks Anthropic access directly. |
| `provider_error` on some turns | Provider outage or rate limits | Add a model from another provider to `LLM_FALLBACKS`. Retries and fallbacks are logged as warnings. |
| `rate_limit_exceeded` | User sent more than the per-minute limit | Wait a minute, or raise `RATE_LIMIT_REQUESTS_PER_MINUTE`. |
| `cost_cap_exceeded` | User's spend since midnight UTC reached the cap | Raise `COST_CAP_USD_PER_USER_PER_DAY`, or wait for the next UTC day. |
| `context_overflow` | The system prompt, tool schemas and current turn alone exceed `CONTEXT_MAX_INPUT_TOKENS` | Raise the budget, or shorten the message. Older history is trimmed automatically. |
| `step_limit_exceeded` | The model kept calling tools for `AGENT_MAX_STEPS` steps | Look at its tool calls in the trace; usually a tool error it cannot recover from. |
| `tool_error` results in answers | Tool input rejected, database timeout, MCP unreachable | `docker compose logs mcp`. The read-only role has a 5-second statement timeout. |
| Search finds nothing | Documents not ingested, or the user lacks the document's groups | `make ingest`; check the user's groups in `users.groups` and the document's `groups:` front matter. |
| `store_error` | Postgres down or full | `docker compose ps postgres`, then its logs. |
| No traces in Langfuse | Tracing off, wrong host or keys, or the process exited before flushing | See [Tracing](#tracing). Chat keeps working; export failures are logged, not raised. |
| Feedback not in Langfuse | Scores are retried in the background | `SELECT count(*) FROM feedback WHERE next_export_at IS NOT NULL` shows votes still waiting. |

## Tracing

Tracing is off by default and optional: Postgres is the record of every turn,
and nothing breaks without Langfuse. To turn it on, set in `.env`:

```dotenv
TRACING_ENABLED=true
LANGFUSE_HOST=https://cloud.langfuse.com    # or your region, or a self-hosted URL
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

Then restart the backend (`docker compose up -d`). Each turn becomes one trace
named `chat` with `session` and `turn` spans, a generation per model call and
a span per tool call. Turns of one conversation are grouped in Langfuse's
Sessions view, and thumbs up/down arrive as a `user-feedback` score on the
turn's trace. What each span records is in
[chatbot/README.md](../chatbot/README.md#tracing-to-langfuse).

Traces contain questions, tool arguments, retrieved text and answers. The
cloud free tier is fine for the synthetic demo data (decision D6). For real
company data, self-host it.

## Self-hosting Langfuse

[`docker-compose.langfuse.yml`](../docker-compose.langfuse.yml) runs Langfuse
next to the chatbot as an overlay on the main Compose file. It is trimmed from
Langfuse's own
[Docker Compose deployment](https://langfuse.com/self-hosting/deployment/docker-compose):
the web UI and worker, their own Postgres, ClickHouse, Redis and MinIO, no AI
gateway or media uploads, and nothing published except the UI on
`127.0.0.1:3000`. Its service and variable names are prefixed so they do not
clash with the chatbot's Postgres.

Expect about 1 GB of images and several GB of memory for Docker. Langfuse
recommends 4 cores and 16 GB for a test VM.

1. Generate the secrets and add them to `.env`. Use hex values: they go into a
   database URL unescaped.

   ```bash
   for name in DB_PASSWORD CLICKHOUSE_PASSWORD REDIS_PASSWORD MINIO_PASSWORD \
       SALT ENCRYPTION_KEY NEXTAUTH_SECRET ADMIN_PASSWORD; do
     echo "LANGFUSE_$name=$(openssl rand -hex 32)"
   done >> .env
   ```

2. In `.env`, choose the project keys and point the host processes at it:

   ```dotenv
   TRACING_ENABLED=true
   LANGFUSE_HOST=http://localhost:3000
   LANGFUSE_PUBLIC_KEY=pk-lf-local-<random>
   LANGFUSE_SECRET_KEY=sk-lf-local-<random>
   LANGFUSE_ADMIN_EMAIL=you@example.com     # optional, default admin@example.com
   ```

3. Start both stacks together. Always pass both files, also to `down`, or
   Compose treats the Langfuse containers as orphans:

   ```bash
   docker compose -f docker-compose.yml -f docker-compose.langfuse.yml up -d --build
   ```

On first start Langfuse creates an organization and a project named
`your-chatbot`, with your two keys as its API keys, and an admin user with
`LANGFUSE_ADMIN_EMAIL` and `LANGFUSE_ADMIN_PASSWORD`. Sign in at
<http://localhost:3000>. The overlay points the backend container at
`http://langfuse-web:3000`; the CLI, evals and a backend run with `uv` use
`LANGFUSE_HOST` from `.env`. Headless initialization only creates missing
objects: changing the keys later means creating new keys in the Langfuse UI.

To check it: ask one question in the UI, then open **Tracing** in the
`your-chatbot` project. A trace appears within a few seconds of the answer.
Trace IDs from Postgres open directly at
`http://localhost:3000/project/your-chatbot/traces/<trace_id>`.

| Task | Command (with both `-f` files) |
|---|---|
| Langfuse logs | `docker compose ... logs -f langfuse-web langfuse-worker` |
| Upgrade Langfuse | `docker compose ... pull langfuse-web langfuse-worker && docker compose ... up -d` |
| Pin a release | `LANGFUSE_IMAGE_TAG=<version>` in `.env` (default: latest `4`) |
| Delete trace data | `docker volume rm` the `your-chatbot_langfuse_*` volumes after `down` |

This setup has no high availability, backups or TLS. For a shared or
production instance, follow Langfuse's Kubernetes or cloud guides, put it
behind TLS, and keep the same environment variables.

## Evals

```bash
make evals                               # full set, real models, prints scores
uv run python -m evals.runner --baseline # also compare with evals/baseline.json
```

When a change is meant to change scores (a new model, better retrieval), run
`uv run python -m evals.runner --config-from .env.example --write-baseline`
and commit `evals/baseline.json` with the change. CI needs the
`OPENAI_API_KEY` repository secret. Details are in
[evals/README.md](../evals/README.md).

## Adding a provider, a tool or a context strategy

Each one is a single registration. Run `make test` and `make evals` afterwards;
the eval workflow runs on all three kinds of change.

### A model provider

1. Add `chatbot/src/chatbot/llm/<name>.py` with a class that has a `name`
   (`"<provider>:<model>"`), `stream(request: LLMRequest) -> AsyncIterator[LLMEvent]`
   and `aclose()`, and is constructed as `Provider(model, settings)`. Convert
   `LLMRequest` messages and tool definitions to the SDK's format, and its
   stream back to `LLMTextDelta`, `LLMToolCall`, then one `LLMUsage` and one
   `LLMStop`. Raise `ProviderError(..., retryable=True)` for errors worth a
   retry, so the gateway can retry and fall back. `anthropic.py` and
   `openai.py` are the references.
2. Add its key to `Settings` in `config.py` and to `.env.example` (and to the
   `backend` environment in `docker-compose.yml`).
3. Register the class in `ProviderRegistry._factories` in `llm/registry.py`.
4. Add prices for its models to `llm/prices.toml`, or turns cost 0 and the
   cost cap does not apply.
5. Add it to the shared contract tests in `chatbot/tests/test_llm_contract.py`
   with recorded stream fixtures, so no test needs the network.

Then `LLM_PRIMARY=<provider>:<model>` selects it, and it can serve as a
fallback for the others.

### A tool

1. Write an input model (subclass `ToolInput`; field descriptions become the
   schema the model sees) and an async function
   `(inputs, context: ToolContext) -> ToolPage`. Take the user and groups from
   `context`, never from model arguments. Return pages with `paginate` or
   `build_page`. See `tools/query_projects.py`, and
   [chatbot/README.md](../chatbot/README.md#tools) for the rules.
2. Register `Tool(name, description, InputModel, function)` in
   `open_local_tools` in `tools/local.py`. That serves it in-process for the
   CLI and over MCP from the `mcp` service; nothing else changes. A tool that
   writes belongs in the chatbot process instead, as memory does in `app.py`,
   so the MCP server stays read-only.
3. If it reads new tables, grant `SELECT` on a curated view to the read-only
   role in a new migration. Never grant raw tables.
4. To record what it returned in the audit log, add its source fields to
   `returned_sources` in `audit.py`.
5. Add unit tests with a `ToolRegistry`, and eval cases that expect the tool in
   `evals/datasets/demo.yaml`. Regenerate the baseline, since the case counts
   change.

### A context strategy

Strategies are stages that run over whole stored turns.

1. Add the name to `ContextStage` in `config.py`.
2. For a stage that composes with the others, implement it as a function over
   turns and add it to `STAGE_ORDER` and `PipelineContextBuilder.build` in
   `context/pipeline.py`, keeping tool calls paired with their results. For a
   standalone builder, implement the `ContextBuilder` protocol in
   `context/base.py` and select it in `build_context_builder` in
   `context/__init__.py`.
3. Keep the prompt prefix stable from one turn to the next, or prompt caching
   stops working: change earlier turns in one step, as `summary` does, not a
   little every turn.
4. Test it in `chatbot/tests/test_context_pipeline.py`, then compare it with
   the default on the eval set: `CONTEXT_STRATEGY=<stages> make evals`.
