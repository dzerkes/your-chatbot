# your-chatbot

A proof of concept for a chatbot that answers questions from company data. It is
built as a complete, small system: a chat UI, a thin API, an agent loop with
tools, Postgres for history and memory, and tracing plus evals around all of it.

> **Status:** Postgres with pgvector runs locally, the history tables are managed
> by Alembic (C05), and the `chatbot` package provides settings, logging and
> domain types. The agent and web services
> are still planned. The build order is in
> [docs/plan/PROJECT_PLAN.md](docs/plan/PROJECT_PLAN.md) and progress is tracked
> in [CHANGELOG.md](CHANGELOG.md).

## What it will do

- Chat over company data through read-only tools: `search_docs` (hybrid RAG)
  and `query_projects` (curated SQL views).
- Keep every session, message, tool call and tool result in Postgres.
- Trace every turn (session → turn → LLM call or tool call) with tokens,
  latency and cost, and send it to Langfuse.
- Run an eval set on every prompt, tool or model change.

## How the parts fit

| Part | Folder | Owner in a real team | Responsibility |
|---|---|---|---|
| Chat UI | `frontend/` | Frontend | Streamlit app. Talks to the backend over HTTP only. |
| API | `backend/` | Backend | FastAPI: auth, session endpoints, `/chat` stream. Thin. |
| The brain | `chatbot/` | Data science | `respond(session_id, user_message, user) -> stream`: context, agent loop, tools, persistence, traces. |
| Tool servers | `mcp_servers/` | Data science / data owners | Tools exposed over MCP. |
| Evals | `evals/` | Data science | Questions with expected answers, runner, scorers. |
| Database | Postgres + pgvector | Shared | History, memory, vectors, audit log, demo data. |

More detail, including the reasoning behind each choice, is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Planned layout

```
your-chatbot/
├── docker-compose.yml        # postgres, migrate, backend, frontend
├── pyproject.toml            # uv workspace root + shared tooling
├── .env.example              # every setting, documented
├── db/                       # migrations, role setup, demo data
├── chatbot/                  # installable package: the brain
│   ├── src/chatbot/
│   └── tests/
├── backend/                  # FastAPI, imports `chatbot`
├── frontend/                 # Streamlit, calls the backend
├── mcp_servers/              # tools over MCP
├── evals/                    # datasets, runner, scorers
└── docs/
    ├── ARCHITECTURE.md
    ├── plan/PROJECT_PLAN.md
    └── init_prompt/          # the original project brief
```

## Getting started

Requirements: Docker, [uv](https://docs.astral.sh/uv/), Python 3.12.

```bash
cp .env.example .env     # choose database passwords; API keys are needed later
uv sync                  # installs chatbot and the shared dev tools
make up                  # starts Postgres and waits for its healthcheck
docker compose run --rm migrate  # creates or updates the history tables
make psql                # connects as the migration owner (\q to exit)
make test-integration    # checks pgvector, authentication and database permissions
make down                # stops containers; keeps the named database volume
```

Postgres is available on `127.0.0.1:5432` (override `POSTGRES_PORT` in `.env`).
`POSTGRES_USER` names the migration owner; `APP_DB_USER` and `RO_DB_USER` name
the application and tool logins. All three are non-superusers. The container
uses a separate `postgres` superuser for initialization; its password defaults
to `POSTGRES_PASSWORD` for local development and can be overridden with
`POSTGRES_ADMIN_PASSWORD`.

Initialization creates the roles and the `vector` extension only on an empty
volume. Changing `.env` does not change existing database credentials. Apply
credential changes with `ALTER ROLE`, or, for a disposable local database,
`docker compose down -v` deletes **all database data** so the next `make up`
initializes it again. `make down` preserves data.

The read-only role has a five-second statement timeout and cannot create
schemas, regular tables or temporary tables. Object grants will be added with
the migrations in C10.

The migration job uses the owner role and runs `alembic upgrade head` once.
It can be run again safely when no new migrations exist. The initial revision
creates `users`, `sessions`, `turns` and `messages`; see [db/README.md](db/README.md)
for the schema, grants and downgrade command. If the host's port 5432 is in
use, set `POSTGRES_PORT=55432` in `.env` before `make up` or Compose commands.

From milestone M1 onward (commit C16 in the plan):

```bash
docker compose up        # UI on http://localhost:8501, API on http://localhost:8000
```

## Development

```bash
make lint                # lint and check formatting
uv run ruff format .     # format
uv run mypy .            # types
make test                # unit tests (no network, no database)
make test-integration    # starts Postgres, then runs integration tests
```

The `chatbot` package is installed in editable mode by `uv sync`, so source
changes are available immediately. Check the import with:

```bash
uv run python -c "import chatbot; print(chatbot.__file__)"
```

Application code loads configuration explicitly with `Settings()` from
`chatbot.config`. Values resolve from constructor arguments, then environment
variables, then `.env` in the working directory, then defaults. Blank environment
values use defaults; secrets are masked when settings are printed. API keys
are optional until a provider is used. The `.env` file is not loaded on import.

See [chatbot/README.md](chatbot/README.md) for the core types and logging API.

## Documentation

- [Architecture and design decisions](docs/ARCHITECTURE.md)
- [Commit-by-commit project plan](docs/plan/PROJECT_PLAN.md)
- [Changelog](CHANGELOG.md)
- [Original project brief](docs/init_prompt/INIT_PROMPT.txt)

## License

MIT. See [LICENSE](LICENSE).
