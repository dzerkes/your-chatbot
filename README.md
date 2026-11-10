# your-chatbot

A proof of concept for a chatbot that answers questions from company data. It is
built as a complete, small system: a chat UI, a thin API, an agent loop with
tools, Postgres for history and memory, and tracing plus evals around all of it.

> **Status:** planning. Nothing runs yet. The build order is in
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
cp .env.example .env     # then fill in the API keys
uv sync                  # installs the shared dev tools
```

From milestone M1 onward (commit C16 in the plan):

```bash
docker compose up        # UI on http://localhost:8501, API on http://localhost:8000
```

## Development

```bash
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy .            # types
uv run pytest            # unit tests (no network, no database)
uv run pytest -m integration   # needs the Postgres container
```

## Documentation

- [Architecture and design decisions](docs/ARCHITECTURE.md)
- [Commit-by-commit project plan](docs/plan/PROJECT_PLAN.md)
- [Changelog](CHANGELOG.md)
- [Original project brief](docs/init_prompt/INIT_PROMPT.txt)

## License

MIT. See [LICENSE](LICENSE).
