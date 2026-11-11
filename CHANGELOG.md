# Changelog

All important changes to this project are recorded here, newest first.

- Each entry names the planned commit id from
  [docs/plan/PROJECT_PLAN.md](docs/plan/PROJECT_PLAN.md) and the git hash once
  the commit exists.
- Entry types: Added, Changed, Fixed, Removed, Security, Decision.
- Dates are in ISO format (YYYY-MM-DD).

## Unreleased

### 2026-11-12

| Plan id | Commit | Type | Change |
|---|---|---|---|
| C13 | Pending | Added | Private OpenTelemetry tracer provider and Langfuse OTLP/HTTP batch export, with no telemetry setup when tracing is disabled and exporter draining on application exit. |
| C13 | Pending | Added | Session/turn/LLM/tool span hierarchy, persisted OpenTelemetry trace IDs, model and cache usage, timings, tool arguments/result sizes and error status. Cost attributes remain marked uncalculated until C25. |
| C13 | Pending | Added | In-memory tracing tests, CLI integration checks for enabled/disabled export, and a live scripted CLI trace verified through Langfuse's observations and session APIs. |
| C12 | `8c13924` | Added | Context builder interface, full-history and sliding-window strategies selected by settings, and replaceable input-token estimation covering system text, messages and tool schemas. |
| C12 | `8c13924` | Added | Agent rebuilds context before each provider call, evicts only whole stored turns, preserves current tool exchanges, and reports context overflow without changing saved history. |
| C11 | `b99ea69` | Added | Provider-neutral agent loop and streaming `respond()` with tool execution, error recovery, step limits, cumulative usage and atomic turn persistence. |
| C11 | `b99ea69` | Added | Resource-owning `build_chatbot`, packaged system prompt and terminal CLI with one-shot questions and session resume. |
| C11 | `b99ea69` | Added | Scripted conversation tests and a CLI integration test using the real seeded project tool and Postgres history. |
| C10 | `1483e1b` | Added | Raw client/project tables, curated views exposing approved columns, and an explicit repeatable `make seed` for synthetic demo data. |
| C10 | `1483e1b` | Security | Read-only role gets SELECT on named curated views, without raw-table access or write grants; `query_projects` uses that role and parameterized filters. |
| C10 | `1483e1b` | Added | Typed project query tool with status, client, start-date and text filters, stable pagination, and integration tests for seeding, queries, permissions and downgrade. |
| C09 | `078b182` | Added | Typed async tools, Pydantic input schemas, a tool registry, server-supplied identity and trace context, and dispatch to provider-neutral tool results. |
| C09 | `078b182` | Added | Paginated JSON result envelope with scoped offset cursors, item and UTF-8 byte limits, plus tests for dispatch, context isolation and pagination progress. |
| C08 | `dd052cd` | Added | OpenAI Responses streaming adapter for text, function calls, usage and stop reasons; registry selects Anthropic or OpenAI from `LLM_PRIMARY`. |
| C08 | `dd052cd` | Added | Shared text and tool contract tests for both adapters, OpenAI recorded streams and registry selection tests. |
| C07 | `40184a7` | Added | Provider-neutral request, event and tool definitions, an Anthropic streaming adapter, and a deterministic scripted provider. |
| C07 | `40184a7` | Added | Recorded Anthropic stream tests and an opt-in `llm` test for a real API stream. |
| C06 | `be29dc4` | Added | `SessionStore` and `MessageStore` protocols, with async Postgres and in-memory implementations for users, sessions, turns and messages. |
| C06 | `be29dc4` | Security | Session-owner scoping on history reads and turn writes; a failed message insert rolls back the entire turn. |
| C06 | `be29dc4` | Added | Shared store contract tests, including an isolated Postgres round-trip of tool calls/results and rollback checks; `make migrate` prepares the schema for integration tests. |
| C05 | `c8139de` | Added | Alembic history migration for users, sessions, turns and JSONB messages, with keys, checks, indexes and cascading history cleanup. |
| C05 | `c8139de` | Security | Migration-owner connection and table-only read/write grants to `chatbot_app`; `chatbot_ro` has no history-table access. |
| C05 | `c8139de` | Added | One-shot Compose migration service and an isolated-database integration test covering upgrade, permissions, tool-call content and downgrade. |

### 2026-11-11

| Plan id | Commit | Type | Change |
|---|---|---|---|
| C04 | `17bf438` | Added | Installable `chatbot` workspace package with validated Pydantic settings, masked secrets, provider-neutral messages/content blocks, usage and streaming event types, and typed application errors. |
| C04 | `17bf438` | Added | Console and JSON logging with scoped session, turn and trace IDs, including isolation between concurrent tasks and restoration after errors. |
| C04 | `17bf438` | Added | Unit tests for settings precedence and validation, JSON round-trips, domain validation, errors and logging; package usage documentation. |
| C03 | `89c6309` | Added | Docker Compose with pinned pgvector/Postgres 17 image, a persistent named volume, localhost port binding and a TCP healthcheck. |
| C03 | `89c6309` | Security | Environment-driven owner, application and read-only logins, separate from the bootstrap superuser. Removed public database privileges and schema creation; read-only queries have a five-second statement timeout. |
| C03 | `89c6309` | Added | Make targets for database lifecycle, psql, lint, unit tests and integration tests; configuration and database permission tests; uv lockfile for reproducible tooling. |
| C03 | `89c6309` | Changed | Getting-started instructions explain database initialization, credentials, persistence and volume reset. |

### 2026-11-10

| Plan id | Commit | Type | Change |
|---|---|---|---|
| C02 | `8c37349` | Decision | CI runs on GitHub Actions, not GitLab CI, because the repo is hosted on GitHub. |
| C02 | `8c37349` | Decision | Migrations, role setup and demo data live in a top-level `db/` folder, so the `chatbot` package can use the schema without the backend. |
| C02 | `8c37349` | Added | Commit-by-commit project plan in `docs/plan/PROJECT_PLAN.md`: 31 commits in 5 phases, milestones M0 to M5, and a table mapping every requirement in the brief to commits. |
| C01 | `8c37349` | Added | Repository scaffolding: `README.md`, `pyproject.toml` (uv workspace root, ruff, mypy, pytest), `.python-version`, `.gitignore`, `.env.example`, this changelog, and `docs/ARCHITECTURE.md`. |
| C01 | `8c37349` | Decision | Use provider SDKs directly behind one `LLMProvider` interface. No LangChain or LangGraph. |
| C01 | `8c37349` | Decision | Postgres with pgvector is the single store for history, memory, vectors and the audit log. Langfuse is for observability only. |
| C01 | `8c37349` | Decision | Tools are read-only and query the database through a role that can only read curated views. |

## Milestones

| Milestone | Reached on | Commit |
|---|---|---|
| M0 Foundation | 2026-11-10 | `8c37349` |
| M1 Walking skeleton | | |
| M2 Company data | | |
| M3 Long conversations | | |
| M4 Governance | | |
| M5 Quality loop | | |
