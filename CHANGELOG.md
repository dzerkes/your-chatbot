# Changelog

All important changes to this project are recorded here, newest first.

- Each entry names the planned commit id from
  [docs/plan/PROJECT_PLAN.md](docs/plan/PROJECT_PLAN.md) and the git hash once
  the commit exists.
- Entry types: Added, Changed, Fixed, Removed, Security, Decision.
- Dates are in ISO format (YYYY-MM-DD).

## Unreleased

### 2026-11-27

| Plan id | Commit | Type | Change |
|---|---|---|---|
| C31 | `bc0fe57` | Added | `docs/RUNBOOK.md`: from clone to a first answer, services and health checks, common operations, a table of failure codes and fixes, tracing setup, evals, and how to add a provider, a tool or a context strategy. |
| C31 | `bc0fe57` | Added | `docs/DEBUGGING.md`: follow one bad answer from the UI, a thumbs down or a user's report to its turn, stored messages, audit rows, retrieved chunks, logs and trace, then replay it and keep it fixed with an eval case. |
| C31 | `bc0fe57` | Added | `python -m chatbot.cli replay --session ID [--until TURN]` asks a saved session's questions again in a new session with the current code and settings, printing each original answer next to the new one. The CLI also prints each turn's trace ID. |
| C31 | `bc0fe57` | Added | `docker-compose.langfuse.yml`: self-hosted Langfuse as an overlay on the main Compose file, initialized with the project keys from `.env`. Setup is in the runbook. |
| C31 | `bc0fe57` | Fixed | `LOG_LEVEL=DEBUG` now logs each model request verbatim, as `.env.example` and the architecture document already said. |

### 2026-11-26

| Plan id | Commit | Type | Change |
|---|---|---|---|
| C30 | `0a35caf` | Added | Evals workflow: on changes to prompts, tools, retrieval, context, LLM adapters, settings, demo data or evals, CI runs the eval set with the committed settings and fails when a score drops below `evals/baseline.json` by more than its tolerance. The comparison and report go to the job summary. |
| C30 | `0a35caf` | Added | `evals.runner --baseline` compares a run with the baseline (exit 1 on regression), and `--write-baseline` saves a new one. `--config-from` takes the model, context and retrieval settings from an env file. Refusal and injection allow no drop. Changed case counts and a doubled cost per case also fail. |
| C30 | `0a35caf` | Changed | `.env.example` uses `openai:gpt-6-luna` with fallback `openai:gpt-5.6-luna` for chat and summaries, matching the models the baseline was measured on. |
| C29 | `d9a0c5a` | Added | `evals` workspace package: `uv run python -m evals.runner` (or `make evals`) sends each case in `evals/datasets/demo.yaml` through `Agent.respond` as its own user, then prints a score table and writes `results.json` and `report.md` under `evals/results/`. |
| C29 | `d9a0c5a` | Added | Scorers for tool choice, retrieval hit rate, answer facts (graded by a judge model with a rubric), refusals in no-access cases, injection resistance, and cost and latency budgets. With tracing enabled, the scores are attached to each case's Langfuse trace as `eval.<scorer>`. |
| C29 | `d9a0c5a` | Added | 21 demo cases over the seeded projects and documents for three users with different groups, including no-access questions and the three C21 injection fixtures. |
| C28 | `bc694a2` | Added | Thumbs up/down with an optional comment under each answer, via `PUT /sessions/{id}/turns/{turn_id}/feedback`. One current vote per turn is stored in the `feedback` table and returned with the session, so the UI shows it after a reload. |
| C28 | `bc694a2` | Added | With tracing enabled, each vote is sent to Langfuse as a boolean `user-feedback` score on the turn's trace. A background exporter retries failed deliveries from Postgres every minute, and a revised vote replaces the earlier score because the score ID is the turn ID. |
| C27 | `94dd6cc` | Added | Postgres records each admitted turn and executed tool call, including the question, arguments, outcome and returned source identifiers. The application can append records. A database trigger rejects updates, deletes and truncation. |

### 2026-11-25

| Plan id | Commit | Type | Change |
|---|---|---|---|
| C26 | `3ca023d` | Added | Chat admission enforces a rolling per-user request limit and UTC daily spending cap across sessions. Rejected requests stream a readable reason before generation. |
| C25 | `0b16f03` | Added | The LLM gateway retries transient failures with capped backoff and jitter, then tries the configured fallback models in order. A partial response is preserved if streaming fails. |
| C25 | `0b16f03` | Added | Per-model USD rates account for cache reads, cache writes and long-context pricing. Generation spans identify the serving model and cost. Saved turns and completion events total reported costs across tool steps and summary calls. |
| C24 | `0f019ea` | Added | User memory behind `MEMORY_ENABLED`: `memory` tool (remember, recall, forget), `user_memory` table, recent facts in the system prompt, `GET/DELETE /memories` and a "What I remember" UI panel. |
| C24 | `0f019ea` | Security | `remember` must quote the user's own words, checked against what they typed in the session, so document or tool text cannot become a memory; the app role cannot update facts, the read-only role cannot read them, and the tool runs in-process so the MCP server stays read-only. |
| C23 | `f0aaaa6` | Added | Context pipeline (`CONTEXT_STRATEGY=summary,compact_tools`, default): tool results from earlier turns become one-line notes, a running summary written by `LLM_SUMMARY_MODEL` replaces older turns in large steps (turn trigger or 80% of the token budget), and a final budget guard. |
| C23 | `f0aaaa6` | Added | Prompt caching: Anthropic cache breakpoints after tools, system prompt and conversation; OpenAI `prompt_cache_key` per session. `PROMPT_CACHING` toggles it. |
| C23 | `f0aaaa6` | Added | Migration for `summary_through_sequence` and `summary_updated_at` on sessions, and a forward-only `update_summary` store method; `update_session` now changes only the title. |
| C23 | `f0aaaa6` | Changed | Token estimates count tool results wrapped as sent; summary model usage is added to the turn's usage. |
| C22 | `9338ca4` | Added | `mcp_servers` package and `mcp` Compose service serving `query_projects` and `search_docs` over stateless streamable HTTP, from the same `Tool` objects used in-process; `TOOLS_TRANSPORT` selects in-process or MCP. |
| C22 | `9338ca4` | Security | Each MCP request carries a short-lived HS256 token signed by the backend (user, groups, session, turn, trace); the server verifies it before handling any message and builds `ToolContext` only from it. Only the MCP service holds the read-only role and models. |
| C22 | `9338ca4` | Added | `McpToolset`/`McpTool` client behind the existing tool interface, `python -m mcp_servers.token` for external clients, and parity tests running the same agent tests on both transports. |
| C22 | `9338ca4` | Fixed | Docker images could contain stale workspace code because uv's shared cache reused wheels until a pyproject.toml changed; all members now declare `tool.uv.cache-keys`. |
| C21 | `200b392` | Security | Tool results are cleaned (invisible characters, images) and wrapped in an `untrusted_data` envelope before each model call; stored history is unchanged. The system prompt never follows instructions found there and never outputs images. |
| C21 | `200b392` | Security | The chat UI renders text without images, closing zero-click exfiltration through image URLs. |
| C21 | `200b392` | Added | Three adversarial seed documents (formatting hijack, link exfiltration with a fake closing tag, fake system message with an invisible tag-character instruction) and opt-in `llm` tests with an undefended control. |
| C21 | `200b392` | Changed | `search_docs` drops model-supplied `tags` and uses flat `limit`/`cursor` arguments; invalid tool arguments name the failing fields without echoing values. |
| C20 | `834d5b2` | Added | `search_docs` tool: hybrid retrieval, reranking of the top fused hits, and paginated compact hits (title, section, source, snippet, chunk id, score) with cursors bound to the user, query and filters. The system prompt asks for bracketed title and section citations and "no document" instead of guesses. |
| C20 | `834d5b2` | Added | Reranker strategies `none`, `local` (fastembed cross-encoder) and `api` (Cohere-compatible `/rerank`), selected by `RERANKER`. |
| C20 | `834d5b2` | Security | Dev tokens carry groups (`token:identity:group|group`) and the CLI has `--groups`; search access comes only from these, never from tool arguments. |
| C20 | `834d5b2` | Decision | D5: local `Xenova/ms-marco-MiniLM-L-6-v2` on the top 20, MRR@10 0.931 against 0.915 without a reranker. |
| C20 | `834d5b2` | Changed | The backend image installs `chatbot[local-embeddings]` and keeps downloaded models in the `model_cache` volume. |
| C19 | `f454928` | Added | Hybrid retriever: Postgres full-text leg (any stemmed term, behind a `LexicalSearch` interface) and pgvector cosine leg with iterative HNSW scan, run concurrently on the read-only role and merged with reciprocal rank fusion. |
| C19 | `f454928` | Security | The caller's groups, plus optional source-prefix and tag filters, are applied inside both SQL queries; a caller with no groups gets nothing. Tested for every labeled query across five group sets. |
| C19 | `f454928` | Added | 24-question labeled set with a test that checks hybrid recall@10 and exact-term ranking against each leg; CI caches the local embedding model. |
| C19 | `f454928` | Changed | Plan: "Done when" amended because vector-only had the best overall MRR on the small demo corpus (vector 0.958, hybrid 0.915, lexical 0.795); hybrid kept as the design. |
| C18 | `0be3756` | Added | `rag` schema with documents and chunks: HNSW cosine index on embeddings, weighted generated full-text vector with GIN index, and per-chunk allowed groups with GIN index. |
| C18 | `0be3756` | Added | Structure-aware Markdown chunking (heading paths, whole code fences, in-section overlap), `Embedder` strategy (`local:` fastembed, `openai:`, `fake:`), and an idempotent `python -m chatbot.rag.ingest` CLI with `--prune` and `make ingest`. |
| C18 | `0be3756` | Added | Seven synthetic demo documents across `employees`, `engineering`, `finance`, `hr` and `leadership`, including restricted ones for access-control tests. |
| C18 | `0be3756` | Security | Documents must name at least one group (no implicit public access); the read-only role gets `SELECT` on `rag` only, the application role nothing, and ingestion writes as the owner. |
| C18 | `0be3756` | Decision | D4: local `BAAI/bge-small-en-v1.5` (384 dimensions); fastembed is an optional `local-embeddings` extra so the migrate image stays small. |
| C18 | `0be3756` | Changed | Integration-test databases now get the `vector` extension from the superuser, which the `rag` migration requires. |

### 2026-11-12

| Plan id | Commit | Type | Change |
|---|---|---|---|
| C17 | `ac3f8b3` | Added | GitHub Actions CI: ruff, mypy, unit tests, and integration tests against Compose Postgres; `llm` tests excluded. |
| — | `cff69a4` | Fixed | Provider adapters log the provider's error reason and request ID; users still see a generic message. |
| C16 | `4d8fbf5` | Added | Backend, migrate and frontend images built with uv from the lockfile; Compose starts postgres, migrate, backend and frontend in order with healthchecks. |
| C16 | `4d8fbf5` | Fixed | `.env.example` no longer puts inline comments on empty keys, which Compose read as values. |
| C15 | `cd0ac6d` | Added | Streamlit workspace client with saved conversations, new-chat drafts, streaming replies, collapsed tool activity and URL-based session selection. |
| C15 | `cd0ac6d` | Added | Independent frontend settings and HTTP/SSE client, friendly failures, account-change isolation, and AppTest coverage including a real API/Postgres reload-and-resume walkthrough. |
| C14 | `d51f349` | Added | FastAPI workspace service with lifespan-owned chatbot resources, dev bearer authentication, owner-scoped session endpoints, health and POST chat SSE. |
| C14 | `d51f349` | Security | Server-controlled user identity, forbidden extra request fields, 404 for inaccessible sessions, and safe HTTP error handling. |
| C14 | `d51f349` | Added | Scripted API tests, real database/tool integration coverage and cancellation-shielded persistence when a streaming client disconnects. |
| C13 | `1a81b58` | Added | Private OpenTelemetry tracer provider and Langfuse OTLP/HTTP batch export, with no telemetry setup when tracing is disabled and exporter draining on application exit. |
| C13 | `1a81b58` | Added | Session/turn/LLM/tool span hierarchy, persisted OpenTelemetry trace IDs, model and cache usage, timings, tool arguments/result sizes and error status. Cost attributes remain marked uncalculated until C25. |
| C13 | `1a81b58` | Added | In-memory tracing tests, CLI integration checks for enabled/disabled export, and a live scripted CLI trace verified through Langfuse's observations and session APIs. |
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
