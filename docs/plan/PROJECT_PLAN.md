# Project plan

The project is built commit by commit. Each commit is small, leaves the repo in
a working state, and has a clear way to check that it is done. The design
behind the plan is in [../ARCHITECTURE.md](../ARCHITECTURE.md). The original
brief is in [../init_prompt/INIT_PROMPT.txt](../init_prompt/INIT_PROMPT.txt).

## How to use this plan

- Commit ids (`C01`, `C02`, ...) are stable names used in `CHANGELOG.md`.
- Commit messages follow Conventional Commits: `type(scope): summary`.
- When a commit lands, tick its box here and add the date and hash to
  `CHANGELOG.md`.
- If the plan changes, edit this file in the same commit that makes the change
  necessary, and note it in the changelog.
- Every commit that adds code also adds its tests.

## Milestones

| Milestone | After | What works |
|---|---|---|
| M0 Foundation | C02 | Repo, tooling, docs, plan. |
| M1 Walking skeleton | C17 | `docker compose up` gives a Streamlit chat with session history, one tool (`query_projects`), and every turn visible as a trace in Langfuse. |
| M2 Company data | C22 | `search_docs` with hybrid search, reranker and access control. Injection defenses. Tools served over MCP. |
| M3 Long conversations | C24 | Context strategies, prompt caching, optional user memory. |
| M4 Governance | C28 | Fallbacks, cost tracking, rate limits and cost caps, audit log, feedback. |
| M5 Quality loop | C31 | Eval set and runner in CI, runbook. |

## Decisions

D1 and D2 are settled. The others are defaults that can still change without
affecting the commits before them.

| # | Decision | Choice | Decide by |
|---|---|---|---|
| D1 | Where migrations live | Top-level `db/`, not `backend/migrations/`. The `chatbot` package and its tests need the schema without the backend, and a `migrate` compose job runs them once. | Confirmed 2026-11-10 |
| D2 | CI system | GitHub Actions (`.github/workflows/`). The brief said GitLab CI, but the repo is hosted on GitHub. | Confirmed 2026-11-10 |
| D3 | Demo data | A small synthetic dataset (clients, projects, documents) with no real company data. | C10 |
| D4 | Embedding model | A small local model, so no document text leaves the machine and ingestion is free. Alternative: an embeddings API. | C18 |
| D5 | Reranker | A small local cross-encoder, with `none` as the baseline. Alternative: a rerank API. | C20 |
| D6 | Langfuse | Cloud free tier with non-sensitive data. Self-hosting documented in C31. | C13 |

---

## Phase 0: Foundation

### [x] C01 `chore: set up repository scaffolding`

- **Goal:** a repo that explains itself and has its tooling configured.
- **Files:** `README.md`, `pyproject.toml`, `.python-version`, `.gitignore`,
  `.env.example`, `CHANGELOG.md`, `docs/ARCHITECTURE.md`. (`.env` is created
  locally and is git-ignored.)
- **Done when:** `uv sync` installs the dev tools and `uv run ruff check .`
  passes.

### [x] C02 `docs: add commit-by-commit project plan`

- **Goal:** this file, plus the original brief under `docs/init_prompt/`.
- **Done when:** every requirement in the brief maps to a commit below.

---

## Phase 1: Walking skeleton (M1)

The aim is the thinnest slice through every layer, with real persistence and
real tracing, before making any single layer good.

### [x] C03 `feat(infra): add docker compose with Postgres and pgvector`

- **Goal:** one command starts the database with the three roles.
- **Files:** `docker-compose.yml`, `db/init/01_roles.sql`, `Makefile`
  (`up`, `down`, `psql`, `test`, `lint`).
- **Details:** `pgvector/pgvector` image, named volume, healthcheck. The init
  script creates `chatbot_owner`, `chatbot_app` and `chatbot_ro` from the
  environment and enables the `vector` extension. `chatbot_ro` gets a statement
  timeout. The three application roles are non-superusers; the image uses a
  separate `postgres` bootstrap superuser. Integration tests in
  `tests/test_infra.py` check authentication, pgvector and permissions.
- **Done when:** `make up` then `make psql` connects, and `chatbot_ro` cannot
  create a table.

### [ ] C04 `feat(chatbot): add package skeleton with settings, logging and domain types`

- **Goal:** an installable `chatbot` package with the shared vocabulary.
- **Files:** `chatbot/pyproject.toml`, `chatbot/src/chatbot/{config,logging,errors,types}.py`,
  `chatbot/tests/`. Adds `chatbot` to the workspace.
- **Details:** `Settings` (pydantic-settings) is the only place that reads the
  environment. Types: `User`, `Message`, content blocks (`Text`, `ToolUse`,
  `ToolResult`), `Usage`, and the `Event` union returned by `respond`. Logging
  binds `session_id`, `turn_id` and `trace_id` to every line. Typed errors.
- **Done when:** `uv run pytest` passes and `python -c "import chatbot"` works.

### [ ] C05 `feat(db): add migrations for users, sessions, turns and messages`

- **Goal:** the history schema.
- **Files:** `db/alembic.ini`, `db/migrations/`, a `migrate` service in compose.
- **Details:** `users` (external id, display name, groups), `sessions` (owner,
  title, summary, timestamps), `turns` (session, status, trace id, tokens,
  cost, latency), `messages` (session, turn, sequence, role, `content` JSONB,
  model, `usage` JSONB). Tool calls and tool results are content blocks inside
  `content`. Grants: `chatbot_app` read/write, `chatbot_ro` nothing.
- **Done when:** `docker compose run migrate` creates the tables and
  `alembic downgrade base` removes them.

### [ ] C06 `feat(chatbot): add session and message repositories`

- **Goal:** all history SQL in one place.
- **Files:** `chatbot/src/chatbot/store/{base,postgres,memory}.py`.
- **Details:** Repository pattern. `SessionStore` and `MessageStore` protocols
  with a Postgres implementation (async) and an in-memory one for unit tests.
  Writes for one turn happen in one transaction.
- **Done when:** integration tests round-trip a session with a tool call and a
  tool result and read it back identical.

### [ ] C07 `feat(llm): add provider interface, Anthropic adapter and scripted fake`

- **Goal:** one interface the rest of the code talks to.
- **Files:** `chatbot/src/chatbot/llm/{base,anthropic,fake}.py`.
- **Details:** `LLMProvider.stream(LLMRequest) -> AsyncIterator[LLMEvent]`.
  Normalized request (system, messages, tools, limits) and events (text delta,
  tool call, usage, stop reason). The fake provider replays a script, which
  makes agent tests deterministic and free.
- **Done when:** adapter unit tests pass against recorded responses, and one
  `llm`-marked test streams from the real API.

### [ ] C08 `feat(llm): add OpenAI adapter and provider registry`

- **Goal:** prove that a new SDK is one adapter and one registration.
- **Files:** `chatbot/src/chatbot/llm/{openai,registry}.py`.
- **Details:** `registry.get("provider:model")` builds the right adapter. A
  shared contract test suite runs against every adapter, so they all behave
  the same for text, tool calls and usage.
- **Done when:** the contract tests pass for both adapters, and changing
  `LLM_PRIMARY` switches provider with no code change.

### [ ] C09 `feat(tools): add tool interface, registry and paginated results`

- **Goal:** the shape every tool follows.
- **Files:** `chatbot/src/chatbot/tools/{base,registry,pagination}.py`.
- **Details:** a tool is a typed async function plus a name, a description and
  a pydantic input model (which becomes the JSON schema for the model). Tools
  receive a `ToolContext` with the user and the trace context; the model cannot
  set it. Results use one envelope: `items`, `total`, `next_cursor`, and a
  size limit so no result floods the context.
- **Done when:** a toy tool is registered, its schema is generated, and
  pagination tests pass.

### [ ] C10 `feat(data): add demo dataset, curated views and the query_projects tool`

- **Goal:** the first real tool, with database-enforced read-only access.
- **Files:** `db/migrations/` (schemas `raw` and `curated`), `db/seed/`,
  `chatbot/src/chatbot/tools/query_projects.py`.
- **Details:** synthetic clients and projects in `raw`. Views in `curated`
  expose approved columns only. `chatbot_ro` gets `SELECT` on `curated` and
  nothing on `raw`. The tool takes typed filters (status, client, date range,
  text), builds parameterized SQL, and returns a paginated page. The reasons
  for views and a read-only role are in ARCHITECTURE.md section 3.4.
- **Done when:** the tool returns pages of projects, and tests show that
  `chatbot_ro` cannot read `raw` or write anywhere.

### [ ] C11 `feat(agent): add the agent loop, respond() and a CLI`

- **Goal:** the brain works end to end without any web layer.
- **Files:** `chatbot/src/chatbot/{agent,app,cli}.py`, `chatbot/src/chatbot/prompts/system.md`.
- **Details:** `respond(session_id, user_message, user)` loads history, calls
  the model, runs requested tools, repeats until a final answer or
  `AGENT_MAX_STEPS`, saves all new messages, and yields events. Tool errors go
  back to the model as tool results, not exceptions. `build_chatbot(settings)`
  is the composition root. `python -m chatbot.cli chat` gives a terminal chat.
- **Done when:** scripted-provider tests cover: plain answer, one tool call,
  several tool calls, tool error, step limit. The CLI answers a question about
  the demo projects.

### [ ] C12 `feat(context): add context builder with full and sliding-window strategies`

- **Goal:** separate what is stored from what the model sees.
- **Files:** `chatbot/src/chatbot/context/{base,full,sliding_window,tokens}.py`.
- **Details:** Strategy pattern. A strategy turns stored history into the
  messages for the next call. It never splits a tool call from its result.
  Token counting is approximate and sits behind a small interface.
- **Done when:** tests show the window keeps the last N turns, keeps tool
  pairs intact, and `CONTEXT_STRATEGY` selects the strategy.

### [ ] C13 `feat(tracing): add OpenTelemetry spans exported to Langfuse`

- **Goal:** every turn is visible as a trace.
- **Files:** `chatbot/src/chatbot/tracing.py`, small hooks in the agent,
  the provider call and the tool runner.
- **Details:** hierarchy session → turn → LLM call or tool call. LLM spans
  carry model, tokens in and out, cached tokens, cost and latency. Tool spans
  carry arguments, result size, latency and errors. The trace id is saved on
  the turn. With `TRACING_ENABLED=false` everything is a no-op.
- **Done when:** an in-memory exporter test asserts the span tree, and a CLI
  chat shows up in Langfuse grouped by session.

### [ ] C14 `feat(backend): add FastAPI service with auth, sessions and chat stream`

- **Goal:** the thin web layer.
- **Files:** `backend/pyproject.toml`, `backend/app/{main,auth,deps}.py`,
  `backend/app/routes/{sessions,chat,health}.py`, `backend/tests/`.
- **Details:** dev token auth behind one dependency, so SSO can replace it
  later. Endpoints: list sessions, create session, get a session with its
  messages, `POST /sessions/{id}/chat` streaming server-sent events, health.
  A user can only reach their own sessions. No AI logic here.
- **Done when:** API tests pass with the scripted provider, and `curl -N`
  shows a streamed answer.

### [ ] C15 `feat(frontend): add Streamlit chat UI`

- **Goal:** what the end user sees.
- **Files:** `frontend/pyproject.toml`, `frontend/app.py`, `frontend/api_client.py`.
- **Details:** sidebar with past sessions and "New chat". Clicking a session
  loads its full conversation. Answers stream in. Tool activity shows as a
  collapsed status line ("Searched projects: 12 results"). Only HTTP calls to
  the backend.
- **Done when:** a manual run-through works: new chat, ask, reload the page,
  reopen the session, continue it.

### [ ] C16 `feat(infra): add Dockerfiles and wire all services in compose`

- **Goal:** milestone M1.
- **Files:** `backend/Dockerfile`, `frontend/Dockerfile`, `.dockerignore`,
  `docker-compose.yml`.
- **Details:** services `postgres`, `migrate`, `backend`, `frontend`, with
  healthchecks and start order. Images are built with uv from the workspace
  lockfile.
- **Done when:** on a clean machine, `cp .env.example .env`, add keys,
  `docker compose up`, and the M1 description holds.

### [ ] C17 `ci: add lint, type-check and test pipeline`

- **Goal:** every push is checked.
- **Files:** `.github/workflows/ci.yml`.
- **Details:** jobs: lint (ruff), types (mypy), unit tests, integration tests
  with a Postgres service container. `llm` tests are excluded.
- **Done when:** the workflow is green on `main`.

---

## Phase 2: Company data (M2)

### [ ] C18 `feat(rag): add document schema and ingestion pipeline`

- **Goal:** documents in the database, ready for both kinds of search.
- **Files:** `db/migrations/` (schema `rag`), `chatbot/src/chatbot/rag/{chunking,embeddings,ingest}.py`,
  `db/seed/documents/`.
- **Details:** `documents` (source, title, metadata, allowed groups) and
  `chunks` (text, embedding vector, full-text vector, allowed groups, position).
  Structure-aware chunking with overlap. `Embedder` is a strategy. Ingestion is
  idempotent (content hash). Indexes: HNSW for vectors, GIN for full text and
  for groups.
- **Done when:** `python -m chatbot.rag.ingest db/seed/documents` loads the
  demo documents, and running it twice changes nothing.

### [ ] C19 `feat(rag): add hybrid retriever with access-control filter`

- **Goal:** good candidates, and only ones the user may see.
- **Files:** `chatbot/src/chatbot/rag/{lexical,vector,fusion,retriever}.py`.
- **Details:** lexical leg and vector leg each return candidates, merged with
  reciprocal rank fusion. The group filter and other metadata filters are
  applied inside both SQL queries. The lexical leg sits behind an interface
  (see ARCHITECTURE.md section 3.5).
- **Done when:** tests show that a user never gets a chunk outside their
  groups, and that hybrid beats each leg alone on a small labeled set.

### [ ] C20 `feat(rag): add reranker strategies and the search_docs tool`

- **Goal:** the second real tool.
- **Files:** `chatbot/src/chatbot/rag/rerank/`, `chatbot/src/chatbot/tools/search_docs.py`.
- **Details:** `Reranker` strategy: `none`, local cross-encoder, API. The tool
  takes a query and optional filters and returns a paginated list of compact
  hits (title, source, short snippet, chunk id, score) so the model can cite
  sources and ask for the next page.
- **Done when:** the chatbot answers a question from the demo documents with
  citations, and a user without access gets no answer from restricted ones.

### [ ] C21 `feat(security): add prompt-injection defenses for retrieved content`

- **Goal:** retrieved text is treated as data.
- **Files:** `chatbot/src/chatbot/safety.py`, prompt updates, frontend
  rendering rules, adversarial fixtures in `db/seed/documents/`.
- **Details:** tool results are wrapped and labeled as untrusted. The system
  prompt states that instructions inside tool results are never followed. The
  UI does not load remote images from model output. Seed documents include
  injected instructions for testing.
- **Done when:** `llm`-marked tests with the adversarial documents show that
  the model does not follow the injected instructions.

### [ ] C22 `refactor(tools): serve tools over MCP`

- **Goal:** tools become reusable by other clients.
- **Files:** `mcp_servers/pyproject.toml`, `mcp_servers/data_server.py`,
  `chatbot/src/chatbot/tools/mcp_client.py`, compose service.
- **Details:** the existing typed functions are registered on an MCP server
  without changing their logic. The agent discovers tools through an MCP client
  that implements the same tool interface, so the agent loop does not change.
  The user's identity travels from the backend to the server in a signed
  token; the server verifies it and never trusts identity from tool arguments.
- **Done when:** the same agent tests pass with tools in-process and over MCP,
  and an external MCP client can list and call the tools.

---

## Phase 3: Long conversations (M3)

### [ ] C23 `feat(context): add tool-result compaction, running summary and prompt caching`

- **Goal:** long sessions stay cheap, fast and accurate.
- **Files:** `chatbot/src/chatbot/context/{compact_tools,summary,pipeline}.py`,
  migration for the summary columns.
- **Details:** `compact_tools` replaces old tool results with a one-line note.
  `summary` summarizes older turns with `LLM_SUMMARY_MODEL` once past a
  threshold and stores the result on the session. Strategies compose in a
  pipeline. Cache breakpoints go after the system prompt and tool definitions.
  Summaries happen in large, infrequent steps to keep the prefix stable.
- **Done when:** a 60-turn scripted session stays under
  `CONTEXT_MAX_INPUT_TOKENS`, and traces show cached tokens on later turns.

### [ ] C24 `feat(memory): add user memory tool behind a feature flag`

- **Goal:** cross-session facts about the user.
- **Files:** migration for `user_memory`, `chatbot/src/chatbot/tools/memory.py`,
  `chatbot/src/chatbot/store/memory_store.py`, backend endpoints to list and
  delete, a small "What I remember" panel in the UI.
- **Details:** one tool with `remember`, `recall` and `forget`. Scoped to the
  calling user by `ToolContext`. Only facts the user stated are saved, never
  content from retrieved documents. Off unless `MEMORY_ENABLED=true`.
- **Done when:** a fact stated in one session is used in a new one, and the
  user can see and delete it.

---

## Phase 4: Governance (M4)

### [ ] C25 `feat(gateway): add fallbacks, retries and cost accounting`

- **Goal:** one place for reliability and spend.
- **Files:** `chatbot/src/chatbot/llm/{gateway,pricing}.py`.
- **Details:** Decorator over `LLMProvider`. Retries with backoff on transient
  errors, then the fallback chain from `LLM_FALLBACKS`. A fallback never
  happens after text has started streaming. Cost is computed from a price
  table per model and written to the turn and the span.
- **Done when:** fault-injection tests show retry then fallback, and each turn
  has a cost.

### [ ] C26 `feat(limits): add per-user rate limits and cost caps`

- **Goal:** no single user can run up the bill.
- **Files:** `chatbot/src/chatbot/limits.py`, backend dependency.
- **Details:** requests per minute and a daily cost cap, both computed from the
  `turns` table, so no extra infrastructure is needed. Clear error events to
  the UI when a limit is hit.
- **Done when:** tests show a blocked request after the limit, and the UI
  shows a readable message.

### [ ] C27 `feat(audit): add append-only audit log`

- **Goal:** a compliance record that does not depend on Langfuse.
- **Files:** migration for `audit_log`, `chatbot/src/chatbot/audit.py`.
- **Details:** one row per turn and per tool call: who, when, what was asked,
  which tool with which arguments, which sources were returned.
  `chatbot_app` can insert only; a trigger rejects update and delete.
- **Done when:** tests show inserts succeed and update or delete fails for
  every application role.

### [ ] C28 `feat(feedback): add thumbs up/down linked to traces`

- **Goal:** find bad answers quickly.
- **Files:** migration for `feedback`, backend endpoint, UI buttons.
- **Details:** rating and optional comment per turn, stored in Postgres and
  sent to Langfuse as a score on the turn's trace.
- **Done when:** a thumbs-down in the UI appears on the trace in Langfuse.

---

## Phase 5: Quality loop (M5)

### [ ] C29 `feat(evals): add eval dataset, runner and scorers`

- **Goal:** measure answer quality, not guess it.
- **Files:** `evals/pyproject.toml`, `evals/datasets/*.yaml`, `evals/runner.py`,
  `evals/scorers/`.
- **Details:** each case has a question, a user, the expected tool calls,
  expected sources and reference facts. The runner calls `respond` directly.
  Scorers: tool choice, retrieval hit rate, answer facts (model-graded with a
  rubric), refusals for no-access cases, injection resistance, cost and
  latency. Results are written as JSON and Markdown and sent to Langfuse.
- **Done when:** `uv run python -m evals.runner` prints a score table for the
  demo data.

### [ ] C30 `ci(evals): run evals on prompt, tool and model changes`

- **Goal:** regressions are caught before merge.
- **Files:** `.github/workflows/evals.yml`, `evals/baseline.json`.
- **Details:** the job runs when prompts, tools, retrieval, context or model
  settings change. It fails if a score drops below the baseline by more than
  a set tolerance.
- **Done when:** a deliberately worse prompt fails the pipeline.

### [ ] C31 `docs: add runbook, debugging guide and Langfuse self-hosting notes`

- **Goal:** someone else can run and debug the system.
- **Files:** `docs/RUNBOOK.md`, `docs/DEBUGGING.md`, optional
  `docker-compose.langfuse.yml`.
- **Details:** how to follow one bad answer from the UI to its trace, its
  stored messages and its audit rows. How to replay a session. How to add a
  provider, a tool, a context strategy.
- **Done when:** a new reader can go from clone to a debugged turn using only
  the docs.

---

## Requirement coverage

| Requirement from the brief | Commits |
|---|---|
| Project setup, docs, changelog, plan | C01, C02 |
| Chat UI | C15 |
| API and agent loop | C11, C14 |
| LLM gateway (fallbacks, cost) | C25 |
| Pluggable SDKs (Anthropic, OpenAI) | C07, C08 |
| MCP tool server | C22 |
| `query_projects` over curated views with a read-only role | C03, C10 |
| `search_docs`: hybrid search, reranker, access control | C18, C19, C20 |
| Compact, paginated tool results | C09 |
| History in Postgres with JSONB tool calls | C05, C06 |
| Context trimming and summarizing strategies | C12, C23 |
| Memory | C24 |
| Tracing to Langfuse | C13 |
| Append-only audit log | C27 |
| Docker Compose, one container per part | C03, C16 |
| Evals on every change | C29, C30 |
| Prompt-injection defenses | C21 |
| Feedback linked to traces | C28 |
| Rate limits and cost caps | C26 |
| Debuggability | C04, C11, C13, C31 |
