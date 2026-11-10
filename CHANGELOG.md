# Changelog

All important changes to this project are recorded here, newest first.

- Each entry names the planned commit id from
  [docs/plan/PROJECT_PLAN.md](docs/plan/PROJECT_PLAN.md) and the git hash once
  the commit exists.
- Entry types: Added, Changed, Fixed, Removed, Security, Decision.
- Dates are in ISO format (YYYY-MM-DD).

## Unreleased

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
