# Database migrations and demo data

The database roles and pgvector extension are initialized by `db/init/01_roles.sql`
when Postgres starts with an empty volume. Alembic handles versioned application
schema changes. The migration files live in `db/migrations/` so both the chatbot
package and future backend can use the same schema.

Start Postgres, then run the one-shot migration service:

```bash
make up
make migrate
```

The service mounts the project read-only, installs dependencies from `uv.lock`
into its own temporary environment, and connects as `POSTGRES_USER` from `.env`
(`chatbot_owner` by default). It waits for a healthy Postgres service. The
database URL is built by `chatbot.config.Settings`, with `POSTGRES_HOST=postgres`
and port 5432 inside Compose; special characters in passwords are handled by
SQLAlchemy's `URL.create`. Re-running the command when already at `head` changes
nothing. To see the current revision:

```bash
docker compose run --rm migrate alembic -c db/alembic.ini current
```

To run Alembic from the host, use `uv run alembic -c db/alembic.ini upgrade head`.
The host command reads `POSTGRES_HOST` and `POSTGRES_PORT` from `.env` (or shell
overrides), so the configured port must match the published Compose port.

The first revision creates:

| Table | Stored data |
|---|---|
| `users` | External identity, display name, groups and creation time. |
| `sessions` | Owner, optional title and summary, creation and update times. |
| `turns` | Session, status, trace ID, usage counters, cost, latency and times. |
| `messages` | Session and turn, session-wide sequence, role, JSONB content blocks, optional model and JSONB usage. |

`messages.content` is a JSONB array of provider-neutral `Text`, `ToolUse` or
`ToolResult` blocks as defined in `chatbot.types`. A composite foreign key
ensures every message references a turn in the same session. Unique session-wide
sequence numbers give each message a stable order. Foreign keys cascade from a
deleted user through sessions and turns to messages. The application must update
`updated_at` when it changes a session or turn. The C06 history store does this
when it updates session metadata or saves a turn.

`chatbot_app` can select, insert, update and delete rows in those four tables.
It cannot alter the schema or read Alembic's version table. `chatbot_ro` has no
history-table privileges.

## Company data and curated views

The C10 revision creates `raw.clients` and `raw.projects`. Raw data includes
synthetic contact emails, budgets and internal notes. The views expose only:

| View | Approved columns |
|---|---|
| `curated.clients` | `id`, `name` |
| `curated.projects` | `id`, `name`, `description`, `status`, `start_date`, `end_date`, `client_id`, `client_name` |

The migration owner owns both schemas and their objects. The configured
`RO_DB_USER` gets schema usage and `SELECT` on those two views, with no grants
on raw tables or history. Views use the owner's permissions to read their base
tables. The application role has no company-data grants; the tool opens a
separate pool with the read-only credentials. Grants name specific views so
new objects require an explicit decision.

Load the synthetic dataset explicitly:

```bash
make seed                 # migrate, then load 4 clients and 12 projects as owner
```

`db/seed/01_demo.sql` runs in one transaction. It uses fixed IDs and skips
existing IDs, so running it again adds no duplicates and preserves local edits.
Migrations never load demo rows automatically. Statuses are `planned`, `active`,
`on_hold`, `completed` and `cancelled`; dates are fixed so examples remain
reproducible.

## Documents for search (`rag`)

`rag.documents` holds one row per source file: path, title, metadata, the
groups allowed to read it, a content hash and the embedding model. `rag.chunks`
holds the searchable pieces: heading path, text, an `EMBEDDING_DIM` vector, a
generated weighted full-text vector, and a copy of the document's groups so both
search legs can filter without a join. Indexes: HNSW (cosine) on embeddings and
GIN on full text and on groups. A document must name at least one group.

`RO_DB_USER` gets `SELECT` on both tables; the application role gets nothing.
Ingestion is an operator job that writes as the migration owner:

```bash
make ingest               # migrate, then load db/seed/documents with EMBEDDING_MODEL
uv run python -m chatbot.rag.ingest DIRECTORY [--prune]
```

Each `*.md` file needs front matter with `groups:` (see
`db/seed/documents/README.md.txt`). Unchanged documents are skipped by content
hash; a changed one has its chunks replaced in one transaction. `--prune`
deletes stored documents whose file is gone. The default `local:` model
(`BAAI/bge-small-en-v1.5`) downloads once to `EMBEDDING_CACHE_DIR` and runs on
this machine. The vector size is fixed when the migration runs, so pick
`EMBEDDING_DIM` first; the CLI refuses a model whose size does not match.

The migration requires the `vector` extension, which only a superuser can
create; `db/init/01_roles.sql` creates it in the Compose database.

To remove the company-data schemas while keeping chat history:

```bash
docker compose run --rm migrate alembic -c db/alembic.ini downgrade 20261112_01_history
```

This deletes all raw clients/projects, including local additions and demo rows,
and all ingested documents.

To remove the history tables in a disposable database, run:

```bash
docker compose run --rm migrate alembic -c db/alembic.ini downgrade base
```

Downgrading to `base` deletes company data as well as all users, sessions, turns
and messages. Integration tests use temporary databases for migrations, seed
idempotency, tool queries and permission checks, then delete those databases.


## Audit records

The CLI and backend append an `audit_log` row before the first model call of
each admitted turn, and another when each tool execution ends. The rows share
user, session, turn and trace IDs. Document searches record the returned paths
and chunk IDs. Project and memory calls record the returned row IDs. Questions
and tool arguments are stored verbatim and may contain sensitive information.

The database assigns timestamps. The application role can insert event data
and the read-only tool role has no access. The owner can inspect a turn with
`SELECT * FROM audit_log WHERE turn_id = $1 ORDER BY id`. A trigger rejects
updates, deletes and truncation, including ordinary owner-issued statements.
Audit IDs have no cascading references to mutable history tables.

## Feedback

`feedback` holds the user's current rating (`up` or `down`) and optional comment
for a completed or failed turn. A new vote on the same turn replaces the rating
and comment and keeps the original `created_at`. The application role can
select, insert and update rows. Rows are deleted with their turn.

`next_export_at` is set when a vote still has to reach Langfuse and cleared once
Langfuse accepts or rejects it. To find bad answers, the owner can run
`SELECT f.*, t.trace_id FROM feedback f JOIN turns t ON t.id = f.turn_id
WHERE f.rating = 'down' ORDER BY f.created_at DESC`.
