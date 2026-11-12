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
docker compose run --rm migrate uv run --locked alembic -c db/alembic.ini current
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

To remove the company-data schemas while keeping chat history:

```bash
docker compose run --rm migrate uv run --locked alembic -c db/alembic.ini downgrade 20261112_01_history
```

This deletes all raw clients/projects, including local additions and demo rows.

To remove the history tables in a disposable database, run:

```bash
docker compose run --rm migrate uv run --locked alembic -c db/alembic.ini downgrade base
```

Downgrading to `base` deletes company data as well as all users, sessions, turns
and messages. Integration tests use temporary databases for migrations, seed
idempotency, tool queries and permission checks, then delete those databases.
