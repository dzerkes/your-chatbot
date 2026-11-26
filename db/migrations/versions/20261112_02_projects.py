"""Create raw company data and approved read-only views.

Revision ID: 20261112_02_projects
Revises: 20261112_01_history
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from chatbot.config import Settings

revision: str = "20261112_02_projects"
down_revision: str | None = "20261112_01_history"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA raw")
    op.execute("CREATE SCHEMA curated")
    op.execute("REVOKE ALL ON SCHEMA raw, curated FROM PUBLIC")
    op.create_table(
        "clients",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("contact_email", sa.Text(), nullable=False),
        sa.Column("internal_notes", sa.Text(), nullable=False, server_default=""),
        sa.CheckConstraint("length(btrim(name)) > 0", name="ck_clients_name_nonempty"),
        sa.UniqueConstraint("name", name="uq_clients_name"),
        schema="raw",
    )
    op.create_table(
        "projects",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=False),
        sa.Column("client_id", sa.BigInteger(), sa.ForeignKey("raw.clients.id"), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column("budget_usd", sa.Numeric(12, 2), nullable=False),
        sa.Column("internal_notes", sa.Text(), nullable=False, server_default=""),
        sa.CheckConstraint("length(btrim(name)) > 0", name="ck_projects_name_nonempty"),
        sa.CheckConstraint(
            "status IN ('planned', 'active', 'on_hold', 'completed', 'cancelled')",
            name="ck_projects_status",
        ),
        sa.CheckConstraint(
            "end_date IS NULL OR end_date >= start_date", name="ck_projects_date_range"
        ),
        sa.CheckConstraint("budget_usd >= 0", name="ck_projects_budget_nonnegative"),
        schema="raw",
    )
    op.create_index("ix_projects_client_id", "projects", ["client_id"], schema="raw")
    op.create_index(
        "ix_projects_status_start_date", "projects", ["status", "start_date"], schema="raw"
    )
    # Default view-owner permissions let the reader use these views without
    # granting it privileges on the raw schema or tables. Keep the column lists explicit.
    op.execute("CREATE VIEW curated.clients AS SELECT id, name FROM raw.clients")
    op.execute(
        "CREATE VIEW curated.projects AS "
        "SELECT p.id, p.name, p.description, p.status, p.start_date, p.end_date, "
        "c.id AS client_id, c.name AS client_name "
        "FROM raw.projects AS p JOIN raw.clients AS c ON c.id = p.client_id"
    )
    reader = op.get_bind().dialect.identifier_preparer.quote_identifier(Settings().ro_db_user)
    op.execute(sa.text(f"REVOKE ALL ON SCHEMA raw FROM {reader}"))
    op.execute(sa.text(f"REVOKE ALL ON ALL TABLES IN SCHEMA raw FROM {reader}, PUBLIC"))
    op.execute("REVOKE ALL ON ALL TABLES IN SCHEMA curated FROM PUBLIC")
    op.execute(sa.text(f"GRANT USAGE ON SCHEMA curated TO {reader}"))
    op.execute(sa.text(f"GRANT SELECT ON curated.clients, curated.projects TO {reader}"))


def downgrade() -> None:
    op.execute("DROP VIEW curated.projects")
    op.execute("DROP VIEW curated.clients")
    op.drop_table("projects", schema="raw")
    op.drop_table("clients", schema="raw")
    op.execute("DROP SCHEMA curated")
    op.execute("DROP SCHEMA raw")
