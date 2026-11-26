"""Create users, sessions, turns and messages.

Revision ID: 20261112_01_history
Revises:
Create Date: 2026-11-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from chatbot.config import Settings
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

revision: str = "20261112_01_history"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column(
            "id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")
        ),
        sa.Column("external_id", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column(
            "groups", ARRAY(sa.Text()), nullable=False, server_default=sa.text("'{}'::text[]")
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint("length(btrim(external_id)) > 0", name="ck_users_external_id_nonempty"),
        sa.CheckConstraint(
            "length(btrim(display_name)) > 0", name="ck_users_display_name_nonempty"
        ),
        sa.UniqueConstraint("external_id", name="uq_users_external_id"),
    )

    op.create_table(
        "sessions",
        sa.Column(
            "id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")
        ),
        sa.Column(
            "owner_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_sessions_owner_updated_at", "sessions", ["owner_id", "updated_at"])

    op.create_table(
        "turns",
        sa.Column(
            "id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")
        ),
        sa.Column(
            "session_id",
            UUID(as_uuid=True),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("trace_id", sa.Text(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cache_read_input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cache_creation_input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Numeric(12, 6), nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')", name="ck_turns_status"
        ),
        sa.CheckConstraint("input_tokens >= 0", name="ck_turns_input_tokens_nonnegative"),
        sa.CheckConstraint("output_tokens >= 0", name="ck_turns_output_tokens_nonnegative"),
        sa.CheckConstraint("cache_read_input_tokens >= 0", name="ck_turns_cache_read_nonnegative"),
        sa.CheckConstraint(
            "cache_creation_input_tokens >= 0", name="ck_turns_cache_creation_nonnegative"
        ),
        sa.CheckConstraint("cost_usd >= 0", name="ck_turns_cost_nonnegative"),
        sa.CheckConstraint(
            "latency_ms IS NULL OR latency_ms >= 0", name="ck_turns_latency_nonnegative"
        ),
        sa.UniqueConstraint("id", "session_id", name="uq_turns_id_session_id"),
    )
    op.create_index("ix_turns_session_created_at", "turns", ["session_id", "created_at"])

    op.create_table(
        "messages",
        sa.Column(
            "id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")
        ),
        sa.Column("session_id", UUID(as_uuid=True), nullable=False),
        sa.Column("turn_id", UUID(as_uuid=True), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", JSONB(), nullable=False),
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column("usage", JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["turn_id", "session_id"],
            ["turns.id", "turns.session_id"],
            name="fk_messages_turn_session",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("session_id", "sequence", name="uq_messages_session_sequence"),
        sa.CheckConstraint("sequence >= 0", name="ck_messages_sequence_nonnegative"),
        sa.CheckConstraint(
            "role IN ('system', 'user', 'assistant', 'tool')", name="ck_messages_role"
        ),
        sa.CheckConstraint("jsonb_typeof(content) = 'array'", name="ck_messages_content_array"),
        sa.CheckConstraint(
            "usage IS NULL OR jsonb_typeof(usage) = 'object'", name="ck_messages_usage_object"
        ),
    )
    op.create_index("ix_messages_turn_session", "messages", ["turn_id", "session_id"])

    # No privileges for chatbot_ro. The app gets row operations, not ownership,
    # schema creation, DDL, or access to Alembic's version table.
    app_role = op.get_bind().dialect.identifier_preparer.quote_identifier(Settings().app_db_user)
    op.execute(
        sa.text(
            "GRANT SELECT, INSERT, UPDATE, DELETE "
            f"ON users, sessions, turns, messages TO {app_role}"
        )
    )


def downgrade() -> None:
    op.drop_index("ix_messages_turn_session", table_name="messages")
    op.drop_table("messages")
    op.drop_index("ix_turns_session_created_at", table_name="turns")
    op.drop_table("turns")
    op.drop_index("ix_sessions_owner_updated_at", table_name="sessions")
    op.drop_table("sessions")
    op.drop_table("users")
