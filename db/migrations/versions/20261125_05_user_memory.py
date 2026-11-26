"""Create user_memory: facts users asked the assistant to remember across sessions.

Revision ID: 20261125_05_user_memory
Revises: 20261125_04_session_summary
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from chatbot.config import Settings

revision: str = "20261125_05_user_memory"
down_revision: str | None = "20261125_04_session_summary"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE user_memory (
            id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            user_id uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
            fact text NOT NULL,
            -- The user's own words that support the fact, checked when it was saved.
            evidence text NOT NULL,
            source_session_id uuid REFERENCES sessions (id) ON DELETE SET NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT ck_user_memory_fact_length CHECK (length(btrim(fact)) BETWEEN 1 AND 300),
            CONSTRAINT ck_user_memory_evidence_length
                CHECK (length(btrim(evidence)) BETWEEN 1 AND 300)
        )
        """
    )
    op.execute("CREATE UNIQUE INDEX uq_user_memory_fact ON user_memory (user_id, lower(fact))")
    op.execute("CREATE INDEX ix_user_memory_user_created ON user_memory (user_id, created_at DESC)")
    # Facts are added and forgotten, never rewritten in place: no UPDATE grant.
    app_role = op.get_bind().dialect.identifier_preparer.quote_identifier(Settings().app_db_user)
    op.execute(sa.text(f"GRANT SELECT, INSERT, DELETE ON user_memory TO {app_role}"))


def downgrade() -> None:
    op.execute("DROP TABLE user_memory")
