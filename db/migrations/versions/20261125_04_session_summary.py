"""Record which messages a session's running summary covers.

Revision ID: 20261125_04_session_summary
Revises: 20261125_03_rag
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261125_04_session_summary"
down_revision: str | None = "20261125_03_rag"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # sessions.summary exists since the history migration; this marks its coverage:
    # every message with sequence <= summary_through_sequence is summarized.
    op.add_column("sessions", sa.Column("summary_through_sequence", sa.Integer(), nullable=True))
    op.add_column(
        "sessions", sa.Column("summary_updated_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.create_check_constraint(
        "ck_sessions_summary_coverage",
        "sessions",
        "(summary_through_sequence IS NULL) = (summary IS NULL) "
        "AND (summary_through_sequence IS NULL) = (summary_updated_at IS NULL) "
        "AND (summary_through_sequence IS NULL OR summary_through_sequence >= 0)",
    )


def downgrade() -> None:
    op.drop_constraint("ck_sessions_summary_coverage", "sessions", type_="check")
    op.drop_column("sessions", "summary_updated_at")
    op.drop_column("sessions", "summary_through_sequence")
