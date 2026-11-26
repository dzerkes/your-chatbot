"""Current user feedback per turn and durable Langfuse score delivery."""

from alembic import op
from chatbot.config import Settings

revision = "20261126_07_feedback"
down_revision = "20261126_06_audit"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE feedback (
            turn_id uuid PRIMARY KEY REFERENCES turns(id) ON DELETE CASCADE,
            rating text NOT NULL CHECK (rating IN ('up', 'down')),
            comment text CHECK (length(comment) <= 2000),
            created_at timestamptz NOT NULL DEFAULT now(),
            next_export_at timestamptz
        )
    """)
    op.execute("""
        CREATE INDEX feedback_pending ON feedback(next_export_at)
        WHERE next_export_at IS NOT NULL
    """)
    role = op.get_bind().dialect.identifier_preparer.quote_identifier(Settings().app_db_user)
    op.execute(f"GRANT SELECT, INSERT, UPDATE ON feedback TO {role}")


def downgrade() -> None:
    op.execute("DROP TABLE feedback")
