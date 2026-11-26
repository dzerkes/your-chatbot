"""Append-only turn and tool audit records."""

from alembic import op
from chatbot.config import Settings

revision = "20261126_06_audit"
down_revision = "20261125_05_user_memory"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE audit_log (
            id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            recorded_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            user_id uuid NOT NULL,
            session_id uuid NOT NULL,
            turn_id uuid NOT NULL,
            trace_id text,
            event_type text NOT NULL,
            question text,
            call_id text,
            tool_name text,
            arguments jsonb,
            outcome text,
            sources jsonb NOT NULL DEFAULT '[]' CHECK (jsonb_typeof(sources) = 'array'),
            CHECK (
                (event_type = 'turn' AND question IS NOT NULL AND call_id IS NULL
                 AND tool_name IS NULL AND arguments IS NULL AND outcome IS NULL)
                OR
                (event_type = 'tool' AND question IS NULL AND call_id IS NOT NULL
                 AND tool_name IS NOT NULL AND arguments IS NOT NULL
                 AND jsonb_typeof(arguments) = 'object' AND outcome IS NOT NULL
                 AND outcome IN ('returned', 'failed', 'interrupted'))
            )
        )
    """)
    op.execute("CREATE UNIQUE INDEX audit_turn ON audit_log(turn_id) WHERE event_type = 'turn'")
    op.execute(
        "CREATE UNIQUE INDEX audit_call ON audit_log(turn_id, call_id) WHERE event_type = 'tool'"
    )
    op.execute("""
        CREATE FUNCTION reject_audit_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'audit_log is append-only';
        END;
        $$
    """)
    op.execute("""
        CREATE TRIGGER audit_append_only BEFORE UPDATE OR DELETE OR TRUNCATE ON audit_log
        FOR EACH STATEMENT EXECUTE FUNCTION reject_audit_mutation()
    """)
    role = op.get_bind().dialect.identifier_preparer.quote_identifier(Settings().app_db_user)
    op.execute(
        "GRANT INSERT (user_id, session_id, turn_id, trace_id, event_type, question, "
        f"call_id, tool_name, arguments, outcome, sources) ON audit_log TO {role}"
    )


def downgrade() -> None:
    op.execute("DROP TABLE audit_log")
    op.execute("DROP FUNCTION reject_audit_mutation()")
