"""Create the rag schema: documents and their searchable chunks.

Revision ID: 20261125_03_rag
Revises: 20261112_02_projects
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from chatbot.config import Settings

revision: str = "20261125_03_rag"
down_revision: str | None = "20261112_02_projects"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    settings = Settings()
    bind = op.get_bind()
    # pgvector needs a superuser to install; db/init does it for the app database.
    if not bind.execute(sa.text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).scalar():
        raise RuntimeError(
            "The pgvector extension is missing. A superuser must run CREATE EXTENSION vector "
            "in this database (db/init/01_roles.sql does this for Compose)."
        )
    dim = int(settings.embedding_dim)  # int() keeps the DDL below free of injected text

    op.execute("CREATE SCHEMA rag")
    op.execute("REVOKE ALL ON SCHEMA rag FROM PUBLIC")
    op.execute(
        """
        CREATE TABLE rag.documents (
            id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            source text NOT NULL,
            title text NOT NULL,
            metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
            allowed_groups text[] NOT NULL,
            content_hash text NOT NULL,
            embedding_model text NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_documents_source UNIQUE (source),
            CONSTRAINT ck_documents_source_nonempty CHECK (length(btrim(source)) > 0),
            CONSTRAINT ck_documents_title_nonempty CHECK (length(btrim(title)) > 0),
            CONSTRAINT ck_documents_metadata_object CHECK (jsonb_typeof(metadata) = 'object'),
            -- An empty list would be ambiguous; a document must name who may read it.
            CONSTRAINT ck_documents_groups_nonempty CHECK (cardinality(allowed_groups) > 0)
        )
        """
    )
    op.execute(
        f"""
        CREATE TABLE rag.chunks (
            id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            document_id bigint NOT NULL REFERENCES rag.documents (id) ON DELETE CASCADE,
            position integer NOT NULL,
            heading text NOT NULL,
            text text NOT NULL,
            embedding vector({dim}) NOT NULL,
            search_vector tsvector GENERATED ALWAYS AS (
                setweight(to_tsvector('english', heading), 'A')
                || setweight(to_tsvector('english', text), 'B')
            ) STORED,
            -- Copied from the document so both search legs can filter without a join.
            allowed_groups text[] NOT NULL,
            CONSTRAINT uq_chunks_document_position UNIQUE (document_id, position),
            CONSTRAINT ck_chunks_position_nonnegative CHECK (position >= 0),
            CONSTRAINT ck_chunks_text_nonempty CHECK (length(btrim(text)) > 0),
            CONSTRAINT ck_chunks_groups_nonempty CHECK (cardinality(allowed_groups) > 0)
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_chunks_embedding ON rag.chunks USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute("CREATE INDEX ix_chunks_search_vector ON rag.chunks USING gin (search_vector)")
    op.execute("CREATE INDEX ix_chunks_allowed_groups ON rag.chunks USING gin (allowed_groups)")
    op.execute(
        "CREATE INDEX ix_documents_allowed_groups ON rag.documents USING gin (allowed_groups)"
    )

    # Ingestion writes as the owner; the tool role only reads.
    reader = bind.dialect.identifier_preparer.quote_identifier(settings.ro_db_user)
    op.execute("REVOKE ALL ON ALL TABLES IN SCHEMA rag FROM PUBLIC")
    op.execute(sa.text(f"GRANT USAGE ON SCHEMA rag TO {reader}"))
    op.execute(sa.text(f"GRANT SELECT ON rag.documents, rag.chunks TO {reader}"))


def downgrade() -> None:
    op.execute("DROP TABLE rag.chunks")
    op.execute("DROP TABLE rag.documents")
    op.execute("DROP SCHEMA rag")
