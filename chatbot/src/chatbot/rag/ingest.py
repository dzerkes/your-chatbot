"""Load Markdown documents into rag.documents and rag.chunks.

    python -m chatbot.rag.ingest DIRECTORY [--prune]

Each ``*.md`` file under DIRECTORY is one document, identified by its path
relative to DIRECTORY. It starts with front matter naming who may read it:

    ---
    title: Employee handbook
    groups: employees
    tags: hr, policy
    ---

``groups`` is required: a document without it is rejected rather than made
readable by everyone. ``title`` defaults to the first ``#`` heading, then the
file name. Other keys are kept as metadata (``tags`` as a list).

Ingestion is idempotent. A content hash covers the text, title, groups,
metadata, chunker version and embedding model, so an unchanged document is
skipped and a changed one has its chunks replaced in one transaction.
``--prune`` also deletes stored documents whose file no longer exists.

It connects as the schema owner: this is an operator job, like migrations.
"""

import argparse
import asyncio
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from psycopg import AsyncConnection
from psycopg import Error as DatabaseError
from psycopg.conninfo import make_conninfo
from pydantic import ValidationError

from chatbot.config import Settings
from chatbot.errors import ChatbotError, IngestError
from chatbot.logging import configure_logging
from chatbot.rag.chunking import CHUNKER_VERSION, Chunk, chunk_markdown
from chatbot.rag.embeddings import Embedder, build_embedder, vector_literal

_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
_GROUP = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_LIST_KEYS = {"tags"}


@dataclass(frozen=True)
class SourceDocument:
    source: str
    title: str
    groups: tuple[str, ...]
    metadata: dict[str, str | list[str]]
    body: str


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    chunks_written: int = 0

    def summary(self) -> str:
        total = len(self.added) + len(self.updated) + len(self.unchanged)
        line = (
            f"{total} documents: {len(self.added)} added, {len(self.updated)} updated, "
            f"{len(self.unchanged)} unchanged; {self.chunks_written} chunks written"
        )
        return f"{line}; {len(self.removed)} removed" if self.removed else line


def parse_document(source: str, raw: str) -> SourceDocument:
    lines = raw.replace("\r\n", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        raise IngestError(f"{source}: missing front matter with a groups line")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise IngestError(f"{source}: front matter is not closed with ---") from None

    fields: dict[str, str] = {}
    for number, line in enumerate(lines[1:end], start=2):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition(":")
        key = key.strip().lower()
        if not separator or not _KEY.match(key):
            raise IngestError(f"{source}:{number}: expected 'key: value'")
        if key in fields:
            raise IngestError(f"{source}:{number}: duplicate key {key!r}")
        fields[key] = value.strip()

    groups = sorted({g.strip().lower() for g in fields.pop("groups", "").split(",") if g.strip()})
    if not groups:
        raise IngestError(f"{source}: front matter must list at least one group")
    if bad := [g for g in groups if not _GROUP.match(g)]:
        raise IngestError(f"{source}: invalid group names {bad}")

    body = "\n".join(lines[end + 1 :]).strip()
    if not body:
        raise IngestError(f"{source}: document has no text")
    title = fields.pop("title", "") or _first_heading(body) or Path(source).stem
    metadata: dict[str, str | list[str]] = {
        key: [item.strip() for item in value.split(",") if item.strip()]
        if key in _LIST_KEYS
        else value
        for key, value in sorted(fields.items())
    }
    return SourceDocument(source, title, tuple(groups), metadata, body)


def _first_heading(body: str) -> str | None:
    for line in body.split("\n"):
        if line.startswith("# "):
            return line[2:].strip() or None
    return None


def load_documents(directory: Path) -> list[SourceDocument]:
    if not directory.is_dir():
        raise IngestError(f"{directory} is not a directory")
    documents = []
    for path in sorted(directory.rglob("*.md")):
        relative = path.relative_to(directory)
        if any(part.startswith(".") for part in relative.parts) or not path.is_file():
            continue
        documents.append(parse_document(relative.as_posix(), path.read_text(encoding="utf-8")))
    if not documents:
        raise IngestError(f"No Markdown documents found in {directory}")
    return documents


def content_hash(document: SourceDocument, embedder: Embedder) -> str:
    payload = {
        "chunker": CHUNKER_VERSION,
        "embedder": embedder.name,
        "dim": embedder.dim,
        "title": document.title,
        "groups": list(document.groups),
        "metadata": document.metadata,
        "body": document.body,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


async def _check_schema(conn: AsyncConnection, embedder: Embedder) -> None:
    cursor = await conn.execute(
        "SELECT a.atttypmod FROM pg_attribute AS a "
        "WHERE a.attrelid = to_regclass('rag.chunks') AND a.attname = 'embedding'"
    )
    row = await cursor.fetchone()
    if row is None:
        raise IngestError("The rag schema is missing; run the migrations first")
    if row[0] != embedder.dim:
        raise IngestError(
            f"rag.chunks stores {row[0]}-dimension vectors but {embedder.name} is configured "
            f"for {embedder.dim}; align EMBEDDING_DIM with the migrated schema"
        )


async def ingest_documents(
    documents: Sequence[SourceDocument],
    settings: Settings,
    embedder: Embedder,
    *,
    prune: bool = False,
) -> IngestReport:
    sources = [document.source for document in documents]
    if len(set(sources)) != len(sources):
        raise IngestError("Document sources must be unique")
    report = IngestReport()
    conninfo = make_conninfo(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password.get_secret_value(),
    )
    async with await AsyncConnection.connect(conninfo) as conn:
        await _check_schema(conn, embedder)
        await conn.commit()
        for document in documents:
            digest = content_hash(document, embedder)
            cursor = await conn.execute(
                "SELECT content_hash FROM rag.documents WHERE source = %s", (document.source,)
            )
            existing = await cursor.fetchone()
            await conn.commit()
            if existing is not None and existing[0] == digest:
                report.unchanged.append(document.source)
                continue

            chunks = chunk_markdown(document.body, title=document.title)
            # Embed outside the transaction so no locks are held while the model runs.
            vectors = await embedder.embed_documents([_embedding_text(c) for c in chunks])
            async with conn.transaction():
                cursor = await conn.execute(
                    """
                    INSERT INTO rag.documents
                        (source, title, metadata, allowed_groups, content_hash, embedding_model)
                    VALUES (%s, %s, %s::jsonb, %s, %s, %s)
                    ON CONFLICT (source) DO UPDATE SET
                        title = EXCLUDED.title,
                        metadata = EXCLUDED.metadata,
                        allowed_groups = EXCLUDED.allowed_groups,
                        content_hash = EXCLUDED.content_hash,
                        embedding_model = EXCLUDED.embedding_model,
                        updated_at = now()
                    RETURNING id
                    """,
                    (
                        document.source,
                        document.title,
                        json.dumps(document.metadata),
                        list(document.groups),
                        digest,
                        embedder.name,
                    ),
                )
                row = await cursor.fetchone()
                assert row is not None
                document_id = row[0]
                await conn.execute("DELETE FROM rag.chunks WHERE document_id = %s", (document_id,))
                async with conn.cursor() as insert:
                    await insert.executemany(
                        "INSERT INTO rag.chunks "
                        "(document_id, position, heading, text, embedding, allowed_groups) "
                        "VALUES (%s, %s, %s, %s, %s::vector, %s)",
                        [
                            (
                                document_id,
                                chunk.position,
                                chunk.heading,
                                chunk.text,
                                vector_literal(vector),
                                list(document.groups),
                            )
                            for chunk, vector in zip(chunks, vectors, strict=True)
                        ],
                    )
            (report.added if existing is None else report.updated).append(document.source)
            report.chunks_written += len(chunks)

        if prune:
            async with conn.transaction():
                cursor = await conn.execute(
                    "DELETE FROM rag.documents WHERE NOT (source = ANY(%s)) RETURNING source",
                    (sources,),
                )
                report.removed = sorted(row[0] for row in await cursor.fetchall())
    return report


def _embedding_text(chunk: Chunk) -> str:
    return f"{chunk.heading}\n\n{chunk.text}" if chunk.heading else chunk.text


async def _run(directory: Path, settings: Settings, prune: bool) -> IngestReport:
    documents = load_documents(directory)
    embedder = build_embedder(settings)
    try:
        return await ingest_documents(documents, settings, embedder, prune=prune)
    finally:
        await embedder.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m chatbot.rag.ingest", description="Load Markdown documents for search."
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--prune", action="store_true", help="delete stored documents missing from DIRECTORY"
    )
    args = parser.parse_args(argv)
    try:
        settings = Settings()
    except ValidationError as exc:
        print(f"Invalid settings: {exc}", file=sys.stderr)
        return 2
    configure_logging(settings)
    try:
        report = asyncio.run(_run(args.directory, settings, args.prune))
    except ChatbotError as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1
    except DatabaseError as exc:
        print(f"Ingestion failed: database error: {exc}", file=sys.stderr)
        return 1
    print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
