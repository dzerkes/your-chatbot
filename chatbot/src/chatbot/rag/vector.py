"""Semantic search leg: cosine distance between the query and chunk embeddings.

The access filter sits in the WHERE clause. An HNSW scan returns its nearest
``ef_search`` rows before the filter runs, so a selective filter could leave
fewer than ``limit`` results. pgvector 0.8's iterative scan keeps searching the
index until enough rows pass the filter, in exact distance order.
"""

from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from chatbot.rag.base import CANDIDATE_COLUMNS, Candidate, SearchFilters, candidate
from chatbot.rag.embeddings import Embedder, vector_literal


class VectorSearch:
    name = "vector"

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    async def search(
        self, pool: AsyncConnectionPool, query: str, filters: SearchFilters, limit: int
    ) -> list[Candidate]:
        if not filters.groups or not query.strip():
            return []
        # Embed before taking a connection, so a slow embedding API holds no pool slot.
        embedding = vector_literal(await self.embedder.embed_query(query))
        where, params = filters.where()
        async with pool.connection() as conn, conn.transaction():
            await conn.execute("SET TRANSACTION READ ONLY")
            await conn.execute("SET LOCAL hnsw.iterative_scan = strict_order")
            await conn.execute(
                sql.SQL("SET LOCAL hnsw.ef_search = {}").format(sql.Literal(max(40, limit)))
            )
            cursor = await conn.execute(
                f"""
                SELECT {CANDIDATE_COLUMNS},
                       1 - (c.embedding <=> %s::vector) AS score
                FROM rag.chunks AS c
                JOIN rag.documents AS d ON d.id = c.document_id
                WHERE {where}
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                [embedding, *params, embedding, limit],
            )
            return [candidate(row) for row in await cursor.fetchall()]
