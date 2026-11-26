"""Lexical search leg: Postgres full-text search, the "BM25 leg".

Postgres ranking is not true BM25, so this sits behind ``LexicalSearch`` and can
be replaced (for example by a BM25 extension) if evals show it matters.

``websearch_to_tsquery`` and ``plainto_tsquery`` require every word to match,
which suits a search box but not a question such as "how many vacation days do
I get": most chunks miss at least one word. This leg instead takes the query's
stemmed English terms, without stop words, and matches any of them. The rank
then rewards chunks that contain more of the terms and that contain them in the
heading (weight A) rather than the body (weight B).

The rank is not normalized by length. Chunks are already size-limited, and on
the labeled set in tests/fixtures length normalization lowered this leg's MRR
(0.740 against 0.795) by favouring short chunks with little context.
"""

from typing import Protocol

from psycopg_pool import AsyncConnectionPool

from chatbot.rag.base import CANDIDATE_COLUMNS, Candidate, SearchFilters, SearchLeg, candidate


class LexicalSearch(SearchLeg, Protocol):
    """Marker for full-text implementations, so the retriever can name its legs."""


class PostgresFullTextSearch:
    name = "lexical"

    async def search(
        self, pool: AsyncConnectionPool, query: str, filters: SearchFilters, limit: int
    ) -> list[Candidate]:
        if not filters.groups or not query.strip():
            return []
        where, params = filters.where()
        async with pool.connection() as conn, conn.transaction():
            await conn.execute("SET TRANSACTION READ ONLY")
            cursor = await conn.execute(
                f"""
                WITH q AS (
                    -- Lexemes are already stemmed, so the 'simple' config keeps them as is.
                    SELECT to_tsquery('simple', string_agg(quote_literal(lexeme), ' | ')) AS terms
                    FROM unnest(tsvector_to_array(to_tsvector('english', %s))) AS lexeme
                )
                SELECT {CANDIDATE_COLUMNS},
                       ts_rank(c.search_vector, q.terms) AS score
                FROM rag.chunks AS c
                JOIN rag.documents AS d ON d.id = c.document_id
                CROSS JOIN q
                WHERE q.terms IS NOT NULL AND c.search_vector @@ q.terms AND {where}
                ORDER BY score DESC, c.id
                LIMIT %s
                """,
                [query, *params, limit],
            )
            return [candidate(row) for row in await cursor.fetchall()]
