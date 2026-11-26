"""Hybrid retriever: run the lexical and vector legs, then fuse their rankings.

Both legs run concurrently on the read-only database role, each with the same
filters, so access control is enforced inside both SQL queries. Each leg returns
up to ``RETRIEVAL_CANDIDATES`` chunks; reciprocal rank fusion merges them. The
reranker and pagination for the model come in the search_docs tool.
"""

from typing import Self

import anyio
from psycopg import Error
from psycopg.conninfo import make_conninfo
from psycopg_pool import AsyncConnectionPool

from chatbot.config import Settings
from chatbot.errors import RetrievalError
from chatbot.rag.base import Candidate, SearchFilters, SearchLeg
from chatbot.rag.embeddings import Embedder
from chatbot.rag.fusion import DEFAULT_RRF_K, FusedHit, reciprocal_rank_fusion
from chatbot.rag.lexical import LexicalSearch, PostgresFullTextSearch
from chatbot.rag.vector import VectorSearch

MAX_QUERY_CHARS = 1000


class HybridRetriever:
    """Own the read-only pool with ``async with``; ``search`` is safe to call concurrently."""

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        *,
        lexical: LexicalSearch | None = None,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self.lexical: SearchLeg = lexical or PostgresFullTextSearch()
        self.vector: SearchLeg = VectorSearch(embedder)
        self.candidates = settings.retrieval_candidates
        self.rrf_k = rrf_k
        self.pool = AsyncConnectionPool(
            conninfo=make_conninfo(
                host=settings.postgres_host,
                port=settings.postgres_port,
                dbname=settings.postgres_db,
                user=settings.ro_db_user,
                password=settings.ro_db_password.get_secret_value(),
            ),
            min_size=1,
            max_size=5,
            open=False,
        )

    async def __aenter__(self) -> Self:
        await self.pool.open(wait=True)
        return self

    async def __aexit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        await self.pool.close()

    async def search_leg(
        self, leg: SearchLeg, query: str, filters: SearchFilters
    ) -> list[Candidate]:
        """One leg on its own, for evaluation and debugging."""
        query = _clean(query)
        if not query:
            return []
        try:
            return await leg.search(self.pool, query, filters, self.candidates)
        except Error as exc:
            raise RetrievalError("Document search failed") from exc

    async def search(self, query: str, filters: SearchFilters) -> list[FusedHit]:
        query = _clean(query)
        if not query or not filters.groups:
            return []  # fail closed: no groups means nothing is visible
        results: dict[str, list[Candidate]] = {}

        async def run(leg: SearchLeg) -> None:
            results[leg.name] = await self.search_leg(leg, query, filters)

        async with anyio.create_task_group() as group:
            group.start_soon(run, self.lexical)
            group.start_soon(run, self.vector)
        # Fixed leg order keeps fusion ties deterministic.
        ordered = {leg.name: results[leg.name] for leg in (self.lexical, self.vector)}
        return reciprocal_rank_fusion(ordered, k=self.rrf_k)


def _clean(query: str) -> str:
    query = " ".join(query.split())
    if len(query) > MAX_QUERY_CHARS:
        raise RetrievalError(f"Search query is longer than {MAX_QUERY_CHARS} characters")
    return query
