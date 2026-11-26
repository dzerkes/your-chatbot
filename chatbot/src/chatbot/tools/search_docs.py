"""search_docs: hybrid document search, reranked, paginated and access-controlled.

The caller's groups come from ``ToolContext`` (the authenticated user), never
from the model's arguments. Each page reruns the search, which is deterministic
for the same query, filters and index, so a cursor only needs an offset. The
cursor is bound to the user, query and filters.
"""

import json
import logging
import re
from typing import Self

from pydantic import Field, JsonValue, field_validator

from chatbot.config import Settings
from chatbot.errors import ChatbotError, ToolError
from chatbot.rag.base import SearchFilters
from chatbot.rag.embeddings import Embedder, build_embedder
from chatbot.rag.rerank import RankedHit, Reranker, build_reranker
from chatbot.rag.retriever import MAX_QUERY_CHARS, HybridRetriever
from chatbot.tools.base import Tool, ToolContext, ToolInput
from chatbot.tools.pagination import ToolPage, build_page, decode_cursor
from chatbot.types import DomainModel

logger = logging.getLogger(__name__)
SNIPPET_CHARS = 800
MAX_PAGE = 10
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


class SearchDocsInput(ToolInput):
    query: str = Field(
        min_length=1,
        max_length=min(MAX_QUERY_CHARS, 500),
        description="What to look for, in natural language or key terms",
    )
    source_prefix: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_./-]*$",
        description=(
            "Only documents under this folder, e.g. 'handbook/'. Leave it out unless the "
            "user names a folder; a wrong folder returns nothing."
        ),
    )
    # Flat fields: small models often misplace a nested pagination object.
    limit: int | None = Field(
        default=None, ge=1, le=MAX_PAGE, description="Results per page (default 5)"
    )
    cursor: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
        description="next_cursor from the previous page, with the same query and source_prefix",
    )

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Query cannot be blank")
        return value


class DocumentHit(DomainModel):
    chunk_id: int
    title: str
    section: str  # heading path inside the document; "" for its opening text
    source: str
    snippet: str
    score: float


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """The chunk text, cut at a sentence or word boundary when it is too long."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = max((m.start() for m in _SENTENCE_END.finditer(cut)), default=-1)
    if boundary < limit // 2:
        boundary = cut.rfind(" ")
    return (cut[: boundary + 1] if boundary > 0 else cut).rstrip() + " …"


def to_hit(ranked: RankedHit) -> DocumentHit:
    candidate = ranked.hit.candidate
    prefix = f"{candidate.title} > "
    section = candidate.heading[len(prefix) :] if candidate.heading.startswith(prefix) else ""
    return DocumentHit(
        chunk_id=candidate.chunk_id,
        title=candidate.title,
        section=section,
        source=candidate.source,
        snippet=snippet(candidate.text),
        score=round(ranked.score, 4),
    )


class SearchDocs:
    """Own the retriever, embedder and reranker with ``async with``; register ``tool()``."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.embedder = embedder or build_embedder(settings)
        self.reranker = reranker or build_reranker(settings)
        self.retriever = HybridRetriever(settings, self.embedder)
        self.rerank_candidates = settings.rerank_candidates
        self.page_size = settings.retrieval_page_size

    async def __aenter__(self) -> Self:
        await self.retriever.__aenter__()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            await self.retriever.__aexit__(exc_type, exc, tb)
        finally:
            await self.reranker.aclose()
            await self.embedder.aclose()

    def tool(self) -> Tool[SearchDocsInput]:
        return Tool(
            "search_docs",
            "Search company documents (policies, handbooks, runbooks, project briefs) and "
            "return the most relevant passages, best first, each with its document title, "
            "section and source path for citation. Only documents the user may read are "
            "searched. Use next_cursor with the same query and filters for more results.",
            SearchDocsInput,
            self.search,
        )

    async def search(self, inputs: SearchDocsInput, context: ToolContext) -> ToolPage:
        # Tag filters exist in the retriever but are not offered to the model: it cannot
        # know the tag vocabulary, and guessed tags silently returned nothing.
        filters = SearchFilters(
            groups=tuple(context.user.groups), source_prefix=inputs.source_prefix
        )
        scope = json.dumps(
            {
                "tool": "search_docs",
                "user": str(context.user.id),
                "query": inputs.query,
                "source_prefix": inputs.source_prefix,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        offset = decode_cursor(inputs.cursor, scope=scope)
        limit = inputs.limit or self.page_size
        try:
            fused = await self.retriever.search(inputs.query, filters)
            ranked = await self.reranker.rerank(inputs.query, fused[: self.rerank_candidates])
        except ChatbotError as exc:
            logger.warning("Document search failed: %s", exc)
            raise ToolError("Document search failed; try again or rephrase the query") from exc
        items: list[JsonValue] = [to_hit(hit).model_dump(mode="json") for hit in ranked]
        return build_page(
            items[offset : offset + limit],
            total=len(items),
            offset=offset,
            limit=limit,
            scope=scope,
        )
