"""Rerankers and the search_docs tool without a database or model downloads."""

import json
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import httpx
import pytest
from chatbot.config import Settings
from chatbot.errors import ProviderError, RetrievalError, ToolError
from chatbot.rag.base import Candidate, SearchFilters
from chatbot.rag.embeddings import FakeEmbedder
from chatbot.rag.fusion import FusedHit
from chatbot.rag.rerank import (
    ApiReranker,
    LocalReranker,
    NoReranker,
    RankedHit,
    build_reranker,
)
from chatbot.rag.rerank.base import order_by_scores
from chatbot.tools.base import ToolContext
from chatbot.tools.search_docs import SearchDocs, SearchDocsInput, snippet, to_hit
from chatbot.types import User
from pydantic import HttpUrl, SecretStr, ValidationError

pytestmark = pytest.mark.unit


def fused(chunk_id: int, heading: str = "Doc > Section", text: str = "Text.") -> FusedHit:
    candidate = Candidate(chunk_id, 1, "dir/doc.md", "Doc", heading, text, 0.0)
    return FusedHit(candidate, 1.0 / (60 + chunk_id), {"lexical": chunk_id})


# --- rerankers --------------------------------------------------------------


async def test_no_reranker_keeps_fused_order() -> None:
    hits = [fused(1), fused(2)]
    ranked = await NoReranker().rerank("q", hits)
    assert [(r.hit.candidate.chunk_id, r.fused_rank) for r in ranked] == [(1, 1), (2, 2)]
    assert ranked[0].score == hits[0].score


def test_scores_reorder_and_ties_keep_fused_order() -> None:
    ranked = order_by_scores([fused(1), fused(2), fused(3)], [0.2, 0.9, 0.2])
    assert [(r.hit.candidate.chunk_id, r.fused_rank) for r in ranked] == [(2, 2), (1, 1), (3, 3)]
    with pytest.raises(ValueError, match="One score"):
        order_by_scores([fused(1)], [])


def test_build_reranker_selects_strategy() -> None:
    assert build_reranker(Settings(reranker="none")).name == "none"
    local = build_reranker(Settings())
    assert isinstance(local, LocalReranker)
    assert local.name == "local:Xenova/ms-marco-MiniLM-L-6-v2"
    with pytest.raises(ProviderError, match="RERANKER_API_URL"):
        build_reranker(Settings(reranker="api"))
    api = build_reranker(
        Settings(
            reranker="api",
            reranker_model="rerank-v3.5",
            reranker_api_url=HttpUrl("https://rerank.example/v2/rerank"),
            reranker_api_key=SecretStr("k"),
        )
    )
    assert isinstance(api, ApiReranker) and api.name == "api:rerank-v3.5"


def api_reranker(handler: Any) -> ApiReranker:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ApiReranker("model-x", url="https://rerank.example/rerank", client=client)


async def test_api_reranker_sends_documents_and_orders_by_score() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                ]
            },
        )

    reranker = api_reranker(handler)
    ranked = await reranker.rerank("leave", [fused(1, text="A"), fused(2, text="B")])
    assert [r.hit.candidate.chunk_id for r in ranked] == [2, 1]
    assert requests == [
        {
            "model": "model-x",
            "query": "leave",
            "documents": ["Doc > Section\n\nA", "Doc > Section\n\nB"],
            "top_n": 2,
        }
    ]
    assert await reranker.rerank("leave", []) == []
    await reranker.aclose()


@pytest.mark.parametrize(
    ("response", "message", "retryable"),
    [
        (httpx.Response(429, text="slow down"), "Rerank request failed", True),
        (httpx.Response(400, text="bad"), "Rerank request failed", False),
        (
            httpx.Response(200, json={"results": [{"index": 5, "relevance_score": 1}]}),
            "invalid",
            False,
        ),
        (httpx.Response(200, json={"results": []}), "invalid", False),
        (httpx.Response(200, json={"oops": 1}), "invalid", False),
    ],
)
async def test_api_reranker_failures_are_provider_errors(
    response: httpx.Response, message: str, retryable: bool
) -> None:
    reranker = api_reranker(lambda _request: response)
    with pytest.raises(ProviderError, match=message) as caught:
        await reranker.rerank("q", [fused(1)])
    assert caught.value.retryable is retryable


# --- tool -------------------------------------------------------------------


def test_snippet_keeps_short_text_and_cuts_long_text_at_a_boundary() -> None:
    assert snippet("Short.") == "Short."
    text = "First sentence is here. " * 10
    cut = snippet(text, limit=100)
    assert cut.endswith(" …") and len(cut) <= 102
    assert cut.removesuffix(" …").endswith(".")
    assert snippet("x" * 300, limit=100) == "x" * 100 + " …"


def test_hit_section_drops_the_document_title() -> None:
    def ranked(heading: str) -> RankedHit:
        return RankedHit(fused(1, heading=heading), 0.123456, 1)

    assert to_hit(ranked("Doc > Leave > Sick")).section == "Leave > Sick"
    assert to_hit(ranked("Doc")).section == ""
    hit = to_hit(ranked("Doc > A"))
    assert (hit.title, hit.source, hit.score, hit.chunk_id) == ("Doc", "dir/doc.md", 0.1235, 1)


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "   "},
        {"query": "x" * 501},
        {"query": "q", "source_prefix": "../etc"},
        {"query": "q", "groups": ["finance"]},  # the model cannot widen access
        {"query": "q", "pagination": {"limit": 11}},
        {"query": "q", "tags": ["t"] * 6},
    ],
)
def test_invalid_arguments_are_rejected(arguments: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SearchDocsInput.model_validate(arguments)


class FakeRetriever:
    def __init__(self, hits: Sequence[FusedHit], error: Exception | None = None) -> None:
        self.hits = list(hits)
        self.error = error
        self.filters: list[SearchFilters] = []

    async def search(self, query: str, filters: SearchFilters) -> list[FusedHit]:
        self.filters.append(filters)
        if self.error:
            raise self.error
        return self.hits


class RecordingReranker(NoReranker):
    def __init__(self) -> None:
        self.sizes: list[int] = []

    async def rerank(self, query: str, hits: Sequence[FusedHit]) -> list[RankedHit]:
        self.sizes.append(len(hits))
        return await super().rerank(query, hits)


def tool(
    hits: Sequence[FusedHit], **settings: Any
) -> tuple[SearchDocs, FakeRetriever, RecordingReranker]:
    reranker = RecordingReranker()
    docs = SearchDocs(Settings(**settings), embedder=FakeEmbedder(8), reranker=reranker)
    retriever = FakeRetriever(hits)
    docs.retriever = retriever  # type: ignore[assignment]
    return docs, retriever, reranker


def context(*groups: str) -> ToolContext:
    user = User(external_id="u", display_name="U", groups=list(groups))
    return ToolContext(user=user, session_id=uuid4(), turn_id=uuid4())


async def test_search_uses_the_callers_groups_and_reranks_the_top_candidates() -> None:
    docs, retriever, reranker = tool([fused(i) for i in range(1, 31)], rerank_candidates=20)
    inputs = SearchDocsInput.model_validate(
        {"query": " leave  policy ", "source_prefix": "handbook/", "tags": ["Policy", "policy"]}
    )
    page = await docs.search(inputs, context("employees", "hr"))
    assert retriever.filters == [
        SearchFilters(groups=("employees", "hr"), source_prefix="handbook/", tags=("policy",))
    ]
    assert reranker.sizes == [20]
    assert page.total == 20 and len(page.items) == 5 and page.next_cursor
    first = page.items[0]
    assert isinstance(first, dict)
    assert set(first) == {"chunk_id", "title", "section", "source", "snippet", "score"}


async def test_cursor_pages_forward_and_is_bound_to_the_user_and_query() -> None:
    docs, _, _ = tool([fused(i) for i in range(1, 8)])
    alice = context("employees")
    first = await docs.search(SearchDocsInput(query="leave"), alice)
    second = await docs.search(
        SearchDocsInput.model_validate(
            {"query": "leave", "pagination": {"cursor": first.next_cursor}}
        ),
        alice,
    )
    ids = [item["chunk_id"] for item in [*first.items, *second.items]]  # type: ignore[index, call-overload]
    assert ids == list(range(1, 8)) and second.next_cursor is None
    for other, query in ((context("employees"), "leave"), (alice, "other query")):
        with pytest.raises(ToolError, match="cursor"):
            await docs.search(
                SearchDocsInput.model_validate(
                    {"query": query, "pagination": {"cursor": first.next_cursor}}
                ),
                other,
            )


async def test_empty_results_and_failures() -> None:
    docs, _, _ = tool([])
    page = await docs.search(SearchDocsInput(query="leave"), context())
    assert (page.items, page.total, page.next_cursor) == ([], 0, None)
    docs.retriever = FakeRetriever([], RetrievalError("db down"))  # type: ignore[assignment]
    with pytest.raises(ToolError, match="Document search failed"):
        await docs.search(SearchDocsInput(query="leave"), context("employees"))
