"""Fusion, filters and retriever guards; no database."""

import json

import pytest
from chatbot.config import Settings
from chatbot.errors import RetrievalError
from chatbot.rag.base import Candidate, SearchFilters
from chatbot.rag.embeddings import FakeEmbedder
from chatbot.rag.fusion import reciprocal_rank_fusion
from chatbot.rag.retriever import MAX_QUERY_CHARS, HybridRetriever

pytestmark = pytest.mark.unit


def hit(chunk_id: int, score: float = 0.0) -> Candidate:
    return Candidate(chunk_id, 1, "s.md", "T", "T", f"text {chunk_id}", score)


def test_rrf_rewards_agreement_and_records_ranks() -> None:
    fused = reciprocal_rank_fusion(
        {"lexical": [hit(1), hit(2), hit(3)], "vector": [hit(3), hit(4), hit(1)]}, k=60
    )
    assert [h.candidate.chunk_id for h in fused] == [1, 3, 2, 4]
    first = fused[0]
    assert first.ranks == {"lexical": 1, "vector": 3}
    assert first.score == pytest.approx(1 / 61 + 1 / 63)
    assert fused[2].ranks == {"lexical": 2}


def test_rrf_breaks_ties_by_best_rank_then_chunk_id_and_ignores_repeats() -> None:
    fused = reciprocal_rank_fusion({"a": [hit(5), hit(6)], "b": [hit(6), hit(5)]})
    assert [h.candidate.chunk_id for h in fused] == [5, 6]
    repeated = reciprocal_rank_fusion({"a": [hit(7), hit(7), hit(8)]})
    assert [(h.candidate.chunk_id, h.ranks) for h in repeated] == [(7, {"a": 1}), (8, {"a": 3})]
    assert reciprocal_rank_fusion({"a": [], "b": []}) == []
    with pytest.raises(ValueError, match="k must be"):
        reciprocal_rank_fusion({}, k=0)


def test_filters_always_include_the_group_check() -> None:
    where, params = SearchFilters(groups=("hr", "employees", "hr")).where()
    assert where == "c.allowed_groups && %s::text[]"
    assert params == [["employees", "hr"]]


def test_filters_escape_prefix_and_match_all_tags() -> None:
    where, params = SearchFilters(
        groups=("a",), source_prefix="eng_100%/", tags=("ops", "engineering")
    ).where()
    assert where == (
        "c.allowed_groups && %s::text[] AND d.source LIKE %s AND d.metadata -> 'tags' @> %s::jsonb"
    )
    assert params[1] == r"eng\_100\%/%"
    assert json.loads(str(params[2])) == ["engineering", "ops"]


async def test_retriever_fails_closed_without_groups_or_query() -> None:
    retriever = HybridRetriever(Settings(), FakeEmbedder(8))  # pool is never opened
    assert await retriever.search("annual leave", SearchFilters(groups=())) == []
    assert await retriever.search("   ", SearchFilters(groups=("employees",))) == []
    with pytest.raises(RetrievalError, match="longer than"):
        await retriever.search("x" * (MAX_QUERY_CHARS + 1), SearchFilters(groups=("a",)))
