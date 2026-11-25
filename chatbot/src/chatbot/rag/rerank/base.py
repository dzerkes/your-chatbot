"""Reranker strategy: rescore the fused candidates against the query.

Fusion only knows positions. A reranker reads the query and each candidate
together and orders them by how well the text answers the question. It runs on
the top ``RERANK_CANDIDATES`` fused hits only, because cross-encoders cost one
model pass per candidate.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from chatbot.rag.fusion import FusedHit


@dataclass(frozen=True)
class RankedHit:
    hit: FusedHit
    score: float  # the reranker's score, or the fusion score for "none"
    fused_rank: int  # 1-based position before reranking, kept for debugging


class Reranker(Protocol):
    name: str

    async def rerank(self, query: str, hits: Sequence[FusedHit]) -> list[RankedHit]:
        """The same hits, best first."""
        ...

    async def aclose(self) -> None: ...


def passage(hit: FusedHit) -> str:
    """The text a reranker scores: heading path, then the chunk."""
    candidate = hit.candidate
    return f"{candidate.heading}\n\n{candidate.text}" if candidate.heading else candidate.text


def order_by_scores(hits: Sequence[FusedHit], scores: Sequence[float]) -> list[RankedHit]:
    """Sort by score, keeping the fused order for ties."""
    if len(scores) != len(hits):
        raise ValueError("One score is needed per hit")
    ranked = [
        RankedHit(hit, float(score), rank)
        for rank, (hit, score) in enumerate(zip(hits, scores, strict=True), start=1)
    ]
    return sorted(ranked, key=lambda item: (-item.score, item.fused_rank))


class NoReranker:
    """Baseline: keep the fused order."""

    name = "none"

    async def rerank(self, query: str, hits: Sequence[FusedHit]) -> list[RankedHit]:
        return [RankedHit(hit, hit.score, rank) for rank, hit in enumerate(hits, start=1)]

    async def aclose(self) -> None:
        return None
