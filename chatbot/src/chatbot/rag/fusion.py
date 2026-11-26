"""Reciprocal rank fusion: merge ranked lists whose scores are not comparable.

Each list contributes ``1 / (k + rank)`` for every chunk it returns, with ranks
starting at 1. Only positions matter, so a full-text rank and a cosine
similarity can be combined without calibrating one against the other. A chunk
found by both legs beats one found by a single leg at a similar position. The
usual ``k = 60`` keeps a single first place from dominating a list.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from chatbot.rag.base import Candidate

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedHit:
    candidate: Candidate
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # leg name -> 1-based rank


def reciprocal_rank_fusion(
    results: Mapping[str, Sequence[Candidate]], *, k: int = DEFAULT_RRF_K
) -> list[FusedHit]:
    """Fused hits, best first. Ties go to the better best-rank, then the lower chunk id."""
    if k < 1:
        raise ValueError("k must be at least 1")
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    chunks: dict[int, Candidate] = {}
    for leg, candidates in results.items():
        seen: set[int] = set()
        for rank, item in enumerate(candidates, start=1):
            if item.chunk_id in seen:
                continue  # a leg counts each chunk once, at its best rank
            seen.add(item.chunk_id)
            chunks.setdefault(item.chunk_id, item)
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(item.chunk_id, {})[leg] = rank
    order = sorted(scores, key=lambda cid: (-scores[cid], min(ranks[cid].values()), cid))
    return [FusedHit(chunks[cid], scores[cid], ranks[cid]) for cid in order]
