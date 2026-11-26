"""Local cross-encoder reranker: a small ONNX model run with fastembed.

The default ``Xenova/ms-marco-MiniLM-L-6-v2`` (about 80 MB) was chosen in C20 on
the labeled set: it raised hybrid MRR@10 from 0.915 to 0.931 at about 250 ms per
query for 20 candidates on a laptop CPU. The 12-layer variant scored 0.938 at
twice the latency.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import anyio

from chatbot.errors import ProviderError
from chatbot.rag.fusion import FusedHit
from chatbot.rag.rerank.base import RankedHit, order_by_scores, passage

if TYPE_CHECKING:
    from fastembed.rerank.cross_encoder import TextCrossEncoder


class LocalReranker:
    def __init__(self, model: str, *, cache_dir: str) -> None:
        self.model = model
        self.name = f"local:{model}"
        self._cache_dir = cache_dir
        self._encoder: TextCrossEncoder | None = None
        self._lock = anyio.Lock()

    async def _load(self) -> "TextCrossEncoder":
        async with self._lock:
            if self._encoder is None:
                try:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder
                except ImportError as exc:
                    raise ProviderError(
                        "The local reranker needs the extra: install chatbot[local-embeddings]"
                    ) from exc
                self._encoder = await anyio.to_thread.run_sync(
                    lambda: TextCrossEncoder(model_name=self.model, cache_dir=self._cache_dir)
                )
            return self._encoder

    async def rerank(self, query: str, hits: Sequence[FusedHit]) -> list[RankedHit]:
        if not hits:
            return []
        encoder = await self._load()
        texts = [passage(hit) for hit in hits]
        scores = await anyio.to_thread.run_sync(lambda: list(encoder.rerank(query, texts)))
        return order_by_scores(hits, scores)

    async def aclose(self) -> None:
        self._encoder = None
