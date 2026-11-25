"""API reranker for Cohere-compatible ``/rerank`` endpoints.

Cohere, Jina, Voyage and several self-hosted servers accept the same shape:
``{"model", "query", "documents", "top_n"}`` in, ``{"results": [{"index",
"relevance_score"}]}`` out. Set ``RERANKER_API_URL`` to the full endpoint and
``RERANKER_MODEL`` to the provider's model name. Candidate text leaves the
machine, so only use a provider approved for the documents' classification.
"""

import logging
from collections.abc import Sequence

import httpx

from chatbot.errors import ProviderError
from chatbot.rag.fusion import FusedHit
from chatbot.rag.rerank.base import RankedHit, order_by_scores, passage

logger = logging.getLogger(__name__)


class ApiReranker:
    def __init__(self, model: str, *, url: str, client: httpx.AsyncClient) -> None:
        self.model = model
        self.name = f"api:{model}"
        self._url = url
        self._client = client

    async def rerank(self, query: str, hits: Sequence[FusedHit]) -> list[RankedHit]:
        if not hits:
            return []
        payload = {
            "model": self.model,
            "query": query,
            "documents": [passage(hit) for hit in hits],
            "top_n": len(hits),
        }
        try:
            response = await self._client.post(self._url, json=payload)
            response.raise_for_status()
            results = response.json()["results"]
            scores = [0.0] * len(hits)
            seen: set[int] = set()
            for item in results:
                index = int(item["index"])
                if not 0 <= index < len(hits) or index in seen:
                    raise ValueError(f"Invalid result index {index}")
                seen.add(index)
                scores[index] = float(item["relevance_score"])
            if len(seen) != len(hits):
                raise ValueError("Reranker did not score every document")
        # Users get a generic message; operators need the provider's reason in the logs.
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Rerank request failed: status=%s %s",
                exc.response.status_code,
                exc.response.text[:500],
            )
            status = exc.response.status_code
            raise ProviderError(
                "Rerank request failed", retryable=status == 429 or status >= 500
            ) from exc
        except httpx.HTTPError as exc:
            logger.warning("Rerank connection failed: %s", exc)
            raise ProviderError("Rerank connection failed", retryable=True) from exc
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Rerank response was invalid: %s", exc)
            raise ProviderError("Rerank response was invalid") from exc
        return order_by_scores(hits, scores)

    async def aclose(self) -> None:
        await self._client.aclose()
