"""Reranker strategies selected by ``RERANKER``: none, local or api."""

import httpx

from chatbot.config import Settings
from chatbot.errors import ProviderError
from chatbot.rag.rerank.api import ApiReranker
from chatbot.rag.rerank.base import NoReranker, RankedHit, Reranker
from chatbot.rag.rerank.local import LocalReranker

__all__ = ["ApiReranker", "LocalReranker", "NoReranker", "RankedHit", "Reranker", "build_reranker"]


def build_reranker(settings: Settings) -> Reranker:
    match settings.reranker:
        case "none":
            return NoReranker()
        case "local":
            return LocalReranker(settings.reranker_model, cache_dir=settings.embedding_cache_dir)
        case "api":
            key = settings.reranker_api_key.get_secret_value()
            if settings.reranker_api_url is None or not key:
                raise ProviderError("RERANKER=api needs RERANKER_API_URL and RERANKER_API_KEY")
            client = httpx.AsyncClient(
                headers={"Authorization": f"Bearer {key}"},
                timeout=settings.llm_timeout_seconds,
            )
            return ApiReranker(
                settings.reranker_model, url=str(settings.reranker_api_url), client=client
            )
