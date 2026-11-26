"""Embedder strategy: turn text into vectors for the semantic search leg.

``EMBEDDING_MODEL`` uses the same ``provider:model`` notation as the LLMs:

- ``local:<model>``: a small ONNX model run on this machine with fastembed
  (install ``chatbot[local-embeddings]``). No document text leaves the machine.
- ``openai:<model>``: the OpenAI embeddings API, shortened to ``EMBEDDING_DIM``.
- ``fake:<name>``: deterministic hashed bag-of-words vectors for tests.

Documents and queries are embedded separately because some models (bge, e5)
expect a different prefix for each.
"""

import hashlib
import logging
import math
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import anyio
from openai import APIConnectionError, APIError, APIStatusError, AsyncOpenAI

from chatbot.config import Settings
from chatbot.errors import ProviderError

if TYPE_CHECKING:
    from fastembed import TextEmbedding

logger = logging.getLogger(__name__)
_WORD = re.compile(r"\w+", re.UNICODE)


class Embedder(Protocol):
    name: str  # provider:model, stored with each document so a model change re-embeds
    dim: int

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...

    async def aclose(self) -> None: ...


def build_embedder(settings: Settings) -> Embedder:
    provider, _, model = settings.embedding_model.partition(":")
    dim = settings.embedding_dim
    match provider:
        case "local":
            return LocalEmbedder(model, dim, cache_dir=settings.embedding_cache_dir)
        case "openai":
            key = settings.openai_api_key.get_secret_value()
            if not key:
                raise ProviderError("OpenAI API key is not configured")
            return OpenAIEmbedder(model, dim, client=AsyncOpenAI(api_key=key, max_retries=2))
        case "fake":
            return FakeEmbedder(dim, model)
        case _:
            raise ProviderError(f"Unknown embedding provider: {provider}")


def vector_literal(vector: Sequence[float]) -> str:
    """pgvector's text form, for a ``%s::vector`` parameter."""
    return "[" + ",".join(format(x, ".8g") for x in vector) + "]"


def _checked(vectors: Sequence[Sequence[float]], count: int, dim: int) -> list[list[float]]:
    result = [[float(x) for x in vector] for vector in vectors]
    if len(result) != count or any(len(vector) != dim for vector in result):
        raise ProviderError(
            f"Embedder returned {len(result)} vectors for {count} texts; expected {dim} dimensions"
        )
    return result


class FakeEmbedder:
    """Hash each word into a signed bucket. Texts sharing words get similar vectors."""

    def __init__(self, dim: int, model: str = "hash") -> None:
        self.dim = dim
        self.name = f"fake:{model}"

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in _WORD.findall(text.lower()):
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector] if norm else [1.0 / math.sqrt(self.dim)] * self.dim

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    async def aclose(self) -> None:
        return None


class LocalEmbedder:
    """fastembed ONNX model, loaded on first use and run in a worker thread."""

    def __init__(self, model: str, dim: int, *, cache_dir: str) -> None:
        self.model = model
        self.dim = dim
        self.name = f"local:{model}"
        self._cache_dir = cache_dir
        self._encoder: TextEmbedding | None = None
        self._lock = anyio.Lock()

    async def _load(self) -> "TextEmbedding":
        async with self._lock:
            if self._encoder is None:
                try:
                    from fastembed import TextEmbedding
                except ImportError as exc:
                    raise ProviderError(
                        "Local embeddings need the extra: install chatbot[local-embeddings]"
                    ) from exc
                self._encoder = await anyio.to_thread.run_sync(
                    lambda: TextEmbedding(model_name=self.model, cache_dir=self._cache_dir)
                )
            return self._encoder

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        encoder = await self._load()
        vectors = await anyio.to_thread.run_sync(lambda: list(encoder.passage_embed(list(texts))))
        return _checked([v.tolist() for v in vectors], len(texts), self.dim)

    async def embed_query(self, text: str) -> list[float]:
        encoder = await self._load()
        vectors = await anyio.to_thread.run_sync(lambda: list(encoder.query_embed(text)))
        return _checked([v.tolist() for v in vectors], 1, self.dim)[0]

    async def aclose(self) -> None:
        self._encoder = None


class OpenAIEmbedder:
    """OpenAI embeddings, requested at ``dim`` dimensions so they fit the column."""

    batch_size = 128

    def __init__(self, model: str, dim: int, *, client: AsyncOpenAI) -> None:
        self.model = model
        self.dim = dim
        self.name = f"openai:{model}"
        self._client = client

    async def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            try:
                response = await self._client.embeddings.create(
                    model=self.model, input=batch, dimensions=self.dim
                )
            # Users get a generic message; operators need the provider's reason in the logs.
            except APIConnectionError as exc:
                logger.warning("OpenAI embedding connection failed: %s", exc)
                raise ProviderError("OpenAI embedding connection failed", retryable=True) from exc
            except APIStatusError as exc:
                logger.warning(
                    "OpenAI embedding failed (request_id=%s): %s", exc.request_id, exc.message
                )
                retryable = exc.status_code == 429 or exc.status_code >= 500
                raise ProviderError("OpenAI embedding failed", retryable=retryable) from exc
            except APIError as exc:
                logger.warning("OpenAI embedding failed: %s", exc.message)
                raise ProviderError("OpenAI embedding failed") from exc
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(_checked([item.embedding for item in ordered], len(batch), self.dim))
        return vectors

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts) if texts else []

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text]))[0]

    async def aclose(self) -> None:
        await self._client.close()
