"""LLM gateway: retries, fallbacks and cost, behind the ``LLMProvider`` interface.

A decorator over providers, so the agent and tracing see one provider:

- Retries: a retryable ``ProviderError`` (rate limit, 5xx, connection) is
  retried on the same model with exponential backoff and full jitter, up to
  ``LLM_MAX_RETRIES`` times.
- Fallbacks: when a model still fails, or fails in a way retrying cannot fix,
  the next model in ``LLM_FALLBACKS`` gets the same request.
- Never replay an attempt once it has delivered text, a tool call, or its
  final usage or stop event. Callers consume exactly one response stream.
- Cost: each usage event is tagged with the model that served it and its cost
  from the price table (``None`` for a model without a price).

Adapters run with their SDK retries off, so this is the only retry layer.
"""

import asyncio
import logging
import random
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack

from chatbot.errors import ProviderError
from chatbot.llm.base import (
    LLMEvent,
    LLMProvider,
    LLMRequest,
    LLMTextDelta,
    LLMToolCall,
    LLMUsage,
    provider_stream,
)
from chatbot.llm.pricing import PriceTable

logger = logging.getLogger(__name__)


class LLMGateway:
    def __init__(
        self,
        primary: LLMProvider,
        fallbacks: Sequence[LLMProvider] = (),
        *,
        prices: PriceTable,
        max_retries: int = 2,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[float], float] = lambda limit: random.uniform(0, limit),
    ) -> None:
        if max_retries < 0 or base_delay <= 0 or max_delay < base_delay:
            raise ValueError("Invalid retry settings")
        self.providers = [primary, *fallbacks]
        self.name = primary.name
        self.prices = prices
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sleep = sleep
        self._jitter = jitter
        missing = prices.missing([provider.name for provider in self.providers])
        if missing:
            logger.warning(
                "No price for %s. Turn totals will exclude these calls.", ", ".join(missing)
            )

    async def aclose(self) -> None:
        async with AsyncExitStack() as stack:
            for provider in self.providers:
                stack.push_async_callback(provider.aclose)

    def delay(self, retry: int) -> float:
        """Full jitter: a random wait up to base * 2**retry, capped."""
        return self._jitter(min(self.max_delay, self.base_delay * 2**retry))

    async def stream(self, request: LLMRequest) -> AsyncGenerator[LLMEvent, None]:
        failure: ProviderError | None = None
        for index, provider in enumerate(self.providers):
            for attempt in range(self.max_retries + 1):
                started = False
                try:
                    async with provider_stream(provider, request) as stream:
                        async for event in stream:
                            if isinstance(event, LLMTextDelta) and not event.text:
                                continue
                            if isinstance(event, (LLMTextDelta, LLMToolCall)):
                                event = event.model_copy(update={"model": provider.name})
                            elif isinstance(event, LLMUsage):
                                event = event.model_copy(
                                    update={
                                        "model": provider.name,
                                        "cost_usd": self.prices.cost(provider.name, event.usage),
                                    }
                                )
                            started = True
                            yield event
                    if not started:
                        raise ProviderError("Provider returned an empty stream", retryable=True)
                    if index or attempt:
                        logger.info(
                            "%s answered on attempt %d (model %d of %d)",
                            provider.name,
                            attempt + 1,
                            index + 1,
                            len(self.providers),
                        )
                    return
                except ProviderError as exc:
                    if started:
                        raise  # never retry or fall back once output was passed on
                    failure = exc
                    if exc.retryable and attempt < self.max_retries:
                        wait = self.delay(attempt)
                        logger.warning(
                            "%s failed (%s); retry %d of %d in %.2fs",
                            provider.name,
                            exc,
                            attempt + 1,
                            self.max_retries,
                            wait,
                        )
                        await self._sleep(wait)
                        continue
                    break
            if index + 1 < len(self.providers):
                logger.warning(
                    "%s failed (%s); falling back to %s",
                    provider.name,
                    failure,
                    self.providers[index + 1].name,
                )
        assert failure is not None
        raise failure  # the last model's error; earlier ones were logged
