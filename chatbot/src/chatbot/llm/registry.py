"""Build LLM adapters from provider:model settings."""

from collections.abc import Callable, Sequence
from functools import cached_property

from chatbot.config import Settings
from chatbot.errors import ProviderError
from chatbot.llm.anthropic import AnthropicProvider
from chatbot.llm.base import LLMProvider
from chatbot.llm.gateway import LLMGateway
from chatbot.llm.openai import OpenAIProvider
from chatbot.llm.pricing import PriceTable

type ProviderFactory = Callable[[str, Settings], LLMProvider]


class ProviderRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._factories: dict[str, ProviderFactory] = {
            "anthropic": AnthropicProvider,
            "openai": OpenAIProvider,
        }

    def get(self, name: str) -> LLMProvider:
        provider, separator, model = name.partition(":")
        if not separator or not provider.strip() or not model.strip():
            raise ProviderError("Model must use provider:model notation")
        factory = self._factories.get(provider)
        if factory is None:
            raise ProviderError(f"Unknown LLM provider: {provider}")
        return factory(model, self.settings)

    def get_primary(self) -> LLMProvider:
        return self.get(self.settings.llm_primary)

    @cached_property
    def _prices(self) -> PriceTable:
        return PriceTable.load(self.settings.llm_prices_file)

    def gateway(self, model: str, fallbacks: Sequence[str] = ()) -> LLMGateway:
        """``model`` with retries, then ``fallbacks`` in order, priced from the table."""
        settings = self.settings
        prices = self._prices
        primary, *backups = (self.get(name) for name in dict.fromkeys((model, *fallbacks)))
        return LLMGateway(
            primary,
            backups,
            prices=prices,
            max_retries=settings.llm_max_retries,
            base_delay=settings.llm_retry_base_seconds,
            max_delay=settings.llm_retry_max_seconds,
        )

    def primary_gateway(self) -> LLMGateway:
        return self.gateway(self.settings.llm_primary, self.settings.llm_fallbacks)
