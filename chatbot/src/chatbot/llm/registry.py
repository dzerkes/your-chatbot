"""Build LLM adapters from provider:model settings."""

from collections.abc import Callable

from chatbot.config import Settings
from chatbot.errors import ProviderError
from chatbot.llm.anthropic import AnthropicProvider
from chatbot.llm.base import LLMProvider
from chatbot.llm.openai import OpenAIProvider

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
