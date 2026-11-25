"""Model prices and the cost of one call, in exact decimals.

Prices live in ``prices.toml`` (USD per 1M tokens, with sources and dates) so
they can be updated without code changes; ``LLM_PRICES_FILE`` adds or
overrides entries. A model without a price has no cost (``None``), never a
guessed one, and the gateway warns about it at startup.
"""

import tomllib
from decimal import Decimal
from importlib.resources import files
from pathlib import Path

from pydantic import Field, PositiveInt, ValidationError

from chatbot.errors import ChatbotError
from chatbot.types import DomainModel, Usage

PER_TOKENS = Decimal(1_000_000)


class TokenRates(DomainModel):
    input: Decimal = Field(ge=0)
    cached_input: Decimal = Field(ge=0)
    output: Decimal = Field(ge=0)
    cache_write: Decimal | None = Field(default=None, ge=0)  # None: billed as input

    def cost(self, usage: Usage) -> Decimal:
        """Usage counts cached tokens inside input_tokens; split them out before pricing."""
        cache_read = usage.cache_read_input_tokens
        cache_write = usage.cache_creation_input_tokens
        uncached = max(usage.input_tokens - cache_read - cache_write, 0)
        write_price = self.cache_write if self.cache_write is not None else self.input
        return (
            uncached * self.input
            + cache_read * self.cached_input
            + cache_write * write_price
            + usage.output_tokens * self.output
        ) / PER_TOKENS


class LongContextRates(TokenRates):
    above_input_tokens: PositiveInt


class ModelPrice(TokenRates):
    long_context: LongContextRates | None = None

    def cost(self, usage: Usage) -> Decimal:
        if self.long_context and usage.input_tokens > self.long_context.above_input_tokens:
            return self.long_context.cost(usage)
        return super().cost(usage)


class PriceTable:
    def __init__(self, prices: dict[str, ModelPrice]) -> None:
        self.prices = dict(prices)

    @classmethod
    def load(cls, override: str | None = None) -> "PriceTable":
        default = files("chatbot").joinpath("llm/prices.toml").read_text(encoding="utf-8")
        prices = _parse(default, "built-in prices.toml")
        if override:
            try:
                content = Path(override).read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise ChatbotError(f"Could not read model prices from {override}") from exc
            prices |= _parse(content, override)
        return cls(prices)

    def get(self, model: str) -> ModelPrice | None:
        return self.prices.get(model)

    def cost(self, model: str, usage: Usage) -> Decimal | None:
        price = self.get(model)
        return price.cost(usage) if price is not None else None

    def missing(self, models: list[str]) -> list[str]:
        return [model for model in models if model not in self.prices]


def _parse(text: str, source: str) -> dict[str, ModelPrice]:
    try:
        raw = tomllib.loads(text, parse_float=Decimal)
        return {name: ModelPrice.model_validate(entry) for name, entry in raw.items()}
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ChatbotError(f"Invalid model prices in {source}") from exc
