"""Select context strategies from application settings."""

from chatbot.config import Settings
from chatbot.context.base import ContextBuilder, SessionSummary
from chatbot.context.full import FullContextBuilder
from chatbot.context.pipeline import PipelineContextBuilder
from chatbot.context.sliding_window import SlidingWindowContextBuilder
from chatbot.context.tokens import TokenCounter

__all__ = ["ContextBuilder", "SessionSummary", "build_context_builder"]


def build_context_builder(
    settings: Settings, *, counter: TokenCounter | None = None
) -> ContextBuilder:
    strategy = settings.context_strategy
    if strategy == ("full",):
        return FullContextBuilder(settings.context_max_input_tokens, counter=counter)
    if strategy == ("sliding_window",):
        return SlidingWindowContextBuilder(
            settings.context_keep_last_turns, settings.context_max_input_tokens, counter=counter
        )
    return PipelineContextBuilder(
        strategy,
        keep_last_turns=settings.context_keep_last_turns,
        max_input_tokens=settings.context_max_input_tokens,
        counter=counter,
    )
