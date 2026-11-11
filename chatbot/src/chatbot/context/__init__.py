"""Select context strategies from application settings."""

from chatbot.config import Settings
from chatbot.context.base import ContextBuilder
from chatbot.context.full import FullContextBuilder
from chatbot.context.sliding_window import SlidingWindowContextBuilder
from chatbot.context.tokens import TokenCounter
from chatbot.errors import ChatbotError


def build_context_builder(
    settings: Settings, *, counter: TokenCounter | None = None
) -> ContextBuilder:
    if settings.context_strategy == "full":
        return FullContextBuilder(settings.context_max_input_tokens, counter=counter)
    if settings.context_strategy == "sliding_window":
        return SlidingWindowContextBuilder(
            settings.context_keep_last_turns, settings.context_max_input_tokens, counter=counter
        )
    raise ChatbotError(f"Context strategy is not implemented: {settings.context_strategy}")
