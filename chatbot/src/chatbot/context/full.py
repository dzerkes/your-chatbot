"""Retain the complete conversation, or fail when its input budget is exceeded."""

from collections.abc import Sequence

from chatbot.context.base import BuiltContext, bounded_context, history_turns
from chatbot.context.tokens import ApproximateTokenCounter, TokenCounter
from chatbot.llm.base import ToolDefinition
from chatbot.types import Message


class FullContextBuilder:
    def __init__(self, max_input_tokens: int, *, counter: TokenCounter | None = None) -> None:
        if max_input_tokens < 1:
            raise ValueError("Input token budget must be positive")
        self.max_input_tokens = max_input_tokens
        self.counter = counter if counter is not None else ApproximateTokenCounter()

    def build(
        self,
        history: Sequence[Message],
        current: Sequence[Message],
        *,
        system: str,
        tools: Sequence[ToolDefinition] = (),
    ) -> BuiltContext:
        history_turns(history, current)
        messages = [*history, *current]
        return bounded_context(
            system, messages, self.counter.count(system, messages, tools), self.max_input_tokens
        )
