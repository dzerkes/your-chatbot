"""Retain recent historical turns and the complete turn currently in progress."""

from collections.abc import Sequence

from chatbot.context.base import BuiltContext, bounded_context, history_turns
from chatbot.context.tokens import ApproximateTokenCounter, TokenCounter
from chatbot.llm.base import ToolDefinition
from chatbot.types import Message


class SlidingWindowContextBuilder:
    def __init__(
        self,
        keep_last_turns: int,
        max_input_tokens: int,
        *,
        counter: TokenCounter | None = None,
    ) -> None:
        if keep_last_turns < 1 or max_input_tokens < 1:
            raise ValueError("Turn count and input token budget must be positive")
        self.keep_last_turns = keep_last_turns
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
        turns = history_turns(history, current)[-self.keep_last_turns :]
        while True:
            messages = [message for turn in turns for message in turn] + list(current)
            estimated = self.counter.count(system, messages, tools)
            if estimated <= self.max_input_tokens or not turns:
                return bounded_context(system, messages, estimated, self.max_input_tokens)
            turns = turns[1:]
