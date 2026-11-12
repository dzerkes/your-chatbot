"""Replaceable input-token estimation without a provider SDK or network call."""

import json
from collections.abc import Sequence
from typing import Protocol

from chatbot.llm.base import ToolDefinition
from chatbot.types import Message


class TokenCounter(Protocol):
    def count(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolDefinition]
    ) -> int:
        """Estimate all input tokens, including the system prompt and tool schemas."""
        ...


class ApproximateTokenCounter:
    """Estimate one token per three UTF-8 bytes, plus message framing overhead.

    This is a budgeting heuristic, not a provider tokenizer or billing counter.
    Only provider-visible content is counted; persistence metadata is excluded.
    """

    def count(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolDefinition]
    ) -> int:
        payload = {
            "system": system,
            "messages": [
                {
                    "role": message.role,
                    "content": [block.model_dump(mode="json") for block in message.content],
                }
                for message in messages
            ],
            "tools": [tool.model_dump(mode="json") for tool in tools],
        }
        size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        return (size + 2) // 3 + 8 * len(messages) + 12
