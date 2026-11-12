"""Context strategy contract and shared whole-turn validation."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from chatbot.errors import ContextOverflow
from chatbot.llm.base import ToolDefinition
from chatbot.types import Message, ToolResult, ToolUse


@dataclass(frozen=True)
class BuiltContext:
    system: str
    messages: list[Message]
    estimated_tokens: int


class ContextBuilder(Protocol):
    def build(
        self,
        history: Sequence[Message],
        current: Sequence[Message],
        *,
        system: str,
        tools: Sequence[ToolDefinition] = (),
    ) -> BuiltContext:
        """Build provider input without modifying stored history or the current turn."""
        ...


def history_turns(history: Sequence[Message], current: Sequence[Message]) -> list[list[Message]]:
    """Group ordered history and reject tool exchanges crossing a turn boundary."""
    if not current or len({message.turn_id for message in current}) != 1:
        raise ValueError("Current context must contain exactly one nonempty turn")
    if len({message.session_id for message in [*history, *current]}) != 1:
        raise ValueError("Context messages must belong to one session")
    groups: list[list[Message]] = []
    seen: set[UUID] = {current[0].turn_id}
    for message in history:
        if not groups or groups[-1][0].turn_id != message.turn_id:
            if message.turn_id in seen:
                raise ValueError(
                    "Context turns must be contiguous and separate from the current turn"
                )
            seen.add(message.turn_id)
            groups.append([])
        groups[-1].append(message)
    for group in [*groups, list(current)]:
        pending: set[str] = set()
        calls: set[str] = set()
        for message in group:
            if pending and message.role != "tool":
                raise ValueError("Tool calls must be followed by their results")
            for block in message.content:
                if isinstance(block, ToolUse):
                    if message.role != "assistant" or block.id in calls:
                        raise ValueError("Invalid or duplicate tool call in context")
                    pending.add(block.id)
                    calls.add(block.id)
                elif isinstance(block, ToolResult):
                    if message.role != "tool" or block.tool_use_id not in pending:
                        raise ValueError("Tool result has no matching call in its turn")
                    pending.remove(block.tool_use_id)
        if pending:
            raise ValueError("Context contains a tool call without its result")
    return groups


def bounded_context(
    system: str,
    messages: Sequence[Message],
    estimated_tokens: int,
    max_input_tokens: int,
) -> BuiltContext:
    if estimated_tokens > max_input_tokens:
        raise ContextOverflow("Input context exceeds the configured token budget")
    return BuiltContext(
        system, [message.model_copy(deep=True) for message in messages], estimated_tokens
    )
