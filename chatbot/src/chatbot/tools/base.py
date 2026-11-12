"""Typed async tool functions with arguments separated from trusted context."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import cast
from uuid import UUID

from pydantic import JsonValue, ValidationError

from chatbot.errors import ToolError
from chatbot.llm.base import ToolDefinition
from chatbot.logging import log_context
from chatbot.tools.pagination import MAX_PAGE_SIZE, MAX_RESULT_BYTES, ToolPage
from chatbot.types import DomainModel, User


class ToolInput(DomainModel):
    """Base for model-supplied arguments; undeclared fields are rejected."""


@dataclass(frozen=True)
class ToolContext:
    user: User
    session_id: UUID
    turn_id: UUID
    trace_id: str | None = None


class Tool[InputT: ToolInput]:
    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[InputT],
        handler: Callable[[InputT, ToolContext], Awaitable[ToolPage]],
    ) -> None:
        if not name or not description:
            raise ValueError("Tools require a name and description")
        self.name = name
        self.description = description
        self.input_model = input_model
        self.handler = handler

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=cast(dict[str, JsonValue], self.input_model.model_json_schema()),
        )

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolContext,
        *,
        max_items: int = MAX_PAGE_SIZE,
        max_bytes: int = MAX_RESULT_BYTES,
    ) -> ToolPage:
        try:
            inputs = self.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolError("Invalid tool arguments") from exc
        with log_context(
            session_id=context.session_id, turn_id=context.turn_id, trace_id=context.trace_id
        ):
            try:
                result = await self.handler(inputs, context)
                if not isinstance(result, ToolPage):
                    raise ToolError("Tool returned an invalid result")
                if len(result.items) > max_items or result.byte_size() > max_bytes:
                    raise ToolError("Tool result exceeds the size limit")
                return result
            except ToolError:
                raise
            except Exception as exc:
                raise ToolError("Tool execution failed") from exc
