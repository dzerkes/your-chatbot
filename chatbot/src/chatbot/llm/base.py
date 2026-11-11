"""The interface used by the agent and implemented by every provider."""

from collections.abc import AsyncIterator
from typing import Annotated, Literal, Protocol

from pydantic import Field, JsonValue, PositiveFloat, PositiveInt, field_validator

from chatbot.types import DomainModel, Message, NonEmptyStr, ToolUse, Usage


class ToolDefinition(DomainModel):
    name: NonEmptyStr
    description: str
    input_schema: dict[str, JsonValue]

    @field_validator("input_schema")
    @classmethod
    def object_schema(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if value.get("type") != "object":
            raise ValueError("Tool input schema must describe an object")
        return value


class LLMRequest(DomainModel):
    system: str = ""
    messages: list[Message] = Field(min_length=1)
    tools: list[ToolDefinition] = Field(default_factory=list)
    max_output_tokens: PositiveInt = 2048
    timeout_seconds: PositiveFloat | None = None


class LLMTextDelta(DomainModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class LLMToolCall(DomainModel):
    type: Literal["tool_call"] = "tool_call"
    tool_use: ToolUse


class LLMUsage(DomainModel):
    type: Literal["usage"] = "usage"
    usage: Usage


type StopReason = Literal[
    "end_turn",
    "tool_use",
    "max_tokens",
    "stop_sequence",
    "pause_turn",
    "refusal",
    "model_context_window_exceeded",
    "unknown",
]


class LLMStop(DomainModel):
    type: Literal["stop"] = "stop"
    reason: StopReason


type LLMEvent = Annotated[
    LLMTextDelta | LLMToolCall | LLMUsage | LLMStop,
    Field(discriminator="type"),
]


class LLMProvider(Protocol):
    name: str

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        """Yield text and completed tool calls, then final usage and stop reason."""
        ...

    async def aclose(self) -> None:
        """Release resources owned by the provider."""
        ...
