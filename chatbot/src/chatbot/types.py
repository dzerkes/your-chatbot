"""Provider-neutral domain objects and the public respond() event contract."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    NonNegativeInt,
)

NonEmptyStr = Annotated[str, Field(min_length=1)]


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class User(DomainModel):
    id: UUID = Field(default_factory=uuid4)
    external_id: NonEmptyStr
    display_name: NonEmptyStr
    groups: list[str] = Field(default_factory=list)


class Text(DomainModel):
    type: Literal["text"] = "text"
    text: str


class ToolUse(DomainModel):
    type: Literal["tool_use"] = "tool_use"
    id: NonEmptyStr
    name: NonEmptyStr
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class ToolResult(DomainModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: NonEmptyStr
    content: JsonValue
    is_error: bool = False


type ContentBlock = Annotated[Text | ToolUse | ToolResult, Field(discriminator="type")]


class Usage(DomainModel):
    """Input tokens include cached input tokens; cache fields describe subsets."""

    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    cache_read_input_tokens: NonNegativeInt = 0
    cache_creation_input_tokens: NonNegativeInt = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Message(DomainModel):
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    turn_id: UUID
    sequence: NonNegativeInt
    role: Literal["system", "user", "assistant", "tool"]
    content: list[ContentBlock]
    model: str | None = None
    usage: Usage | None = None
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))


class TurnEvent(DomainModel):
    session_id: UUID
    turn_id: UUID
    trace_id: str | None = None


class TextDelta(TurnEvent):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ToolCallStarted(TurnEvent):
    type: Literal["tool_call_started"] = "tool_call_started"
    tool_use: ToolUse


class ToolCallFinished(TurnEvent):
    type: Literal["tool_call_finished"] = "tool_call_finished"
    tool_result: ToolResult


class TurnCompleted(TurnEvent):
    type: Literal["turn_completed"] = "turn_completed"
    usage: Usage
    cost_usd: Annotated[Decimal, Field(ge=0)] = Decimal("0")


class TurnFailed(TurnEvent):
    type: Literal["turn_failed"] = "turn_failed"
    code: NonEmptyStr
    message: str
    retryable: bool = False


type Event = Annotated[
    TextDelta | ToolCallStarted | ToolCallFinished | TurnCompleted | TurnFailed,
    Field(discriminator="type"),
]
