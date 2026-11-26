"""Provider contract checks and Anthropic adapter fixtures; no API calls."""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from anthropic import AsyncAnthropic
from anthropic.types import RawMessageStreamEvent
from chatbot.config import Settings
from chatbot.errors import ProviderError
from chatbot.llm.anthropic import AnthropicProvider
from chatbot.llm.base import (
    LLMProvider,
    LLMRequest,
    LLMStop,
    LLMTextDelta,
    LLMToolCall,
    LLMUsage,
    ToolDefinition,
)
from chatbot.llm.fake import ScriptedProvider
from chatbot.types import Message, Text, ToolResult, ToolUse, Usage
from pydantic import SecretStr, TypeAdapter

pytestmark = pytest.mark.unit
FIXTURES = Path(__file__).parent / "fixtures"


def request(*messages: Message, tools: list[ToolDefinition] | None = None) -> LLMRequest:
    if not messages:
        messages = (
            Message(
                session_id=uuid4(),
                turn_id=uuid4(),
                sequence=0,
                role="user",
                content=[Text(text="Hello")],
            ),
        )
    return LLMRequest(
        system="Answer clearly.",
        messages=list(messages),
        tools=tools or [],
        max_output_tokens=64,
        timeout_seconds=12,
    )


def fixture_events(name: str) -> list[RawMessageStreamEvent]:
    raw = json.loads((FIXTURES / name).read_text())
    adapter = TypeAdapter(RawMessageStreamEvent)
    return [adapter.validate_python(event) for event in raw]


class RecordedStream:
    def __init__(self, events: list[RawMessageStreamEvent]) -> None:
        self.events = events
        self.closed = False

    async def __aenter__(self) -> "RecordedStream":
        return self

    async def __aexit__(self, *_args: object) -> None:
        self.closed = True

    async def __aiter__(self) -> AsyncIterator[RawMessageStreamEvent]:
        for event in self.events:
            yield event


class RecordedMessages:
    def __init__(self, stream: RecordedStream) -> None:
        self.stream = stream
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> RecordedStream:
        self.calls.append(kwargs)
        return self.stream


class RecordedClient:
    def __init__(self, events: list[RawMessageStreamEvent]) -> None:
        self.messages = RecordedMessages(RecordedStream(events))


def provider(events: list[RawMessageStreamEvent]) -> tuple[AnthropicProvider, RecordedClient]:
    client = RecordedClient(events)
    settings = Settings(anthropic_api_key=SecretStr("fixture-key"))
    instance = AnthropicProvider("claude-sonnet-5-5", settings, client=cast(AsyncAnthropic, client))
    return instance, client


@pytest.mark.asyncio
async def test_anthropic_text_stream_and_request_mapping() -> None:
    anthropic, client = provider(fixture_events("anthropic_text_stream.json"))
    _static_contract: LLMProvider = anthropic
    result = [event async for event in _static_contract.stream(request())]
    assert result == [
        LLMTextDelta(text="Hel"),
        LLMTextDelta(text="lo!"),
        LLMUsage(usage=Usage(input_tokens=25, output_tokens=4)),
        LLMStop(reason="end_turn"),
    ]
    call = client.messages.calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    assert call["system"] == "Answer clearly."
    assert call["max_tokens"] == 64
    assert call["timeout"] == 12
    assert call["stream"] is True
    assert call["messages"][0]["role"] == "user"
    assert call["messages"][0]["content"] == [{"type": "text", "text": "Hello"}]
    assert client.messages.stream.closed


@pytest.mark.asyncio
async def test_anthropic_tool_stream_and_cache_usage() -> None:
    anthropic, client = provider(fixture_events("anthropic_tool_stream.json"))
    tool = ToolDefinition(
        name="query_projects",
        description="Find projects",
        input_schema={"type": "object", "properties": {"status": {"type": "string"}}},
    )
    result = [event async for event in anthropic.stream(request(tools=[tool]))]
    assert result == [
        LLMTextDelta(text="Checking projects"),
        LLMToolCall(
            tool_use=ToolUse(
                id="toolu_fixture_1", name="query_projects", arguments={"status": "active"}
            )
        ),
        LLMUsage(
            usage=Usage(
                input_tokens=24,
                output_tokens=12,
                cache_creation_input_tokens=5,
                cache_read_input_tokens=7,
            )
        ),
        LLMStop(reason="tool_use"),
    ]
    assert client.messages.calls[0]["tools"] == [
        {
            "name": "query_projects",
            "description": "Find projects",
            "input_schema": tool.input_schema,
        }
    ]


@pytest.mark.asyncio
async def test_history_maps_parallel_tool_results_to_one_user_message() -> None:
    anthropic, client = provider(fixture_events("anthropic_text_stream.json"))
    session_id, turn_id = uuid4(), uuid4()
    messages = [
        Message(
            session_id=session_id,
            turn_id=turn_id,
            sequence=0,
            role="user",
            content=[Text(text="Compare projects")],
        ),
        Message(
            session_id=session_id,
            turn_id=turn_id,
            sequence=1,
            role="assistant",
            content=[
                ToolUse(id="call-1", name="query_projects", arguments={"status": "active"}),
                ToolUse(id="call-2", name="query_projects", arguments={"status": "closed"}),
            ],
        ),
        Message(
            session_id=session_id,
            turn_id=turn_id,
            sequence=2,
            role="tool",
            content=[ToolResult(tool_use_id="call-1", content={"items": [1]})],
        ),
        Message(
            session_id=session_id,
            turn_id=turn_id,
            sequence=3,
            role="tool",
            content=[ToolResult(tool_use_id="call-2", content="failed", is_error=True)],
        ),
    ]
    _ = [event async for event in anthropic.stream(request(*messages))]
    sent = client.messages.calls[0]["messages"]
    assert [message["role"] for message in sent] == ["user", "assistant", "user"]
    assert sent[1]["content"][0]["input"] == {"status": "active"}
    assert sent[2]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "call-1",
            "content": '{"items":[1]}',
            "is_error": False,
        },
        {"type": "tool_result", "tool_use_id": "call-2", "content": "failed", "is_error": True},
    ]


@pytest.mark.asyncio
async def test_malformed_tool_arguments_fail_and_close_stream() -> None:
    events = fixture_events("anthropic_tool_stream.json")
    events[6] = TypeAdapter(RawMessageStreamEvent).validate_python(
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": "broken"},
        }
    )
    anthropic, client = provider(events)
    with pytest.raises(ProviderError, match="Invalid streamed tool arguments"):
        _ = [event async for event in anthropic.stream(request())]
    assert client.messages.stream.closed


@pytest.mark.asyncio
async def test_incomplete_stream_is_retryable() -> None:
    anthropic, client = provider(fixture_events("anthropic_text_stream.json")[:-1])
    with pytest.raises(ProviderError, match="Incomplete response stream") as caught:
        _ = [event async for event in anthropic.stream(request())]
    assert caught.value.retryable
    assert client.messages.stream.closed


@pytest.mark.asyncio
async def test_scripted_provider_replays_and_copies_requests() -> None:
    scripted = ScriptedProvider(
        [
            [
                LLMTextDelta(text="Hi"),
                LLMUsage(usage=Usage(input_tokens=1, output_tokens=1)),
                LLMStop(reason="end_turn"),
            ],
            [ProviderError("Temporary failure", retryable=True)],
        ]
    )
    _static_contract: LLMProvider = scripted
    first = request()
    assert [event async for event in _static_contract.stream(first)] == [
        LLMTextDelta(text="Hi"),
        LLMUsage(usage=Usage(input_tokens=1, output_tokens=1)),
        LLMStop(reason="end_turn"),
    ]
    first.messages[0].content.clear()
    assert scripted.requests[0].messages[0].content == [Text(text="Hello")]
    with pytest.raises(ProviderError, match="Temporary failure") as caught:
        _ = [event async for event in scripted.stream(request())]
    assert caught.value.retryable
    with pytest.raises(ProviderError, match="No scripted response remains"):
        _ = [event async for event in scripted.stream(request())]


def test_missing_api_key_fails_before_network() -> None:
    with pytest.raises(ProviderError, match="API key is not configured"):
        AnthropicProvider("claude-sonnet-5-5", Settings())
