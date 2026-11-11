"""Shared provider contract and OpenAI Responses fixture checks."""

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
from chatbot.llm.openai import OpenAIProvider
from chatbot.llm.registry import ProviderRegistry
from chatbot.types import Message, Text, ToolResult, ToolUse, Usage
from openai import AsyncOpenAI
from openai.types.responses import ResponseStreamEvent
from pydantic import SecretStr, TypeAdapter

pytestmark = pytest.mark.unit
FIXTURES = Path(__file__).parent / "fixtures"


class RecordedStream:
    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.closed = False

    async def __aenter__(self) -> "RecordedStream":
        return self

    async def __aexit__(self, *_args: object) -> None:
        self.closed = True

    async def __aiter__(self) -> AsyncIterator[Any]:
        for event in self.events:
            yield event


class RecordedEndpoint:
    def __init__(self, stream: RecordedStream) -> None:
        self.stream = stream
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> RecordedStream:
        self.calls.append(kwargs)
        return self.stream


class RecordedClient:
    def __init__(self, events: list[Any]) -> None:
        endpoint = RecordedEndpoint(RecordedStream(events))
        self.messages = endpoint
        self.responses = endpoint


def provider(kind: str, fixture: str) -> tuple[LLMProvider, RecordedClient]:
    adapter: TypeAdapter[Any]
    if kind == "anthropic":
        adapter = TypeAdapter(RawMessageStreamEvent)
    else:
        adapter = TypeAdapter(ResponseStreamEvent)
    events = [
        adapter.validate_python(event) for event in json.loads((FIXTURES / fixture).read_text())
    ]
    client = RecordedClient(events)
    settings = Settings(
        anthropic_api_key=SecretStr("fixture-key"), openai_api_key=SecretStr("fixture-key")
    )
    if kind == "anthropic":
        return AnthropicProvider(
            "claude-sonnet-5-5", settings, client=cast(AsyncAnthropic, client)
        ), client
    return OpenAIProvider("gpt-5.5", settings, client=cast(AsyncOpenAI, client)), client


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


@pytest.mark.parametrize(
    ("kind", "fixture"),
    [("anthropic", "anthropic_text_stream.json"), ("openai", "openai_text_stream.json")],
)
async def test_text_contract(kind: str, fixture: str) -> None:
    llm, client = provider(kind, fixture)
    events = [event async for event in llm.stream(request())]
    assert events == [
        LLMTextDelta(text="Hel"),
        LLMTextDelta(text="lo!"),
        LLMUsage(
            usage=Usage(
                input_tokens=25,
                output_tokens=4,
                cache_read_input_tokens=5 if kind == "openai" else 0,
                cache_creation_input_tokens=2 if kind == "openai" else 0,
            )
        ),
        LLMStop(reason="end_turn"),
    ]
    assert client.messages.stream.closed
    call = client.messages.calls[0]
    assert call["model"] == ("gpt-5.5" if kind == "openai" else "claude-sonnet-5-5")
    assert call["max_output_tokens" if kind == "openai" else "max_tokens"] == 64
    assert call["stream"] is True
    assert call["timeout"] == 12


@pytest.mark.parametrize(
    ("kind", "fixture", "call_id"),
    [
        ("anthropic", "anthropic_tool_stream.json", "toolu_fixture_1"),
        ("openai", "openai_tool_stream.json", "call_fixture_1"),
    ],
)
async def test_tool_contract(kind: str, fixture: str, call_id: str) -> None:
    llm, client = provider(kind, fixture)
    tool = ToolDefinition(
        name="query_projects",
        description="Find projects",
        input_schema={"type": "object", "properties": {"status": {"type": "string"}}},
    )
    events = [event async for event in llm.stream(request(tools=[tool]))]
    assert events == [
        LLMTextDelta(text="Checking projects"),
        LLMToolCall(
            tool_use=ToolUse(id=call_id, name="query_projects", arguments={"status": "active"})
        ),
        LLMUsage(
            usage=Usage(
                input_tokens=24,
                output_tokens=12,
                cache_creation_input_tokens=5 if kind == "anthropic" else 0,
                cache_read_input_tokens=7,
            )
        ),
        LLMStop(reason="tool_use"),
    ]
    assert client.responses.stream.closed


async def test_openai_history_tools_and_request_mapping() -> None:
    llm, client = provider("openai", "openai_text_stream.json")
    session, turn = uuid4(), uuid4()
    messages = [
        Message(
            session_id=session, turn_id=turn, sequence=0, role="user", content=[Text(text="Find")]
        ),
        Message(
            session_id=session,
            turn_id=turn,
            sequence=1,
            role="assistant",
            content=[ToolUse(id="call-1", name="query_projects", arguments={"status": "active"})],
        ),
        Message(
            session_id=session,
            turn_id=turn,
            sequence=2,
            role="tool",
            content=[
                ToolResult(tool_use_id="call-1", content={"items": [1]}),
                ToolResult(tool_use_id="call-2", content="denied", is_error=True),
            ],
        ),
    ]
    tool = ToolDefinition(
        name="query_projects", description="Find", input_schema={"type": "object"}
    )
    _ = [event async for event in llm.stream(request(*messages, tools=[tool]))]
    call = client.responses.calls[0]
    assert call["instructions"] == "Answer clearly."
    assert call["store"] is False
    assert call["input"] == [
        {"role": "user", "content": "Find"},
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "query_projects",
            "arguments": '{"status":"active"}',
        },
        {"type": "function_call_output", "call_id": "call-1", "output": '{"items":[1]}'},
        {"type": "function_call_output", "call_id": "call-2", "output": '{"error":"denied"}'},
    ]
    assert call["tools"] == [
        {
            "type": "function",
            "name": "query_projects",
            "description": "Find",
            "parameters": {"type": "object"},
            "strict": False,
        }
    ]


async def test_openai_incomplete_stream_and_bad_arguments() -> None:
    llm, client = provider("openai", "openai_text_stream.json")
    client.responses.stream.events.pop()
    with pytest.raises(ProviderError, match="Incomplete response stream") as caught:
        _ = [event async for event in llm.stream(request())]
    assert caught.value.retryable
    assert client.responses.stream.closed

    llm, client = provider("openai", "openai_tool_stream.json")
    bad = client.responses.stream.events[1].model_copy(deep=True)
    bad.item.arguments = "not JSON"
    client.responses.stream.events[1] = bad
    with pytest.raises(ProviderError, match="Invalid streamed tool arguments"):
        _ = [event async for event in llm.stream(request())]
    assert client.responses.stream.closed


async def test_registry_uses_llm_primary_without_code_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PRIMARY", "openai:gpt-5.5")
    settings = Settings(
        anthropic_api_key=SecretStr("fixture-key"), openai_api_key=SecretStr("fixture-key")
    )
    registry = ProviderRegistry(settings)
    openai = registry.get_primary()
    assert isinstance(openai, OpenAIProvider)
    assert openai.name == "openai:gpt-5.5"
    await openai.aclose()
    monkeypatch.setenv("LLM_PRIMARY", "anthropic:claude-sonnet-5-5")
    registry = ProviderRegistry(
        Settings(
            anthropic_api_key=SecretStr("fixture-key"), openai_api_key=SecretStr("fixture-key")
        )
    )
    anthropic = registry.get_primary()
    assert isinstance(anthropic, AnthropicProvider)
    assert anthropic.name == "anthropic:claude-sonnet-5-5"
    await anthropic.aclose()


@pytest.mark.parametrize("name", ["openai", "openai:", ":model", "unknown:model"])
def test_registry_rejects_invalid_or_unknown_provider(name: str) -> None:
    with pytest.raises(ProviderError):
        ProviderRegistry(Settings()).get(name)


def test_openai_missing_key_fails_before_network() -> None:
    with pytest.raises(ProviderError, match="API key is not configured"):
        OpenAIProvider("gpt-5.5", Settings())
