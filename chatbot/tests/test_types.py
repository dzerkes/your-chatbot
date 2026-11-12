from datetime import datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from chatbot.errors import ChatbotError, ProviderError
from chatbot.types import (
    ContentBlock,
    Event,
    Message,
    Text,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    ToolResult,
    ToolUse,
    TurnCompleted,
    TurnFailed,
    Usage,
    User,
)
from pydantic import TypeAdapter, ValidationError

pytestmark = pytest.mark.unit


def test_message_round_trip_preserves_content_and_metadata() -> None:
    message = Message(
        session_id=uuid4(),
        turn_id=uuid4(),
        sequence=2,
        role="assistant",
        content=[
            Text(text="Looking up projects"),
            ToolUse(id="call_1", name="query_projects", arguments={"status": "active"}),
            ToolResult(
                tool_use_id="call_1",
                content={"items": [{"id": 1}], "next_cursor": None},
            ),
        ],
        model="fake:scripted",
        usage=Usage(input_tokens=100, output_tokens=20, cache_read_input_tokens=50),
    )
    restored = Message.model_validate_json(message.model_dump_json())
    assert restored == message
    assert isinstance(restored.content[1], ToolUse)
    assert isinstance(restored.content[2], ToolResult)
    assert restored.created_at.utcoffset() is not None
    assert restored.usage is not None
    assert restored.usage.total_tokens == 120  # Cache reads are already part of input_tokens.


def test_user_groups_are_not_shared() -> None:
    alice = User(external_id="alice", display_name="Alice")
    bob = User(external_id="bob", display_name="Bob")
    alice.groups.append("engineering")
    assert bob.groups == []
    assert User.model_validate_json(alice.model_dump_json()) == alice


def test_content_requires_known_discriminator_and_json_values() -> None:
    adapter = TypeAdapter(ContentBlock)
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "unknown", "text": "hello"})
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {"type": "tool_use", "id": "call", "name": "query", "arguments": {"bad": object()}}
        )


def test_usage_rejects_negative_counts() -> None:
    with pytest.raises(ValidationError):
        Usage(input_tokens=-1)


def test_message_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(ValidationError):
        Message(
            session_id=uuid4(),
            turn_id=uuid4(),
            sequence=0,
            role="user",
            content=[Text(text="Hello")],
            created_at=datetime(2026, 1, 1),
        )


def test_all_event_variants_round_trip() -> None:
    session_id, turn_id = uuid4(), uuid4()
    tool_use = ToolUse(id="call_1", name="query_projects", arguments={"limit": 5})
    tool_result = ToolResult(tool_use_id="call_1", content="Tool unavailable", is_error=True)
    events: list[Event] = [
        TextDelta(session_id=session_id, turn_id=turn_id, text="Hello"),
        ToolCallStarted(session_id=session_id, turn_id=turn_id, tool_use=tool_use),
        ToolCallFinished(session_id=session_id, turn_id=turn_id, tool_result=tool_result),
        TurnCompleted(
            session_id=session_id,
            turn_id=turn_id,
            trace_id="trace-1",
            usage=Usage(input_tokens=100, output_tokens=20),
            cost_usd=Decimal("0.0012"),
        ),
        TurnFailed(
            session_id=session_id,
            turn_id=turn_id,
            code="provider_error",
            message="Provider unavailable",
            retryable=True,
        ),
    ]
    adapter = TypeAdapter(Event)
    for event in events:
        assert adapter.validate_json(adapter.dump_json(event)) == event
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "unknown", "session_id": session_id, "turn_id": turn_id})


def test_provider_failure_can_be_handled_as_application_error() -> None:
    with pytest.raises(ChatbotError) as caught:
        raise ProviderError("Provider unavailable", retryable=True)
    assert caught.value.code == "provider_error"
    assert caught.value.retryable is True
    assert str(caught.value) == "Provider unavailable"
