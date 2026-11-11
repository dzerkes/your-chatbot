"""Scripted conversations exercise the loop without network or a database."""

import asyncio
from collections.abc import Sequence
from uuid import UUID

import pytest
from chatbot.agent import Agent
from chatbot.config import Settings
from chatbot.errors import ProviderError, ToolError
from chatbot.llm.base import LLMEvent, LLMStop, LLMTextDelta, LLMToolCall, LLMUsage
from chatbot.llm.fake import ScriptedProvider
from chatbot.store.base import Session, Turn
from chatbot.store.memory import MemoryStore
from chatbot.tools.base import Tool, ToolContext, ToolInput
from chatbot.tools.pagination import ToolPage
from chatbot.tools.registry import ToolRegistry
from chatbot.types import (
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

pytestmark = pytest.mark.unit


def answer(text: str = "Hello") -> list[LLMEvent | Exception]:
    return [
        LLMTextDelta(text=text),
        LLMUsage(usage=Usage(input_tokens=10, output_tokens=2)),
        LLMStop(reason="end_turn"),
    ]


def calls(*ids: str) -> list[LLMEvent | Exception]:
    return [
        *(
            LLMToolCall(tool_use=ToolUse(id=value, name="lookup", arguments={"query": value}))
            for value in ids
        ),
        LLMUsage(usage=Usage(input_tokens=5, output_tokens=1)),
        LLMStop(reason="tool_use"),
    ]


async def setup(
    scripts: Sequence[Sequence[LLMEvent | Exception]],
    *,
    max_steps: int = 8,
) -> tuple[Agent, MemoryStore, ScriptedProvider, User, Session]:
    store = MemoryStore()
    user = await store.upsert_user(User(external_id="alice", display_name="Alice"))
    session = await store.create_session(user.id)
    provider = ScriptedProvider(scripts)
    registry = ToolRegistry()

    class LookupInput(ToolInput):
        query: str

    async def lookup(inputs: LookupInput, context: ToolContext) -> ToolPage:
        assert context.user.id == user.id
        assert context.session_id == session.id
        assert context.trace_id
        if inputs.query == "fail":
            raise ToolError("Lookup unavailable", retryable=True)
        return ToolPage(items=[inputs.query], total=1)

    registry.register(Tool("lookup", "Look up a value", LookupInput, lookup))
    agent = Agent(
        store, provider, registry, Settings(agent_max_steps=max_steps), system="Test system"
    )
    return agent, store, provider, user, session


async def test_plain_answer_persistence_usage_and_followup_history() -> None:
    agent, store, provider, user, session = await setup([answer(), answer("Again")])
    events = [event async for event in agent.respond(session.id, "Hi", user)]
    assert isinstance(events[0], TextDelta) and events[0].text == "Hello"
    assert isinstance(events[-1], TurnCompleted)
    assert events[-1].usage == Usage(input_tokens=10, output_tokens=2)
    turn = (await store.list_turns(session.id, user.id))[0]
    assert turn.status == "completed" and turn.input_tokens == 10 and turn.output_tokens == 2
    assert turn.trace_id == events[-1].trace_id
    assert turn.completed_at is not None and turn.latency_ms is not None
    messages = await store.list_messages(session.id, user.id)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[1].model == provider.name
    assert messages[1].usage == events[-1].usage
    assert provider.requests[0].system == "Test system"
    assert provider.requests[0].max_output_tokens == agent.settings.llm_max_output_tokens
    _ = [event async for event in agent.respond(session.id, "And now?", user)]
    assert [m.sequence for m in await store.list_messages(session.id, user.id)] == [0, 1, 2, 3]
    assert provider.requests[1].messages[:2] == messages


@pytest.mark.parametrize("count", [1, 2])
async def test_single_and_multiple_tools_across_steps(count: int) -> None:
    agent, store, provider, user, session = await setup(
        [calls(*(f"first-{index}" for index in range(count))), calls("second"), answer("Found")]
    )
    events = [event async for event in agent.respond(session.id, "Find values", user)]
    assert sum(isinstance(e, ToolCallStarted) for e in events) == count + 1
    assert sum(isinstance(e, ToolCallFinished) for e in events) == count + 1
    assert isinstance(events[-1], TurnCompleted)
    assert events[-1].usage == Usage(input_tokens=20, output_tokens=4)
    messages = await store.list_messages(session.id, user.id)
    results = [block for m in messages for block in m.content if isinstance(block, ToolResult)]
    assert len(results) == count + 1 and not any(result.is_error for result in results)
    assert len(provider.requests) == 3
    assert provider.requests[1].messages[-1].role == "tool"
    assert isinstance(provider.requests[1].messages[-1].content[0], ToolResult)


@pytest.mark.parametrize(
    "call",
    [
        ToolUse(id="bad-1", name="lookup", arguments={"query": "fail"}),
        ToolUse(id="bad-2", name="lookup", arguments={}),
        ToolUse(id="bad-3", name="missing", arguments={}),
    ],
)
async def test_tool_error_goes_back_to_model(call: ToolUse) -> None:
    agent, store, provider, user, session = await setup(
        [
            [LLMToolCall(tool_use=call), LLMUsage(usage=Usage()), LLMStop(reason="tool_use")],
            answer(),
        ]
    )
    events = [event async for event in agent.respond(session.id, "Find", user)]
    result = next(e.tool_result for e in events if isinstance(e, ToolCallFinished))
    assert result.is_error and result.tool_use_id == call.id
    assert provider.requests[1].messages[-1].content == [result]
    assert isinstance(events[-1], TurnCompleted)
    assert (await store.list_turns(session.id, user.id))[0].status == "completed"


async def test_step_limit_saves_failed_turn_with_paired_tools() -> None:
    agent, store, provider, user, session = await setup([calls("one"), answer()], max_steps=1)
    events = [event async for event in agent.respond(session.id, "Find", user)]
    assert isinstance(events[-1], TurnFailed) and events[-1].code == "step_limit_exceeded"
    assert len(provider.requests) == 1
    messages = await store.list_messages(session.id, user.id)
    assert [m.role for m in messages] == ["user", "assistant", "tool"]
    assert (await store.list_turns(session.id, user.id))[0].status == "failed"


async def test_partial_provider_failure_is_saved_and_can_be_followed_up() -> None:
    agent, store, provider, user, session = await setup(
        [
            [LLMTextDelta(text="Partial"), ProviderError("Unavailable", retryable=True)],
            answer(),
        ]
    )
    events = [event async for event in agent.respond(session.id, "Hi", user)]
    assert isinstance(events[-1], TurnFailed) and events[-1].retryable
    messages = await store.list_messages(session.id, user.id)
    assert messages[-1].content == [Text(text="Partial")]
    _ = [event async for event in agent.respond(session.id, "Try again", user)]
    assert provider.requests[1].messages[:2] == messages


async def test_ownership_and_blank_input_do_not_call_model_or_save() -> None:
    agent, store, provider, user, session = await setup([answer()])
    bob = await store.upsert_user(User(external_id="bob", display_name="Bob"))
    for text, caller in [("Hi", bob), ("  ", user)]:
        events = [event async for event in agent.respond(session.id, text, caller)]
        assert len(events) == 1 and isinstance(events[0], TurnFailed)
    assert provider.requests == []
    assert await store.list_messages(session.id, user.id) == []


async def test_closing_stream_saves_partial_turn_and_releases_session() -> None:
    agent, store, _provider, user, session = await setup([answer(), answer("Recovered")])
    stream = agent.respond(session.id, "Hi", user)
    assert isinstance(await anext(stream), TextDelta)
    await stream.aclose()
    assert (await store.list_turns(session.id, user.id))[0].status == "failed"
    events = [event async for event in agent.respond(session.id, "Again", user)]
    assert isinstance(events[-1], TurnCompleted)


async def test_interrupted_tool_batch_fills_missing_results() -> None:
    agent, store, _provider, user, session = await setup([calls("one", "two")])
    stream = agent.respond(session.id, "Find", user)
    assert isinstance(await anext(stream), ToolCallStarted)
    await stream.aclose()
    messages = await store.list_messages(session.id, user.id)
    results = [block for m in messages for block in m.content if isinstance(block, ToolResult)]
    assert [result.tool_use_id for result in results] == ["one", "two"]
    assert all(result.is_error for result in results)


async def test_same_session_turns_are_serialized() -> None:
    agent, store, provider, user, session = await setup([answer("One"), answer("Two")])

    async def consume(text: str) -> list[Event]:
        return [event async for event in agent.respond(session.id, text, user)]

    results = await asyncio.gather(consume("First"), consume("Second"))
    assert all(isinstance(events[-1], TurnCompleted) for events in results)
    assert [len(request.messages) for request in provider.requests] == [1, 3]
    assert [m.sequence for m in await store.list_messages(session.id, user.id)] == [0, 1, 2, 3]


async def test_persistence_failure_does_not_report_completion() -> None:
    class BrokenStore(MemoryStore):
        async def save_turn(self, turn: Turn, messages: Sequence[Message], owner_id: UUID) -> None:
            raise RuntimeError("private database information")

    agent, _store, _provider, user, _session = await setup([answer()])
    broken = BrokenStore()
    await broken.upsert_user(user)
    new_session = await broken.create_session(user.id)
    agent.store = broken
    events = [event async for event in agent.respond(new_session.id, "Hi", user)]
    assert isinstance(events[-1], TurnFailed) and events[-1].code == "store_error"
    assert "private" not in events[-1].message
