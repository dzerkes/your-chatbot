"""Whole-turn selection, tool pairing and input budgets are provider-independent."""

from collections.abc import Sequence
from typing import Literal
from uuid import UUID, uuid4

import pytest
from chatbot.agent import Agent
from chatbot.config import Settings
from chatbot.context import build_context_builder
from chatbot.context.full import FullContextBuilder
from chatbot.context.sliding_window import SlidingWindowContextBuilder
from chatbot.context.tokens import ApproximateTokenCounter
from chatbot.errors import ChatbotError, ContextOverflow
from chatbot.llm.base import LLMStop, LLMTextDelta, LLMToolCall, LLMUsage, ToolDefinition
from chatbot.llm.fake import ScriptedProvider
from chatbot.store.memory import MemoryStore
from chatbot.tools.base import Tool, ToolContext, ToolInput
from chatbot.tools.pagination import ToolPage
from chatbot.tools.registry import ToolRegistry
from chatbot.types import Message, Text, ToolResult, ToolUse, TurnCompleted, TurnFailed, Usage, User

pytestmark = pytest.mark.unit


class MessageCounter:
    """Predictable test budget independent of tokenization heuristics."""

    def count(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolDefinition]
    ) -> int:
        return len(messages)


def turn(session: UUID, sequence: int, *, tools: bool = False) -> list[Message]:
    turn_id = uuid4()
    messages = [
        Message(
            session_id=session,
            turn_id=turn_id,
            sequence=sequence,
            role="user",
            content=[Text(text=f"Question {sequence}")],
        )
    ]
    if tools:
        messages.extend(
            [
                Message(
                    session_id=session,
                    turn_id=turn_id,
                    sequence=sequence + 1,
                    role="assistant",
                    content=[ToolUse(id=f"call-{sequence}", name="lookup", arguments={})],
                ),
                Message(
                    session_id=session,
                    turn_id=turn_id,
                    sequence=sequence + 2,
                    role="tool",
                    content=[ToolResult(tool_use_id=f"call-{sequence}", content={"items": [1]})],
                ),
            ]
        )
    messages.append(
        Message(
            session_id=session,
            turn_id=turn_id,
            sequence=sequence + len(messages),
            role="assistant",
            content=[Text(text="Answer")],
        )
    )
    return messages


def test_full_context_preserves_order_and_makes_independent_copies() -> None:
    session = uuid4()
    history = [*turn(session, 0), *turn(session, 2, tools=True)]
    current = turn(session, 6)[:1]
    context = FullContextBuilder(100, counter=MessageCounter()).build(
        history, current, system="Rules"
    )
    assert context.messages == [*history, *current]
    assert context.system == "Rules" and context.estimated_tokens == 7
    context.messages[0].content.clear()
    assert history[0].content
    assert current[0].content
    with pytest.raises(ContextOverflow):
        FullContextBuilder(6, counter=MessageCounter()).build(history, current, system="Rules")


def test_window_keeps_last_n_stored_turns_and_all_current_tool_steps() -> None:
    session = uuid4()
    old = turn(session, 0, tools=True)
    recent = [*turn(session, 4), *turn(session, 6, tools=True)]
    current = turn(session, 10, tools=True)[:-1]
    context = SlidingWindowContextBuilder(2, 100, counter=MessageCounter()).build(
        [*old, *recent], current, system="Rules"
    )
    assert context.messages == [*recent, *current]
    assert [
        block.id
        for message in context.messages
        for block in message.content
        if isinstance(block, ToolUse)
    ] == ["call-6", "call-10"]
    assert [
        block.tool_use_id
        for message in context.messages
        for block in message.content
        if isinstance(block, ToolResult)
    ] == ["call-6", "call-10"]


def test_budget_eviction_removes_entire_tool_turns_and_never_current() -> None:
    session = uuid4()
    old, recent = turn(session, 0), turn(session, 2, tools=True)
    current = turn(session, 6)[:1]
    context = SlidingWindowContextBuilder(10, 5, counter=MessageCounter()).build(
        [*old, *recent], current, system="Rules"
    )
    assert context.messages == [*recent, *current]
    smaller = SlidingWindowContextBuilder(10, 4, counter=MessageCounter()).build(
        [*old, *recent], current, system="Rules"
    )
    assert smaller.messages == current
    current_tools = turn(session, 6, tools=True)[:-1]
    with pytest.raises(ContextOverflow):
        SlidingWindowContextBuilder(10, 2, counter=MessageCounter()).build(
            [*old, *recent], current_tools, system="Rules"
        )


@pytest.mark.parametrize("strategy", ["full", "sliding_window"])
def test_incomplete_and_cross_turn_tool_pairs_are_rejected(strategy: str) -> None:
    session = uuid4()
    exchange = turn(session, 0, tools=True)
    current = turn(session, 4)[:1]
    builder = FullContextBuilder(100) if strategy == "full" else SlidingWindowContextBuilder(1, 100)
    with pytest.raises(ValueError, match="tool call without its result"):
        builder.build(exchange[:2], current, system="")
    split = [message.model_copy(deep=True) for message in exchange]
    split[2].turn_id = uuid4()
    with pytest.raises(ValueError):
        builder.build(split, current, system="")
    with pytest.raises(ValueError, match="no matching call"):
        builder.build([exchange[2]], current, system="")


def test_token_estimate_covers_system_tools_and_payload_not_history_metadata() -> None:
    counter = ApproximateTokenCounter()
    current = turn(uuid4(), 0)[:1]
    baseline = counter.count("", current, [])
    assert counter.count("Rules " * 100, current, []) > baseline
    tool = ToolDefinition(
        name="lookup", description="Lookup " * 100, input_schema={"type": "object"}
    )
    assert counter.count("", current, [tool]) > baseline
    changed = current[0].model_copy(deep=True)
    changed.id, changed.session_id, changed.turn_id = uuid4(), uuid4(), uuid4()
    changed.sequence = 999
    changed.model = "model-metadata"
    changed.usage = Usage(input_tokens=999)
    assert counter.count("", [changed], []) == baseline
    changed.content = [Text(text="界" * 100)]
    assert counter.count("", [changed], []) > baseline
    with pytest.raises(ContextOverflow):
        FullContextBuilder(baseline).build([], current, system="Rules " * 100, tools=[tool])


def test_settings_select_strategy_and_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_STRATEGY", "full")
    monkeypatch.setenv("CONTEXT_MAX_INPUT_TOKENS", "50")
    full = build_context_builder(Settings())
    assert isinstance(full, FullContextBuilder) and full.max_input_tokens == 50
    monkeypatch.setenv("CONTEXT_STRATEGY", "sliding_window")
    monkeypatch.setenv("CONTEXT_KEEP_LAST_TURNS", "2")
    window = build_context_builder(Settings())
    assert isinstance(window, SlidingWindowContextBuilder) and window.keep_last_turns == 2
    for strategy in ("compact_tools", "summary"):
        monkeypatch.setenv("CONTEXT_STRATEGY", strategy)
        with pytest.raises(ChatbotError, match="not implemented"):
            build_context_builder(Settings())


@pytest.mark.parametrize(
    "strategy,request_sizes",
    [
        ("full", [1, 3, 5]),
        ("sliding_window", [1, 3, 3]),
    ],
)
async def test_agent_uses_configured_context_and_keeps_all_stored_history(
    strategy: Literal["full", "sliding_window"],
    request_sizes: list[int],
) -> None:
    store = MemoryStore()
    user = await store.upsert_user(User(external_id="alice", display_name="Alice"))
    session = await store.create_session(user.id)
    provider = ScriptedProvider(
        [
            [LLMTextDelta(text="Answer"), LLMUsage(usage=Usage()), LLMStop(reason="end_turn")]
            for _ in range(3)
        ]
    )
    agent = Agent(
        store,
        provider,
        ToolRegistry(),
        Settings(context_strategy=strategy, context_keep_last_turns=1),
        system="Rules",
    )
    for question in ("First", "Second", "Third"):
        events = [event async for event in agent.respond(session.id, question, user)]
        assert isinstance(events[-1], TurnCompleted)
    assert [len(request.messages) for request in provider.requests] == request_sizes
    stored = await store.list_messages(session.id, user.id)
    assert len(stored) == 6 and [message.sequence for message in stored] == list(range(6))


async def test_agent_rechecks_budget_after_tools_and_saves_failed_turn() -> None:
    class EmptyInput(ToolInput):
        pass

    async def lookup(inputs: EmptyInput, context: ToolContext) -> ToolPage:
        return ToolPage(items=["Found"], total=1)

    store = MemoryStore()
    user = await store.upsert_user(User(external_id="alice", display_name="Alice"))
    session = await store.create_session(user.id)
    tools = ToolRegistry()
    tools.register(Tool("lookup", "Lookup", EmptyInput, lookup))
    provider = ScriptedProvider(
        [
            [
                LLMToolCall(tool_use=ToolUse(id="call-1", name="lookup", arguments={})),
                LLMUsage(usage=Usage()),
                LLMStop(reason="tool_use"),
            ]
        ]
    )
    agent = Agent(
        store,
        provider,
        tools,
        Settings(),
        system="Rules",
        context_builder=SlidingWindowContextBuilder(1, 2, counter=MessageCounter()),
    )
    events = [event async for event in agent.respond(session.id, "Find", user)]
    assert isinstance(events[-1], TurnFailed) and events[-1].code == "context_overflow"
    assert len(provider.requests) == 1
    assert [message.role for message in await store.list_messages(session.id, user.id)] == [
        "user",
        "assistant",
        "tool",
    ]
    assert (await store.list_turns(session.id, user.id))[0].status == "failed"
