"""Tool-result compaction, running summaries, the stage pipeline and cache-stable prefixes."""

import itertools
from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest
from chatbot.agent import Agent
from chatbot.config import Settings
from chatbot.context import tokens as tokens_module
from chatbot.context.base import SessionSummary
from chatbot.context.compact_tools import compact_note, compact_turn
from chatbot.context.pipeline import SUMMARY_INTRO, PipelineContextBuilder
from chatbot.context.summary import Summarizer
from chatbot.context.tokens import ApproximateTokenCounter
from chatbot.errors import ContextOverflow, ProviderError
from chatbot.llm.base import (
    LLMEvent,
    LLMRequest,
    LLMStop,
    LLMTextDelta,
    LLMToolCall,
    LLMUsage,
    ToolDefinition,
)
from chatbot.llm.fake import ScriptedProvider
from chatbot.store.memory import MemoryStore
from chatbot.tools.base import Tool, ToolContext, ToolInput
from chatbot.tools.pagination import ToolPage
from chatbot.tools.registry import ToolRegistry
from chatbot.types import Message, Text, ToolResult, ToolUse, TurnCompleted, Usage, User

pytestmark = pytest.mark.unit
BIG = "Detail " * 400  # about 2.8 KB per tool result


def stored_turn(session: UUID, start: int, *, error: bool = False) -> list[Message]:
    turn = uuid4()

    def message(offset: int, role: str, *content: Text | ToolUse | ToolResult) -> Message:
        return Message(
            session_id=session,
            turn_id=turn,
            sequence=start + offset,
            role=role,  # type: ignore[arg-type]
            content=list(content),
        )

    result = (
        ToolResult(tool_use_id=f"c{start}", is_error=True, content="Bad arguments")
        if error
        else ToolResult(tool_use_id=f"c{start}", content={"items": [BIG], "total": 3})
    )
    return [
        message(0, "user", Text(text=f"Question {start}")),
        message(1, "assistant", ToolUse(id=f"c{start}", name="search_docs", arguments={})),
        message(2, "tool", result),
        message(3, "assistant", Text(text=f"Answer {start}")),
    ]


def current_turn(session: UUID, sequence: int) -> list[Message]:
    return [
        Message(
            session_id=session,
            turn_id=uuid4(),
            sequence=sequence,
            role="user",
            content=[Text(text="Now")],
        )
    ]


# --- compaction -------------------------------------------------------------


def test_compaction_replaces_results_with_a_stable_note_and_keeps_pairs() -> None:
    session = uuid4()
    turn = stored_turn(session, 0)
    compacted = compact_turn(turn)
    result = compacted[2].content[0]
    assert isinstance(result, ToolResult) and result.tool_use_id == "c0"
    assert result.content == compact_note("search_docs", {"items": [BIG], "total": 3})
    assert str(result.content).startswith(
        "[Earlier search_docs result omitted to save context: 1 of 3 items, 2."
    )
    assert compacted[:2] == turn[:2] and compacted[3] == turn[3]
    assert compact_turn(turn) == compacted  # deterministic, so prefixes stay cacheable
    assert turn[2].content[0].content == {"items": [BIG], "total": 3}  # type: ignore[union-attr]
    errored = compact_turn(stored_turn(session, 4, error=True))
    assert errored[2].content[0].content == "Bad arguments"  # type: ignore[union-attr]


# --- pipeline ---------------------------------------------------------------


def test_pipeline_compacts_history_but_not_the_turn_in_progress() -> None:
    session = uuid4()
    history = stored_turn(session, 0) + stored_turn(session, 4)
    current = [
        *current_turn(session, 8),
        Message(
            session_id=session,
            turn_id=uuid4(),
            sequence=9,
            role="assistant",
            content=[ToolUse(id="now", name="search_docs", arguments={})],
        ),
    ]
    current[1] = current[1].model_copy(update={"turn_id": current[0].turn_id})
    current.append(
        Message(
            session_id=session,
            turn_id=current[0].turn_id,
            sequence=10,
            role="tool",
            content=[ToolResult(tool_use_id="now", content={"items": [BIG], "total": 1})],
        )
    )
    built = PipelineContextBuilder(
        ("compact_tools",), keep_last_turns=10, max_input_tokens=100_000
    ).build(history, current, system="S")
    results = [b for m in built.messages for b in m.content if isinstance(b, ToolResult)]
    assert [isinstance(r.content, str) for r in results] == [True, True, False]


def test_summary_replaces_covered_turns_with_a_stable_pair() -> None:
    session = uuid4()
    history = [m for start in (0, 4, 8) for m in stored_turn(session, start)]
    builder = PipelineContextBuilder(
        ("summary", "compact_tools"), keep_last_turns=10, max_input_tokens=100_000
    )
    summary = SessionSummary("The user asked about leave.", through_sequence=7)
    first = builder.build(history, current_turn(session, 12), system="S", summary=summary)
    texts = [b.text for m in first.messages for b in m.content if isinstance(b, Text)]
    assert texts[0] == SUMMARY_INTRO + "The user asked about leave."
    assert first.messages[0].role == "user" and first.messages[1].role == "assistant"
    assert "Question 0" not in texts and "Question 4" not in texts and "Question 8" in texts
    again = builder.build(history, current_turn(session, 12), system="S", summary=summary)
    # Same ids and the same provider-visible content on every request.
    assert [(m.id, m.role, m.content) for m in again.messages[:2]] == [
        (m.id, m.role, m.content) for m in first.messages[:2]
    ]
    # Without the summary stage the summary is ignored.
    plain = PipelineContextBuilder(("compact_tools",), keep_last_turns=10, max_input_tokens=10**5)
    assert "Question 0" in str(
        plain.build(history, current_turn(session, 12), system="S", summary=summary).messages
    )


def test_window_and_budget_guard_drop_oldest_turns_only() -> None:
    session = uuid4()
    history = [m for start in range(0, 40, 4) for m in stored_turn(session, start)]
    window = PipelineContextBuilder(
        ("compact_tools", "sliding_window"), keep_last_turns=2, max_input_tokens=100_000
    ).build(history, current_turn(session, 40), system="S")
    assert [m.sequence for m in window.messages if m.role == "user"] == [32, 36, 40]
    counter = ApproximateTokenCounter()
    tight = PipelineContextBuilder(("compact_tools",), keep_last_turns=10, max_input_tokens=300)
    built = tight.build(history, current_turn(session, 40), system="S")
    assert built.estimated_tokens <= 300
    assert counter.count("S", built.messages, []) == built.estimated_tokens
    assert built.messages[-1].sequence == 40
    with pytest.raises(ContextOverflow):
        PipelineContextBuilder(("compact_tools",), keep_last_turns=10, max_input_tokens=5).build(
            history, current_turn(session, 40), system="S"
        )
    with pytest.raises(ValueError, match="Unknown"):
        PipelineContextBuilder(("magic",), keep_last_turns=1, max_input_tokens=1)


# --- summarizer -------------------------------------------------------------


def summary_reply(text: str = "Summary text") -> list[LLMEvent | Exception]:
    return [
        LLMTextDelta(text=text),
        LLMUsage(usage=Usage(input_tokens=50, output_tokens=10)),
        LLMStop(reason="end_turn"),
    ]


def test_summary_is_due_only_in_large_steps() -> None:
    session = uuid4()
    summarizer = Summarizer(ScriptedProvider([]), keep_last_turns=2, trigger_turns=3, system="S")
    history = [m for start in range(0, 16, 4) for m in stored_turn(session, start)]  # 4 turns
    assert summarizer.due(history, current_turn(session, 16), None) == []
    history += stored_turn(session, 16)  # 5 turns = keep 2 + trigger 3
    due = summarizer.due(history, current_turn(session, 20), None)
    assert [turn[0].sequence for turn in due] == [0, 4, 8]
    covered = SessionSummary("S", through_sequence=11)
    assert summarizer.due(history, current_turn(session, 20), covered) == []


async def test_summarizer_reads_compacted_turns_and_the_previous_summary() -> None:
    session = uuid4()
    provider = ScriptedProvider([summary_reply("New summary")])
    summarizer = Summarizer(provider, keep_last_turns=1, trigger_turns=1, system="Rules")
    turns = [stored_turn(session, 0), stored_turn(session, 4)]
    summary, usage = await summarizer.summarize(SessionSummary("Old summary", 0), turns)
    assert summary == SessionSummary("New summary", through_sequence=7)
    assert usage.input_tokens == 50
    prompt = provider.requests[0].messages[0].content[0]
    assert isinstance(prompt, Text)
    assert "Previous summary:\nOld summary" in prompt.text
    assert "Question 4" in prompt.text and "Answer 4" in prompt.text
    assert "Detail Detail" not in prompt.text  # tool results are compacted
    assert "search_docs result omitted" in prompt.text
    empty = Summarizer(ScriptedProvider([summary_reply("  ")]), keep_last_turns=1, trigger_turns=1)
    with pytest.raises(ProviderError, match="no text"):
        await empty.summarize(None, turns)


# --- a long session through the agent (C23 "done when") ---------------------


class LookupInput(ToolInput):
    query: str


async def lookup(inputs: LookupInput, context: ToolContext) -> ToolPage:
    return ToolPage(items=[f"{inputs.query}: {BIG}"], total=1)


def chat_steps(turns: int) -> list[list[LLMEvent | Exception]]:
    steps: list[list[LLMEvent | Exception]] = []
    for n in range(turns):
        steps.append(
            [
                LLMToolCall(
                    tool_use=ToolUse(id=f"t{n}", name="lookup", arguments={"query": f"q{n}"})
                ),
                LLMUsage(usage=Usage(input_tokens=1, output_tokens=1)),
                LLMStop(reason="tool_use"),
            ]
        )
        steps.append(
            [
                LLMTextDelta(text=f"Answer {n}: the lookup found what you needed."),
                LLMUsage(usage=Usage(input_tokens=1, output_tokens=1)),
                LLMStop(reason="end_turn"),
            ]
        )
    return steps


def visible(message: Message) -> tuple[str, list[dict[str, object]]]:
    """What a provider receives for a message; ids and timestamps are not sent."""
    return message.role, [block.model_dump() for block in message.content]


def starts_with(longer: Sequence[Message], shorter: Sequence[Message]) -> bool:
    return len(longer) >= len(shorter) and [visible(m) for m in longer[: len(shorter)]] == [
        visible(m) for m in shorter
    ]


async def run_session(
    strategy: str, turns: int, budget: int
) -> tuple[list[LLMRequest], int, ScriptedProvider]:
    store = MemoryStore()
    user = await store.upsert_user(User(external_id="alice", display_name="Alice"))
    session = await store.create_session(user.id)
    tools = ToolRegistry()
    tools.register(Tool("lookup", "Look something up.", LookupInput, lookup))
    settings = Settings(
        context_strategy=strategy,  # type: ignore[arg-type]
        context_max_input_tokens=budget,
        context_keep_last_turns=10,
        context_summary_trigger_turns=20,
    )
    provider = ScriptedProvider(chat_steps(turns))
    summaries = ScriptedProvider([summary_reply(f"Summary {i}") for i in range(10)])
    summarizer = (
        Summarizer(summaries, keep_last_turns=10, trigger_turns=20, system="Summarize")
        if "summary" in strategy
        else None
    )
    agent = Agent(store, provider, tools, settings, system="Rules " * 100, summarizer=summarizer)
    for n in range(turns):
        events = [event async for event in agent.respond(session.id, f"Question {n}?", user)]
        assert isinstance(events[-1], TurnCompleted), (n, events[-1])
    assert all(r.cache_key == str(session.id) for r in provider.requests)
    return provider.requests, len(summaries.requests), summaries


async def test_sixty_turn_session_stays_in_budget_and_keeps_a_stable_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = 6000
    requests, summary_calls, summaries = await run_session("summary,compact_tools", 60, budget)
    # Requests are already wrapped as untrusted data; count them as sent, without
    # the counter wrapping them a second time.
    monkeypatch.setattr(tokens_module, "wrap_tool_results", lambda messages: list(messages))
    counter = ApproximateTokenCounter()
    tools = [
        ToolDefinition(
            name="lookup",
            description="Look something up.",
            input_schema=LookupInput.model_json_schema(),
        )
    ]
    sizes = [counter.count(r.system, r.messages, tools) for r in requests]
    assert len(requests) == 120 and max(sizes) <= budget
    # Without compaction or summaries, 60 turns of ~2.8 KB results would need ~60k tokens.
    first_steps = requests[::2]

    def summary_text(request: LLMRequest) -> str | None:
        first = request.messages[0].content[0]
        if isinstance(first, Text) and first.text.startswith(SUMMARY_INTRO):
            return first.text
        return None

    summarized_at = [
        n
        for n in range(1, len(first_steps))
        if summary_text(first_steps[n]) != summary_text(first_steps[n - 1])
    ]
    breaks = [
        n
        for n in range(1, len(first_steps))
        if not starts_with(first_steps[n].messages, first_steps[n - 1].messages)
    ]
    # The prefix changes only on turns where a summary was written. Every other
    # request starts with the previous one, so the provider's cache stays valid.
    # This tight budget forces a summary about every 6 turns; the default of
    # 60k tokens leaves room for many more.
    assert breaks == summarized_at
    assert summary_calls == len(summarized_at) <= 60 // 5
    assert all(b - a >= 5 for a, b in itertools.pairwise(summarized_at))
    assert "Summary 0" in str(summaries.requests[1].messages)  # each summary builds on the last


async def test_summaries_follow_the_turn_trigger_when_the_budget_is_roomy() -> None:
    _, summary_calls, _ = await run_session("summary,compact_tools", 60, 100_000)
    assert summary_calls == 2  # turns 31 and 51: keep 10 + trigger 20, then 20 more


async def test_a_sliding_window_alone_breaks_the_prefix_every_turn() -> None:
    requests, _, _ = await run_session("sliding_window", 25, 100_000)
    first_steps = requests[::2]
    breaks = [
        n
        for n in range(1, len(first_steps))
        if not starts_with(first_steps[n].messages, first_steps[n - 1].messages)
    ]
    assert breaks == list(range(11, 25))  # once the window is full, every turn shifts it
