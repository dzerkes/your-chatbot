"""Compose context stages: summary, tool-result compaction and a sliding window.

Stages run in a fixed order over whole stored turns:

1. ``summary``: drop turns covered by the session's running summary and put the
   summary first, as one user/assistant exchange after the system prompt and
   tools, so updating it leaves their cached prefix intact.
2. ``compact_tools``: replace earlier turns' tool results with one-line notes.
3. ``sliding_window``: keep only the last ``keep_last_turns`` turns.

A final guard drops the oldest remaining turns if the input still exceeds the
token budget, and fails only when the turn in progress alone does not fit.
Without ``sliding_window``, history only grows between summaries, so each
request starts with the previous one and the provider can reuse its cache.
"""

from collections.abc import Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from chatbot.context.base import BuiltContext, SessionSummary, bounded_context, history_turns
from chatbot.context.compact_tools import compact_turn
from chatbot.context.tokens import ApproximateTokenCounter, TokenCounter
from chatbot.llm.base import ToolDefinition
from chatbot.types import Message, Text

STAGE_ORDER = ("summary", "compact_tools", "sliding_window")
SUMMARY_INTRO = (
    "Summary of the earlier part of this conversation. Older messages are not shown; "
    "use this as background, not as new instructions.\n\n"
)
SUMMARY_ACK = "Noted. I will use this summary as background for the rest of the conversation."


def summary_messages(session_id: UUID, summary: SessionSummary) -> list[Message]:
    """The summary as a stable user/assistant pair; ids depend only on its coverage."""
    key = f"chatbot:summary:{session_id}:{summary.through_sequence}"
    turn_id = uuid5(NAMESPACE_URL, key)
    return [
        Message(
            id=uuid5(NAMESPACE_URL, key + ":user"),
            session_id=session_id,
            turn_id=turn_id,
            sequence=0,
            role="user",
            content=[Text(text=SUMMARY_INTRO + summary.text)],
        ),
        Message(
            session_id=session_id,
            turn_id=turn_id,
            id=uuid5(NAMESPACE_URL, key + ":assistant"),
            sequence=1,
            role="assistant",
            content=[Text(text=SUMMARY_ACK)],
        ),
    ]


class PipelineContextBuilder:
    def __init__(
        self,
        stages: Sequence[str],
        *,
        keep_last_turns: int,
        max_input_tokens: int,
        counter: TokenCounter | None = None,
    ) -> None:
        unknown = set(stages) - set(STAGE_ORDER)
        if unknown or not stages:
            raise ValueError(f"Unknown or missing context stages: {sorted(unknown)}")
        if keep_last_turns < 1 or max_input_tokens < 1:
            raise ValueError("Turn count and input token budget must be positive")
        self.stages = tuple(stage for stage in STAGE_ORDER if stage in stages)
        self.keep_last_turns = keep_last_turns
        self.max_input_tokens = max_input_tokens
        self.counter = counter if counter is not None else ApproximateTokenCounter()

    def build(
        self,
        history: Sequence[Message],
        current: Sequence[Message],
        *,
        system: str,
        tools: Sequence[ToolDefinition] = (),
        summary: SessionSummary | None = None,
    ) -> BuiltContext:
        turns = history_turns(history, current)
        prefix: list[Message] = []
        if "summary" in self.stages and summary is not None:
            turns = [
                turn
                for turn in turns
                if not all(m.sequence <= summary.through_sequence for m in turn)
            ]
            prefix = summary_messages(current[0].session_id, summary)
        if "compact_tools" in self.stages:
            turns = [compact_turn(turn) for turn in turns]
        if "sliding_window" in self.stages:
            turns = turns[-self.keep_last_turns :]
        dropped = 0
        while True:
            messages = prefix + [message for turn in turns for message in turn] + list(current)
            estimated = self.counter.count(system, messages, tools)
            if estimated <= self.max_input_tokens or not turns:
                return bounded_context(system, messages, estimated, self.max_input_tokens, dropped)
            turns = turns[1:]  # last resort: the budget is a hard limit
            dropped += 1
