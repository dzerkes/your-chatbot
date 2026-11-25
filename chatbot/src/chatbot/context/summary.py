"""Running session summaries, written with LLM_SUMMARY_MODEL in large, infrequent steps.

A summary covers every stored message up to ``through_sequence``. When more
than ``keep_last_turns + trigger_turns`` stored turns lie beyond it, all but
the last ``keep_last_turns`` are folded into the summary in one call, and the
result is saved on the session. Between those steps the summary and the turns
after it do not change, so the prompt prefix stays cacheable: with the
defaults (10 kept, trigger 20) the prefix changes about once every 20 turns.

The summarizer reads turns with tool results already compacted, so retrieved
text never reaches it; it sees the user's messages and the assistant's answers.
"""

import logging
from collections.abc import Callable, Sequence
from importlib.resources import files

from chatbot.context.base import SessionSummary, history_turns
from chatbot.context.compact_tools import compact_turn
from chatbot.errors import ProviderError
from chatbot.llm.base import LLMProvider, LLMRequest, LLMTextDelta, LLMUsage, provider_stream
from chatbot.types import Message, Text, ToolResult, ToolUse, Usage

logger = logging.getLogger(__name__)
MAX_MESSAGE_CHARS = 2000
# Summarize early once the context uses this share of CONTEXT_MAX_INPUT_TOKENS.
PRESSURE_SHARE = 0.8
SUMMARY_MAX_OUTPUT_TOKENS = 1024


def _transcript(turns: Sequence[list[Message]]) -> str:
    lines: list[str] = []
    for turn in turns:
        for message in compact_turn(turn):
            for block in message.content:
                if isinstance(block, Text):
                    text = block.text.strip()
                    if len(text) > MAX_MESSAGE_CHARS:
                        text = text[:MAX_MESSAGE_CHARS] + " …"
                    lines.append(f"{message.role.upper()}: {text}")
                elif isinstance(block, ToolUse):
                    lines.append(f"ASSISTANT called {block.name} with {block.arguments}")
                elif isinstance(block, ToolResult):
                    lines.append(f"TOOL: {block.content}")
        lines.append("")
    return "\n".join(lines).strip()


class Summarizer:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        keep_last_turns: int,
        trigger_turns: int,
        timeout_seconds: float | None = None,
        system: str | None = None,
    ) -> None:
        if keep_last_turns < 1 or trigger_turns < 1:
            raise ValueError("Summary turn counts must be positive")
        self.provider = provider
        self.keep_last_turns = keep_last_turns
        self.trigger_turns = trigger_turns
        self.timeout_seconds = timeout_seconds
        self.system = system or files("chatbot").joinpath("prompts/summary.md").read_text(
            encoding="utf-8"
        )

    def due(
        self,
        history: Sequence[Message],
        current: Sequence[Message],
        summary: SessionSummary | None,
        *,
        under_pressure: bool = False,
    ) -> list[list[Message]]:
        """The stored turns to fold into the summary now, or [] when it is not yet time.

        It is time when ``trigger_turns`` turns beyond the kept ones are unsummarized,
        or earlier when the context nears the token budget (``under_pressure``), so
        the budget guard never has to drop turns one request at a time.
        """
        turns = history_turns(history, current)
        if summary is not None:
            turns = [t for t in turns if not all(m.sequence <= summary.through_sequence for m in t)]
        enough = len(turns) >= self.keep_last_turns + self.trigger_turns
        if not enough and not (under_pressure and len(turns) > self.keep_last_turns):
            return []
        return turns[: -self.keep_last_turns]

    async def summarize(
        self,
        previous: SessionSummary | None,
        turns: Sequence[list[Message]],
        *,
        on_usage: Callable[[LLMUsage], None] | None = None,
    ) -> tuple[SessionSummary, Usage]:
        """Report billable usage as it arrives, even if the summary later fails."""
        if not turns:
            raise ValueError("Nothing to summarize")
        parts = []
        if previous is not None:
            parts.append(f"Previous summary:\n{previous.text}")
        parts.append(f"New turns:\n{_transcript(turns)}")
        request = LLMRequest(
            system=self.system,
            messages=[
                Message(
                    session_id=turns[0][0].session_id,
                    turn_id=turns[-1][0].turn_id,
                    sequence=0,
                    role="user",
                    content=[Text(text="\n\n".join(parts))],
                )
            ],
            max_output_tokens=SUMMARY_MAX_OUTPUT_TOKENS,
            timeout_seconds=self.timeout_seconds,
        )
        text: list[str] = []
        usage: Usage | None = None
        async with provider_stream(self.provider, request) as stream:
            async for event in stream:
                if isinstance(event, LLMTextDelta):
                    text.append(event.text)
                elif isinstance(event, LLMUsage):
                    if usage is not None:
                        raise ProviderError("Summary model sent duplicate usage")
                    usage = event.usage
                    if on_usage is not None:
                        on_usage(event)
        summary = "".join(text).strip()
        if not summary:
            raise ProviderError("The summary model returned no text")
        through = max(message.sequence for turn in turns for message in turn)
        return SessionSummary(text=summary, through_sequence=through), usage or Usage()
