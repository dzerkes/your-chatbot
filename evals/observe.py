"""Run one case through Agent.respond and record what happened."""

import json
from contextlib import aclosing
from decimal import Decimal
from time import perf_counter
from typing import Literal

from chatbot.agent import Agent
from chatbot.types import (
    DomainModel,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnCompleted,
    TurnFailed,
    Usage,
    User,
)
from pydantic import Field, JsonValue

from evals.dataset import SourceRef


class ToolCallRecord(DomainModel):
    name: str
    arguments: dict[str, JsonValue]
    result: JsonValue = None
    is_error: bool = False


class Observation(DomainModel):
    status: Literal["completed", "failed", "error"]
    answer: str = ""
    error: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    trace_id: str | None = None
    usage: Usage = Field(default_factory=Usage)
    cost_usd: Decimal = Decimal("0")
    latency_seconds: float = 0.0
    first_token_seconds: float | None = None

    def returned_sources(self) -> list[str]:
        """Heading paths of returned documents and IDs of returned projects."""
        found: list[str] = []
        for call in self.tool_calls:
            if call.is_error or not isinstance(call.result, dict):
                continue
            items = call.result.get("items")
            for item in items if isinstance(items, list) else []:
                if not isinstance(item, dict):
                    continue
                if call.name == "search_docs" and isinstance(item.get("title"), str):
                    section = item.get("section")
                    found.append(f"{item['title']} > {section}" if section else str(item["title"]))
                elif call.name == "query_projects" and isinstance(item.get("id"), int):
                    found.append(f"project {item['id']}")
        return found

    def returned(self, source: SourceRef) -> bool:
        """A doc ref matches its own heading path and every section beneath it."""
        label = source.label()
        return any(
            found == label or found.startswith(label + " > ") for found in self.returned_sources()
        )

    def tool_text(self) -> str:
        return "\n".join(json.dumps(call.arguments, ensure_ascii=False) for call in self.tool_calls)

    def result_text(self) -> str:
        return "\n".join(json.dumps(call.result, ensure_ascii=False) for call in self.tool_calls)


async def observe(agent: Agent, user: User, question: str) -> Observation:
    """One fresh session, one turn. Errors are recorded rather than raised."""
    session = await agent.store.create_session(user.id)
    answer: list[str] = []
    calls: dict[str, ToolCallRecord] = {}
    observation = Observation(status="error")
    started = perf_counter()
    try:
        async with aclosing(agent.respond(session.id, question, user)) as events:
            async for event in events:
                observation.trace_id = event.trace_id or observation.trace_id
                if isinstance(event, TextDelta):
                    if observation.first_token_seconds is None:
                        observation.first_token_seconds = perf_counter() - started
                    answer.append(event.text)
                elif isinstance(event, ToolCallStarted):
                    calls[event.tool_use.id] = ToolCallRecord(
                        name=event.tool_use.name, arguments=event.tool_use.arguments
                    )
                elif isinstance(event, ToolCallFinished):
                    record = calls.get(event.tool_result.tool_use_id)
                    if record is not None:
                        record.result = event.tool_result.content
                        record.is_error = event.tool_result.is_error
                elif isinstance(event, TurnCompleted):
                    observation.status = "completed"
                    observation.usage, observation.cost_usd = event.usage, event.cost_usd
                elif isinstance(event, TurnFailed):
                    observation.status = "failed"
                    observation.error = f"{event.code}: {event.message}"
    except Exception as exc:  # one broken case must not stop the run
        observation.status, observation.error = "error", f"{type(exc).__name__}: {exc}"
    observation.latency_seconds = perf_counter() - started
    observation.answer = "".join(answer)
    observation.tool_calls = list(calls.values())
    if observation.status == "error" and observation.error is None:
        observation.error = "The turn ended without a completion event"
    return observation
