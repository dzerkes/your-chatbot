"""Span hierarchy, Langfuse attributes and disabled tracing without network calls."""

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from chatbot.agent import Agent
from chatbot.config import Settings
from chatbot.errors import ChatbotError, ProviderError, ToolError
from chatbot.llm.base import LLMEvent, LLMStop, LLMTextDelta, LLMToolCall, LLMUsage
from chatbot.llm.fake import ScriptedProvider
from chatbot.store.base import Session
from chatbot.store.memory import MemoryStore
from chatbot.tools.base import Tool, ToolContext, ToolInput
from chatbot.tools.pagination import ToolPage
from chatbot.tools.registry import ToolRegistry
from chatbot.tracing import Tracing
from chatbot.types import TextDelta, ToolUse, TurnCompleted, TurnFailed, Usage, User
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import SecretStr

from chatbot import tracing as tracing_module

pytestmark = pytest.mark.unit


@pytest.fixture
async def telemetry() -> AsyncIterator[tuple[Tracing, InMemorySpanExporter]]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource({"service.name": "test"}), shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing = Tracing(provider)
    try:
        yield tracing, exporter
    finally:
        await tracing.aclose()


async def agent(
    tracing: Tracing,
    scripts: Sequence[Sequence[LLMEvent | Exception]],
) -> tuple[Agent, MemoryStore, User, Session]:
    store = MemoryStore()
    user = await store.upsert_user(User(external_id="alice", display_name="Alice"))
    session = await store.create_session(user.id)
    registry = ToolRegistry()

    class LookupInput(ToolInput):
        query: str

    async def lookup(inputs: LookupInput, context: ToolContext) -> ToolPage:
        if inputs.query == "fail":
            raise ToolError("Lookup unavailable")
        return ToolPage(items=[inputs.query], total=1)

    registry.register(Tool("lookup", "Lookup", LookupInput, lookup))
    return (
        Agent(
            store, ScriptedProvider(scripts), registry, Settings(), system="Test", tracing=tracing
        ),
        store,
        user,
        session,
    )


def answer() -> list[LLMEvent | Exception]:
    return [
        LLMTextDelta(text="Answer"),
        LLMUsage(usage=Usage(input_tokens=4, output_tokens=2)),
        LLMStop(reason="end_turn"),
    ]


def tool(query: str = "value") -> list[LLMEvent | Exception]:
    return [
        LLMToolCall(tool_use=ToolUse(id="call-1", name="lookup", arguments={"query": query})),
        LLMUsage(
            usage=Usage(
                input_tokens=20,
                output_tokens=4,
                cache_read_input_tokens=5,
                cache_creation_input_tokens=3,
            )
        ),
        LLMStop(reason="tool_use"),
    ]


async def test_span_tree_usage_and_persisted_trace_id(
    telemetry: tuple[Tracing, InMemorySpanExporter],
) -> None:
    tracing, exporter = telemetry
    chatbot, store, user, session = await agent(tracing, [tool(), answer()])
    events = [event async for event in chatbot.respond(session.id, "Find", user)]
    assert isinstance(events[-1], TurnCompleted)
    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["llm", "tool:lookup", "llm", "turn", "session"]
    root, turn = spans[-1], spans[-2]
    root_context, turn_context = root.get_span_context(), turn.get_span_context()
    assert root_context is not None and turn_context is not None
    assert root.parent is None
    assert turn.parent is not None and turn.parent.span_id == root_context.span_id
    for child in spans[:3]:
        assert child.parent is not None and child.parent.span_id == turn_context.span_id
    for span in spans:
        attrs = span.attributes or {}
        assert attrs["session.id"] == str(session.id)
        assert attrs["langfuse.session.id"] == str(session.id)
        assert attrs["user.id"] == str(user.id)
        assert attrs["chatbot.turn.id"] == str(events[-1].turn_id)
        latency = attrs["chatbot.latency_ms"]
        assert isinstance(latency, int | float) and latency >= 0
        span_context = span.get_span_context()
        assert span_context is not None
        assert format(span_context.trace_id, "032x") == events[-1].trace_id
    saved = (await store.list_turns(session.id, user.id))[0]
    assert saved.trace_id == events[-1].trace_id
    generation = spans[0].attributes or {}
    assert generation["gen_ai.request.model"] == "scripted"
    assert generation["gen_ai.usage.input_tokens"] == 20
    assert generation["gen_ai.usage.output_tokens"] == 4
    assert generation["chatbot.cost_usd"] == 0 and generation["chatbot.cost_calculated"] is False
    assert json.loads(str(generation["langfuse.observation.usage_details"])) == {
        "input": 12,
        "output": 4,
        "input_cached_tokens": 5,
        "input_cache_write_tokens": 3,
    }
    tool_span = spans[1].attributes or {}
    assert json.loads(str(tool_span["langfuse.observation.input"])) == {"query": "value"}
    size = tool_span["chatbot.tool.result_bytes"]
    assert isinstance(size, int) and size > 0
    assert tool_span["chatbot.tool.is_error"] is False


async def test_tool_errors_are_traced_but_recovery_completes_turn(
    telemetry: tuple[Tracing, InMemorySpanExporter],
) -> None:
    tracing, exporter = telemetry
    chatbot, _store, user, session = await agent(tracing, [tool("fail"), answer()])
    events = [event async for event in chatbot.respond(session.id, "Find", user)]
    assert isinstance(events[-1], TurnCompleted)
    spans = exporter.get_finished_spans()
    failed = next(span for span in spans if span.name == "tool:lookup")
    assert failed.status.status_code == StatusCode.ERROR
    assert (failed.attributes or {})["error.type"] == "tool_error"
    assert (
        next(span for span in spans if span.name == "turn").status.status_code != StatusCode.ERROR
    )


async def test_concurrent_sessions_keep_separate_trace_attributes(
    telemetry: tuple[Tracing, InMemorySpanExporter],
) -> None:
    tracing, exporter = telemetry
    first, _first_store, alice, first_session = await agent(tracing, [answer()])
    second, _second_store, bob, second_session = await agent(tracing, [answer()])

    async def consume(chatbot: Agent, user: User, session: Session) -> None:
        async for _event in chatbot.respond(session.id, "Hi", user):
            await asyncio.sleep(0)

    await asyncio.gather(consume(first, alice, first_session), consume(second, bob, second_session))
    spans = exporter.get_finished_spans()
    assert len(spans) == 6
    root_ids: set[int] = set()
    for span in spans:
        span_context = span.get_span_context()
        assert span_context is not None
        if span.parent is None:
            root_ids.add(span_context.trace_id)
    assert len(root_ids) == 2
    for session, user in ((first_session, alice), (second_session, bob)):
        grouped = [
            span for span in spans if (span.attributes or {}).get("session.id") == str(session.id)
        ]
        assert len(grouped) == 3
        assert all((span.attributes or {}).get("user.id") == str(user.id) for span in grouped)


async def test_provider_error_and_cancellation_finish_all_spans(
    telemetry: tuple[Tracing, InMemorySpanExporter],
) -> None:
    tracing, exporter = telemetry
    chatbot, store, user, session = await agent(
        tracing, [[ProviderError("Unavailable", retryable=True)], answer(), answer()]
    )
    failed = [event async for event in chatbot.respond(session.id, "Hi", user)]
    assert isinstance(failed[-1], TurnFailed)
    assert len(exporter.get_finished_spans()) == 3
    assert all(
        span.status.status_code == StatusCode.ERROR for span in exporter.get_finished_spans()
    )
    stream = chatbot.respond(session.id, "Again", user)
    assert isinstance(await anext(stream), TextDelta)
    await stream.aclose()
    cancelled = exporter.get_finished_spans()[3:]
    assert len(cancelled) == 3
    assert all(span.status.status_code == StatusCode.ERROR for span in cancelled)
    events = [event async for event in chatbot.respond(session.id, "Recovered", user)]
    assert isinstance(events[-1], TurnCompleted)
    turns = await store.list_turns(session.id, user.id)
    assert len({turn.trace_id for turn in turns}) == 3
    assert [turn.status for turn in turns] == ["failed", "failed", "completed"]
    with tracing.span("outside"):
        pass
    outside = exporter.get_finished_spans()[-1]
    assert outside.parent is None and "session.id" not in (outside.attributes or {})


async def test_disabled_tracing_does_not_construct_an_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(**kwargs: Any) -> None:
        pytest.fail("Disabled tracing constructed a network exporter")

    monkeypatch.setattr(tracing_module, "OTLPSpanExporter", unexpected)
    original = trace.get_tracer_provider()
    tracing = Tracing.from_settings(Settings())
    provider = ScriptedProvider([answer()])
    assert tracing.wrap_provider(provider) is provider
    chatbot, _store, user, session = await agent(tracing, [answer()])
    events = [event async for event in chatbot.respond(session.id, "Hi", user)]
    assert isinstance(events[-1], TurnCompleted) and events[-1].trace_id
    await tracing.aclose()
    assert trace.get_tracer_provider() is original


async def test_langfuse_export_configuration_and_shutdown_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = InMemorySpanExporter()
    captured: dict[str, Any] = {}

    def build(**kwargs: Any) -> InMemorySpanExporter:
        captured.update(kwargs)
        return exporter

    monkeypatch.setattr(tracing_module, "OTLPSpanExporter", build)
    original = trace.get_tracer_provider()
    tracing = Tracing.from_settings(
        Settings(
            tracing_enabled=True,
            langfuse_public_key=SecretStr("pk-test"),
            langfuse_secret_key=SecretStr("sk-test"),
        )
    )
    with tracing.span("shutdown-test"):
        pass
    await tracing.aclose()
    assert len(exporter.get_finished_spans()) == 1
    assert captured["endpoint"] == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    assert captured["headers"] == {
        "Authorization": "Basic " + base64.b64encode(b"pk-test:sk-test").decode(),
        "x-langfuse-ingestion-version": "4",
    }
    assert trace.get_tracer_provider() is original


def test_enabled_tracing_requires_both_keys() -> None:
    with pytest.raises(ChatbotError, match="Tracing requires"):
        Tracing.from_settings(Settings(tracing_enabled=True))
