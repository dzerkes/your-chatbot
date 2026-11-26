"""API authentication, ownership, SSE framing and disconnect persistence."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import anyio
import pytest
from app.auth import parse_dev_tokens
from app.main import create_app
from chatbot.agent import Agent
from chatbot.config import Settings
from chatbot.errors import ChatbotError, NotFoundError, ProviderError
from chatbot.feedback import Feedback, FeedbackInput, ScoreDelivery
from chatbot.llm.base import (
    LLMEvent,
    LLMProvider,
    LLMRequest,
    LLMStop,
    LLMTextDelta,
    LLMToolCall,
    LLMUsage,
)
from chatbot.llm.fake import ScriptedProvider
from chatbot.store.base import Turn
from chatbot.store.memory import MemoryStore
from chatbot.store.memory_store import InMemoryUserMemoryStore
from chatbot.tools.base import Tool, ToolContext, ToolInput
from chatbot.tools.pagination import ToolPage
from chatbot.tools.registry import ToolRegistry
from chatbot.tracing import Tracing
from chatbot.types import Event, ToolUse, TurnFailed, Usage, User
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import SecretStr, TypeAdapter
from starlette.types import Message, Scope

pytestmark = pytest.mark.unit
ALICE = {"Authorization": "Bearer alice-token"}
BOB = {"Authorization": "Bearer bob-token"}


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("app.main.configure_logging", lambda settings: None)


@dataclass
class Harness:
    app: FastAPI
    store: MemoryStore
    closed: list[bool]
    contexts: list[ToolContext]


def api(
    provider: LLMProvider,
    *,
    tracing: Tracing | None = None,
    memories: InMemoryUserMemoryStore | None = None,
) -> Harness:
    settings = Settings(
        auth_dev_tokens=SecretStr("alice-token:alice:engineering|employees,bob-token:bob")
    )
    store = MemoryStore()
    closed: list[bool] = []
    contexts: list[ToolContext] = []

    class LookupInput(ToolInput):
        query: str

    async def lookup(inputs: LookupInput, context: ToolContext) -> ToolPage:
        contexts.append(context)  # asserted by the test; a raise here only becomes a tool error
        return ToolPage(items=[inputs.query], total=1)

    @asynccontextmanager
    async def build(configured: Settings) -> AsyncIterator[Agent]:
        tools = ToolRegistry()
        tools.register(Tool("lookup", "Lookup", LookupInput, lookup))
        try:
            yield Agent(
                store,
                provider,
                tools,
                configured,
                system="Test",
                tracing=tracing,
                memories=memories,
            )
        finally:
            await provider.aclose()
            if tracing is not None:
                await tracing.aclose()
            closed.append(True)

    return Harness(create_app(settings, chatbot_factory=build), store, closed, contexts)


def answer(text: str = "Hello") -> list[LLMEvent | Exception]:
    return [
        LLMTextDelta(text=text),
        LLMUsage(usage=Usage(input_tokens=10, output_tokens=2)),
        LLMStop(reason="end_turn"),
    ]


def parse_events(text: str) -> list[Event]:
    adapter = TypeAdapter(Event)
    result: list[Event] = []
    for frame in text.strip().split("\n\n"):
        if frame.startswith(":"):
            continue
        name, data = frame.split("\n", 1)
        assert name.startswith("event: ") and data.startswith("data: ")
        event = adapter.validate_json(data.removeprefix("data: "))
        assert name == f"event: {event.type}"
        result.append(event)
    return result


def test_health_auth_and_resource_lifecycle() -> None:
    harness = api(ScriptedProvider([]))
    with TestClient(harness.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        for headers in ({}, {"Authorization": "Basic whatever"}, {"Authorization": "Bearer wrong"}):
            response = client.get("/sessions", headers=headers)
            assert response.status_code == 401
            assert response.headers["www-authenticate"] == "Bearer"
        assert client.get("/sessions", headers=ALICE).json() == []
        assert harness.closed == []
    assert harness.closed == [True]
    assert harness.app.state.runtime is None


def test_sessions_are_owned_and_chat_checks_before_streaming() -> None:
    provider = ScriptedProvider([answer()])
    harness = api(provider)
    with TestClient(harness.app) as client:
        created = client.post("/sessions", json={"title": "  Projects  "}, headers=ALICE)
        assert created.status_code == 201
        session = created.json()
        assert session["title"] == "Projects"
        session_id = session["id"]
        assert client.get("/sessions", headers=ALICE).json() == [session]
        assert client.get("/sessions", headers=BOB).json() == []
        for target in (session_id, str(uuid4())):
            assert client.get(f"/sessions/{target}", headers=BOB).status_code == 404
            denied = client.post(f"/sessions/{target}/chat", json={"message": "Hi"}, headers=BOB)
            assert denied.status_code == 404
            assert denied.headers["content-type"].startswith("application/json")
        assert provider.requests == []
        detail = client.get(f"/sessions/{session_id}", headers=ALICE).json()
        assert detail == {**session, "messages": [], "feedback": []}
        assert client.post("/sessions", headers=ALICE).status_code == 201


def test_chat_sse_tools_resume_and_history() -> None:
    provider = ScriptedProvider(
        [
            [
                LLMToolCall(
                    tool_use=ToolUse(id="call-1", name="lookup", arguments={"query": "Project"})
                ),
                LLMUsage(usage=Usage(input_tokens=5, output_tokens=1)),
                LLMStop(reason="tool_use"),
            ],
            answer("First\n\nevent: injected"),
            answer("Follow-up"),
        ]
    )
    harness = api(provider)
    with TestClient(harness.app) as client:
        session_id = client.post("/sessions", json={}, headers=ALICE).json()["id"]
        response = client.post(
            f"/sessions/{session_id}/chat", json={"message": "  Find  "}, headers=ALICE
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-accel-buffering"] == "no"
        events = parse_events(response.text)
        assert [event.type for event in events] == [
            "tool_call_started",
            "tool_call_finished",
            "text_delta",
            "turn_completed",
        ]
        assert all(str(event.session_id) == session_id for event in events)
        assert len({event.turn_id for event in events}) == 1
        # Identity and groups reach tools from configuration, never from the request.
        assert [(c.user.external_id, c.user.groups) for c in harness.contexts] == [
            ("alice", ["employees", "engineering"])
        ]
        assert all(event.trace_id for event in events)
        assert (
            events[-1].model_dump()["usage"] == Usage(input_tokens=15, output_tokens=3).model_dump()
        )
        detail = client.get(f"/sessions/{session_id}", headers=ALICE).json()
        assert [message["role"] for message in detail["messages"]] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert detail["messages"][0]["content"][0]["text"] == "  Find  "
        followup = client.post(
            f"/sessions/{session_id}/chat", json={"message": "Next"}, headers=ALICE
        )
        assert parse_events(followup.text)[-1].type == "turn_completed"
        assert len(provider.requests[-1].messages) == 5
        assert len(client.get(f"/sessions/{session_id}", headers=ALICE).json()["messages"]) == 6


def test_stream_failure_is_a_terminal_event_and_persists_partial_text() -> None:
    harness = api(
        ScriptedProvider(
            [[LLMTextDelta(text="Partial"), ProviderError("Unavailable", retryable=True)]]
        )
    )
    with TestClient(harness.app) as client:
        session_id = client.post("/sessions", json={}, headers=ALICE).json()["id"]
        response = client.post(
            f"/sessions/{session_id}/chat", json={"message": "Hi"}, headers=ALICE
        )
        assert response.status_code == 200
        events = parse_events(response.text)
        assert [event.type for event in events] == ["text_delta", "turn_failed"]
        assert events[-1].model_dump()["retryable"] is True
        detail = client.get(f"/sessions/{session_id}", headers=ALICE).json()
        assert detail["messages"][-1]["content"][0]["text"] == "Partial"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"message": " "},
        {"message": "x" * 16_001},
        {"message": "Hi", "user": {"external_id": "bob"}},
    ],
)
def test_invalid_chat_body_does_not_call_provider(body: dict[str, Any]) -> None:
    provider = ScriptedProvider([])
    harness = api(provider)
    with TestClient(harness.app) as client:
        session_id = client.post("/sessions", json={}, headers=ALICE).json()["id"]
        assert (
            client.post(f"/sessions/{session_id}/chat", json=body, headers=ALICE).status_code == 422
        )
        assert provider.requests == []
        assert (
            client.post("/sessions", json={"owner_id": str(uuid4())}, headers=ALICE).status_code
            == 422
        )


@pytest.mark.parametrize(
    "configuration",
    ["", "secret", "secret:", ":alice", "secret:alice,secret:bob", "secret:alice:Bad Group"],
)
def test_bad_auth_configuration_is_rejected_without_echoing_tokens(configuration: str) -> None:
    with pytest.raises(ChatbotError) as caught:
        parse_dev_tokens(SecretStr(configuration))
    assert str(caught.value) == "Invalid AUTH_DEV_TOKENS configuration"


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
async def test_disconnect_closes_stream_and_saves_interrupted_turn(spec_version: str) -> None:
    class BlockingProvider:
        name = "fake:blocking"
        closed = False

        async def aclose(self) -> None:
            pass

        async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
            try:
                yield LLMTextDelta(text="Partial")
                await asyncio.Event().wait()
            finally:
                self.closed = True

    exporter = InMemorySpanExporter()
    tracer = TracerProvider(resource=Resource({"service.name": "test"}), shutdown_on_exit=False)
    tracer.add_span_processor(SimpleSpanProcessor(exporter))
    provider = BlockingProvider()
    harness = api(provider, tracing=Tracing(tracer))
    async with harness.app.router.lifespan_context(harness.app):
        user = await harness.store.upsert_user(User(external_id="alice", display_name="alice"))
        session = await harness.store.create_session(user.id)
        path = f"/sessions/{session.id}/chat"
        scope: Scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": spec_version},
            "http_version": "1.1",
            "scheme": "http",
            "method": "POST",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"authorization", b"Bearer alice-token"),
                (b"content-type", b"application/json"),
            ],
            "server": ("test", 80),
            "client": ("test", 1234),
        }
        sent_request = False
        disconnected = asyncio.Event()

        async def receive() -> Message:
            nonlocal sent_request
            if not sent_request:
                sent_request = True
                return {"type": "http.request", "body": json.dumps({"message": "Hi"}).encode()}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message: Message) -> None:
            if b"event: text_delta" in message.get("body", b""):
                disconnected.set()
                if spec_version == "2.4":
                    raise OSError("Client disconnected")

        await asyncio.wait_for(harness.app(scope, receive, send), timeout=2)
        assert provider.closed
        turns = await harness.store.list_turns(session.id, user.id)
        assert len(turns) == 1 and turns[0].status == "failed"
        messages = await harness.store.list_messages(session.id, user.id)
        assert [message.role for message in messages] == ["user", "assistant"]
        assert messages[-1].content[0].model_dump()["text"] == "Partial"
    spans = exporter.get_finished_spans()
    assert len(spans) == 3 and all(span.status.status_code == StatusCode.ERROR for span in spans)


def test_dev_tokens_carry_optional_groups() -> None:
    tokens = parse_dev_tokens(SecretStr("t1:alice:finance|employees| finance ,t2:bob,t3:carol:"))
    assert [(t.token, t.identity, t.groups) for t in tokens] == [
        (b"t1", "alice", ("employees", "finance")),
        (b"t2", "bob", ()),
        (b"t3", "carol", ()),
    ]


def test_memories_are_listed_and_deleted_per_user() -> None:
    memories = InMemoryUserMemoryStore()
    harness = api(ScriptedProvider([]), memories=memories)
    with TestClient(harness.app) as client:
        assert client.get("/memories", headers=ALICE).json() == []
        # The API created the user on first request; upsert returns the same canonical id.
        alice = anyio.run(
            harness.store.upsert_user, User(external_id="alice", display_name="alice")
        )

        async def seed() -> None:
            await memories.add(alice.id, "Prefers bullet points", "I like bullet points", None)
            await memories.add(alice.id, "Works on the Research Portal", "I work on it", None)

        anyio.run(seed)
        listed = client.get("/memories", headers=ALICE).json()
        assert [m["fact"] for m in listed] == [
            "Works on the Research Portal",
            "Prefers bullet points",
        ]
        assert set(listed[0]) == {"id", "fact", "evidence", "source_session_id", "created_at"}
        client.get("/sessions", headers=BOB)
        assert client.get("/memories", headers=BOB).json() == []
        assert client.delete(f"/memories/{listed[0]['id']}", headers=BOB).status_code == 404
        assert client.delete(f"/memories/{listed[0]['id']}", headers=ALICE).status_code == 204
        assert [m["fact"] for m in client.get("/memories", headers=ALICE).json()] == [
            "Prefers bullet points"
        ]
        assert client.delete(f"/memories/{listed[0]['id']}", headers=ALICE).status_code == 404
        assert client.get("/memories").status_code == 401


def test_memory_endpoints_are_404_when_memory_is_off() -> None:
    harness = api(ScriptedProvider([]))
    with TestClient(harness.app) as client:
        response = client.get("/memories", headers=ALICE)
        assert response.status_code == 404 and response.json()["detail"] == "Memory is disabled"
        assert client.delete("/memories/1", headers=ALICE).status_code == 404


@pytest.mark.parametrize("limit", ["rate", "spend"])
def test_chat_limits_apply_across_sessions_before_generation(
    limit: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "1" if limit == "rate" else "20")
    script = answer()
    script[1] = LLMUsage(usage=Usage(input_tokens=10), cost_usd=Decimal("2"))
    provider = ScriptedProvider([script if limit == "spend" else answer(), answer()])
    with TestClient(api(provider).app) as client:

        def chat_new(headers: dict[str, str]) -> list[Event]:
            session = client.post("/sessions", json={}, headers=headers).json()["id"]
            return parse_events(
                client.post(
                    f"/sessions/{session}/chat", json={"message": "Hi"}, headers=headers
                ).text
            )

        assert chat_new(ALICE)[-1].type == "turn_completed"
        blocked = chat_new(ALICE)
        assert len(blocked) == 1 and isinstance(blocked[0], TurnFailed)
        assert blocked[0].code == (
            "rate_limit_exceeded" if limit == "rate" else "cost_cap_exceeded"
        )
        assert "Try again" in blocked[0].message
        assert chat_new(BOB)[-1].type == "turn_completed"
        assert len(provider.requests) == 2


def test_feedback_is_owned_revisable_and_queued_for_trace() -> None:
    harness = api(ScriptedProvider([answer()]))
    with TestClient(harness.app) as client:
        session_id = client.post("/sessions", json={}, headers=ALICE).json()["id"]
        client.post(f"/sessions/{session_id}/chat", json={"message": "Hi"}, headers=ALICE)
        detail = client.get(f"/sessions/{session_id}", headers=ALICE).json()
        turn_id = detail["messages"][0]["turn_id"]
        url = f"/sessions/{session_id}/turns/{turn_id}/feedback"
        assert client.put(url, json={"rating": "down"}, headers=BOB).status_code == 404
        missing = f"/sessions/{session_id}/turns/{uuid4()}/feedback"
        assert client.put(missing, json={"rating": "up"}, headers=ALICE).status_code == 404
        for body in ({"rating": "meh"}, {"rating": "up", "comment": "x" * 2001}, {}):
            assert client.put(url, json=body, headers=ALICE).status_code == 422
        first = client.put(url, json={"rating": "down", "comment": "  wrong  "}, headers=ALICE)
        assert first.status_code == 200
        assert first.json()["rating"] == "down" and first.json()["comment"] == "wrong"
        revised = client.put(url, json={"rating": "up"}, headers=ALICE).json()
        assert revised["created_at"] == first.json()["created_at"]
        assert revised["comment"] is None
        detail = client.get(f"/sessions/{session_id}", headers=ALICE).json()
        assert detail["feedback"] == [revised]


async def test_memory_store_exports_latest_vote_and_retries() -> None:
    store = MemoryStore()
    owner = await store.upsert_user(User(external_id="alice", display_name="Alice"))
    session = await store.create_session(owner.id)
    traced = Turn(session_id=session.id, status="completed", trace_id=uuid4().hex)
    untraced = Turn(session_id=session.id, status="completed")
    running = Turn(session_id=session.id, status="running", trace_id=uuid4().hex)
    for turn in (traced, untraced, running):
        await store.save_turn(turn, [], owner.id)
    with pytest.raises(NotFoundError):
        await store.save_feedback(
            session.id, running.id, owner.id, FeedbackInput(rating="up"), export=True
        )
    await store.save_feedback(
        session.id, untraced.id, owner.id, FeedbackInput(rating="up"), export=True
    )
    await store.save_feedback(
        session.id, traced.id, owner.id, FeedbackInput(rating="down"), export=True
    )
    sent: list[Feedback] = []
    outcome: ScoreDelivery = "retry"

    async def send(feedback: Feedback) -> ScoreDelivery:
        sent.append(feedback)
        return outcome

    assert await store.export_next_feedback(send)
    assert not await store.export_next_feedback(send)  # retry is scheduled later
    latest = await store.save_feedback(
        session.id, traced.id, owner.id, FeedbackInput(rating="up", comment="ok"), export=True
    )
    outcome = "accepted"
    assert await store.export_next_feedback(send)
    assert not await store.export_next_feedback(send)
    assert [f.rating for f in sent] == ["down", "up"] and sent[-1] == latest
