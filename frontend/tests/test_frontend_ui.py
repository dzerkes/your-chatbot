"""Streamlit reruns/reloads keep conversations on the HTTP service."""

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.unit
APP = Path(__file__).resolve().parents[1] / "app.py"


class FakeApi:
    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}
        self.owners: dict[str, str] = {}
        self.messages: dict[str, list[dict[str, Any]]] = {}
        self.prompts: list[str] = []
        self.failure: str | None = None

    def request(self, request: httpx.Request) -> httpx.Response:
        owner = {"Bearer alice-token": "alice", "Bearer bob-token": "bob"}.get(
            request.headers.get("authorization", "")
        )
        if owner is None:
            return httpx.Response(401)
        if request.url.path == "/sessions":
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json=[row for key, row in self.sessions.items() if self.owners[key] == owner],
                )
            session_id = str(uuid4())
            row = {
                "id": session_id,
                "title": json.loads(request.content).get("title"),
                "created_at": "2026-11-12T00:00:00Z",
                "updated_at": "2026-11-12T00:00:00Z",
            }
            self.sessions[session_id], self.owners[session_id], self.messages[session_id] = (
                row,
                owner,
                [],
            )
            return httpx.Response(201, json=row)
        session_id = request.url.path.split("/")[2]
        if self.owners.get(session_id) != owner:
            return httpx.Response(404)
        if request.method == "GET":
            return httpx.Response(
                200, json={**self.sessions[session_id], "messages": self.messages[session_id]}
            )
        prompt = json.loads(request.content)["message"]
        self.prompts.append(prompt)
        turn_id, call_id = str(uuid4()), "call-" + str(uuid4())
        result = {
            "type": "tool_result",
            "tool_use_id": call_id,
            "is_error": False,
            "content": {"items": [{"name": "Research Portal"}], "total": 12, "next_cursor": None},
        }
        tool = {"type": "tool_use", "id": call_id, "name": "query_projects", "arguments": {}}
        reply = (
            "Partial reply"
            if self.failure
            else f"Answer {len(self.prompts)}: Research Portal is active."
        )
        for role, content in (
            ("user", [{"type": "text", "text": prompt}]),
            ("assistant", [tool]),
            ("tool", [result]),
            ("assistant", [{"type": "text", "text": reply}]),
        ):
            self.messages[session_id].append(
                {"id": str(uuid4()), "turn_id": turn_id, "role": role, "content": content}
            )
        events: list[dict[str, Any]] = [
            {"type": "tool_call_started", "tool_use": tool},
            {"type": "tool_call_finished", "tool_result": result},
            {"type": "text_delta", "text": reply},
            {"type": "turn_completed"},
        ]
        if self.failure == "terminal":
            events[-1] = {"type": "turn_failed", "message": "The model is temporarily unavailable."}
        elif self.failure == "truncated":
            events.pop()
        frames = ": connected\n\n" + "".join(
            f"event: {event['type']}\ndata: {json.dumps({**event, 'session_id': session_id})}\n\n"
            for event in events
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=frames)


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeApi:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BACKEND_URL", raising=False)
    monkeypatch.delenv("FRONTEND_PORT", raising=False)
    server = FakeApi()
    real_client = httpx.Client

    def client(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(server.request)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    return server


def test_new_ask_reload_reopen_and_continue(service: FakeApi) -> None:
    app = AppTest.from_file(str(APP), default_timeout=10).run()
    assert not app.exception and not app.chat_input
    app.text_input(key="access_token").set_value("alice-token").run()
    app.chat_input[0].set_value("Which projects are active?").run()
    assert not app.exception
    assert len(app.chat_message) == 2 and len(service.sessions) == 1
    assert app.status[0].label == "Searched projects: 12 results"
    assert app.status[0].proto.expanded is False
    assert "Research Portal" in app.markdown[-1].value
    session_id = str(app.query_params["session"])

    reloaded = AppTest.from_file(str(APP), default_timeout=10)
    reloaded.query_params["session"] = session_id
    reloaded.run()
    assert not reloaded.chat_input
    reloaded.text_input(key="access_token").set_value("alice-token").run()
    assert len(reloaded.chat_message) == 2
    reloaded.button(key="new_chat").click().run()
    assert not reloaded.chat_message and "session" not in reloaded.query_params
    reloaded.button(key=f"session-{session_id}").click().run()
    assert len(reloaded.chat_message) == 2
    reloaded.chat_input[0].set_value("When does it start?").run()
    assert not reloaded.exception and len(reloaded.chat_message) == 4
    assert service.prompts == ["Which projects are active?", "When does it start?"]
    assert len(service.sessions) == 1 and len(service.messages[session_id]) == 8


def test_token_change_clears_previous_conversation(service: FakeApi) -> None:
    app = AppTest.from_file(str(APP), default_timeout=10).run()
    app.text_input(key="access_token").set_value("alice-token").run()
    app.chat_input[0].set_value("Alice question").run()
    assert app.chat_message
    app.text_input(key="access_token").set_value("bob-token").run()
    assert not app.exception and not app.chat_message and "session" not in app.query_params
    assert not any(button.label == "Alice question" for button in app.button)


def test_invalid_token_shows_friendly_error(service: FakeApi) -> None:
    app = AppTest.from_file(str(APP), default_timeout=10).run()
    app.text_input(key="access_token").set_value("bad-token").run()
    assert not app.exception and not app.chat_input
    assert "invalid" in app.error[0].value


@pytest.mark.parametrize("failure", ["terminal", "truncated"])
def test_stream_failure_keeps_partial_reply_and_does_not_retry(
    service: FakeApi, failure: str
) -> None:
    service.failure = failure
    app = AppTest.from_file(str(APP), default_timeout=10).run()
    app.text_input(key="access_token").set_value("alice-token").run()
    app.chat_input[0].set_value("Find projects").run()
    assert not app.exception and app.error
    assert any("Partial reply" in element.value for element in app.markdown)
    assert service.prompts == ["Find projects"]
