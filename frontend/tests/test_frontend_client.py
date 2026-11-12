"""HTTP/SSE client behavior without importing the chatbot core."""

import json
from uuid import uuid4

import httpx
import pytest

from frontend.api_client import ApiClient, ApiError, parse_sse

pytestmark = pytest.mark.unit


def test_sse_comments_multiline_and_unicode() -> None:
    session = str(uuid4())
    events = list(
        parse_sse(
            [
                ": connected",
                "",
                "event: text_delta",
                'data: {"type":"text_delta",',
                f'data: "session_id":"{session}","text":"Hello 世界"}}',
                "",
                "retry: 1000",
                "event: turn_completed",
                f'data: {{"type":"turn_completed","session_id":"{session}"}}',
                "",
            ]
        )
    )
    assert [event.type for event in events] == ["text_delta", "turn_completed"]
    assert events[0].text == "Hello 世界"


@pytest.mark.parametrize(
    "lines",
    [
        ["event: text_delta", "data: not-json", ""],
        ["event: text_delta", 'data: {"type":"text_delta"}', ""],
        ["event: text_delta", "data: " + "x" * 65_537, ""],
        ["event: text_delta", "data: {}"],
    ],
)
def test_malformed_and_truncated_frames_fail(lines: list[str]) -> None:
    with pytest.raises(ApiError):
        list(parse_sse(lines))


def test_http_contract_and_authenticated_stream() -> None:
    session_id, turn_id = str(uuid4()), str(uuid4())
    summary = {
        "id": session_id,
        "title": "Projects",
        "created_at": "2026-11-12T00:00:00Z",
        "updated_at": "2026-11-12T00:00:00Z",
    }
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["authorization"] == "Bearer test-token"
        if request.url.path == "/sessions":
            return httpx.Response(
                201 if request.method == "POST" else 200,
                json=summary if request.method == "POST" else [summary],
            )
        if request.url.path.endswith("/chat"):
            assert json.loads(request.content) == {"message": "Hi"}
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=(
                    'event: text_delta\ndata: {"type":"text_delta",'
                    f'"session_id":"{session_id}","text":"Hi"}}\n\n'
                    'event: turn_completed\ndata: {"type":"turn_completed",'
                    f'"session_id":"{session_id}"}}\n\n'
                ),
            )
        return httpx.Response(
            200,
            json={
                **summary,
                "messages": [
                    {
                        "id": str(uuid4()),
                        "turn_id": turn_id,
                        "role": "user",
                        "content": [{"type": "text", "text": "Hi"}],
                    }
                ],
            },
        )

    with (
        httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)) as http,
        ApiClient("http://test", "test-token", client=http) as api,
    ):
        session = api.create_session("Projects")
        assert api.list_sessions()[0].id == session.id
        assert api.get_session(session.id).messages[0].role == "user"
        assert [event.type for event in api.stream_chat(session.id, "Hi")] == [
            "text_delta",
            "turn_completed",
        ]
    assert len(requests) == 4


@pytest.mark.parametrize(
    "failure", ["missing_terminal", "wrong_session", "wrong_name", "after_terminal"]
)
def test_invalid_stream_contract_is_reported(failure: str) -> None:
    session_id = uuid4()
    actual = uuid4() if failure == "wrong_session" else session_id
    name = "text_delta" if failure == "wrong_name" else "turn_completed"
    frame = f'event: {name}\ndata: {{"type":"turn_completed","session_id":"{actual}"}}\n\n'
    if failure == "missing_terminal":
        frame = ": connected\n\n"
    if failure == "after_terminal":
        frame *= 2
    with (
        httpx.Client(
            base_url="http://test",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, text=frame
                )
            ),
        ) as http,
        pytest.raises(ApiError),
    ):
        list(ApiClient("http://test", "token", client=http).stream_chat(session_id, "Hi"))


@pytest.mark.parametrize("status", [401, 404, 422, 500])
def test_http_errors_do_not_expose_tokens_or_server_details(status: int) -> None:
    with httpx.Client(
        base_url="http://test",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(status, json={"detail": "private server info"})
        ),
    ) as http:
        with pytest.raises(ApiError) as caught:
            ApiClient("http://test", "secret-token", client=http).list_sessions()
        assert caught.value.status_code == status
        assert "secret-token" not in str(caught.value) and "private" not in str(caught.value)
