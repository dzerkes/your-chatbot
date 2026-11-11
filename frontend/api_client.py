"""HTTP-only frontend client and parser for the backend's SSE event contract."""

from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import Literal, Self
from uuid import UUID

import httpx
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError, model_validator


class ApiError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SessionSummary(BaseModel):
    id: UUID
    title: str | None = None
    created_at: datetime
    updated_at: datetime


class ChatMessage(BaseModel):
    id: UUID
    turn_id: UUID
    role: Literal["user", "assistant", "tool", "system"]
    content: list[dict[str, JsonValue]]


class SessionDetail(SessionSummary):
    messages: list[ChatMessage]


class StreamEvent(BaseModel):
    type: Literal[
        "text_delta", "tool_call_started", "tool_call_finished", "turn_completed", "turn_failed"
    ]
    session_id: UUID
    text: str | None = None
    tool_use: dict[str, JsonValue] | None = None
    tool_result: dict[str, JsonValue] | None = None
    message: str | None = None

    @model_validator(mode="after")
    def required_payload(self) -> Self:
        if self.type == "text_delta" and self.text is None:
            raise ValueError("Text delta is missing text")
        if self.type == "tool_call_started" and (
            self.tool_use is None
            or not isinstance(self.tool_use.get("id"), str)
            or not isinstance(self.tool_use.get("name"), str)
        ):
            raise ValueError("Tool call is missing its ID or name")
        if self.type == "tool_call_finished" and (
            self.tool_result is None or not isinstance(self.tool_result.get("tool_use_id"), str)
        ):
            raise ValueError("Tool result is missing its call ID")
        if self.type == "turn_failed" and not self.message:
            raise ValueError("Failed turn is missing its message")
        return self


def parse_sse(lines: Iterable[str]) -> Iterator[StreamEvent]:
    """Parse named JSON events, including comments and multiline data fields."""
    data: list[str] = []
    event_name = "message"
    size = 0
    for line in lines:
        line = line.removesuffix("\r")
        if line == "":
            if data:
                try:
                    event = StreamEvent.model_validate_json("\n".join(data))
                except ValidationError as exc:
                    raise ApiError("The chat service sent an invalid event.") from exc
                if event_name not in {"message", event.type}:
                    raise ApiError("The chat service sent an inconsistent event.")
                yield event
            data, event_name, size = [], "message", 0
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            event_name = value
        elif field == "data":
            size += len(value.encode("utf-8")) + 1
            if size > 65_536:
                raise ApiError("The chat service sent an event that is too large.")
            data.append(value)
    if data:
        raise ApiError("The reply ended mid-event. Reload the conversation before trying again.")


class ApiClient:
    def __init__(self, base_url: str, token: str, *, client: httpx.Client | None = None) -> None:
        self._headers = {"Authorization": f"Bearer {token}"}
        self._owns_client = client is None
        self._client = (
            client
            if client is not None
            else httpx.Client(
                base_url=base_url.rstrip("/") + "/",
                timeout=httpx.Timeout(connect=5, read=90, write=10, pool=5),
                follow_redirects=False,
            )
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        if self._owns_client:
            self._client.close()

    @staticmethod
    def _check(response: httpx.Response) -> None:
        if response.is_success:
            return
        messages = {
            401: "Your access token is missing or invalid.",
            404: "This conversation was not found or is not available to this account.",
            422: "The request was not accepted. Check your message and try again.",
        }
        raise ApiError(
            messages.get(response.status_code, "The chat service is temporarily unavailable."),
            status_code=response.status_code,
        )

    def _request(self, method: str, path: str, *, body: dict[str, str] | None = None) -> JsonValue:
        try:
            response = self._client.request(
                method, path, json=body, headers=self._headers, timeout=10
            )
            self._check(response)
            return TypeAdapter(JsonValue).validate_python(response.json())
        except httpx.RequestError as exc:
            raise ApiError("Could not reach the chat service. Try again shortly.") from exc
        except (ValueError, ValidationError) as exc:
            raise ApiError("The chat service sent an invalid response.") from exc

    def list_sessions(self) -> list[SessionSummary]:
        try:
            return TypeAdapter(list[SessionSummary]).validate_python(
                self._request("GET", "sessions")
            )
        except ValidationError as exc:
            raise ApiError("The chat service sent an invalid session list.") from exc

    def create_session(self, title: str | None = None) -> SessionSummary:
        try:
            return SessionSummary.model_validate(
                self._request("POST", "sessions", body={"title": title} if title else {})
            )
        except ValidationError as exc:
            raise ApiError("The chat service sent an invalid conversation.") from exc

    def get_session(self, session_id: UUID) -> SessionDetail:
        try:
            detail = SessionDetail.model_validate(self._request("GET", f"sessions/{session_id}"))
            if detail.id != session_id:
                raise ApiError("The chat service sent an unexpected conversation.")
            return detail
        except ValidationError as exc:
            raise ApiError("The chat service sent an invalid conversation.") from exc

    def stream_chat(self, session_id: UUID, message: str) -> Iterator[StreamEvent]:
        terminal = False
        try:
            with self._client.stream(
                "POST",
                f"sessions/{session_id}/chat",
                json={"message": message},
                headers=self._headers,
            ) as response:
                self._check(response)
                if not response.headers.get("content-type", "").startswith("text/event-stream"):
                    raise ApiError("The chat service did not send a reply stream.")
                for event in parse_sse(response.iter_lines()):
                    if event.session_id != session_id or terminal:
                        raise ApiError("The chat service sent an unexpected event.")
                    terminal = event.type in {"turn_completed", "turn_failed"}
                    yield event
        except httpx.RequestError as exc:
            raise ApiError(
                "The connection was interrupted. Reload the conversation before trying again."
            ) from exc
        if not terminal:
            raise ApiError("The reply ended early. Reload the conversation before trying again.")
