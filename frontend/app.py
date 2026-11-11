"""Streamlit chat UI. All stateful operations go through ApiClient."""

from __future__ import annotations

import hashlib
from itertools import groupby
from typing import TYPE_CHECKING
from uuid import UUID

import streamlit as st
from pydantic import JsonValue, ValidationError

from frontend.api_client import ApiClient, ApiError, ChatMessage
from frontend.config import FrontendSettings

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator
    from streamlit.elements.lib.mutable_status_container import StatusContainer


def tool_summary(name: str, result: dict[str, JsonValue]) -> tuple[str, bool, str]:
    label = {"query_projects": "Searched projects", "search_docs": "Searched documents"}.get(
        name, name.replace("_", " ").capitalize() or "Tool"
    )
    content = result.get("content")
    if result.get("is_error") is True:
        error = content.get("error") if isinstance(content, dict) else content
        return (
            f"{label}: could not complete",
            True,
            str(error or "The request could not be completed."),
        )
    if isinstance(content, dict) and type(content.get("total")) is int:
        total = content["total"]
        items = content.get("items")
        shown = len(items) if isinstance(items, list) else 0
        noun = "result" if total == 1 else "results"
        return f"{label}: {total} {noun}", False, f"Showing {shown} of {total} {noun} in this page."
    return f"{label}: complete", False, "The request completed."


def _show_tool(name: str, result: dict[str, JsonValue]) -> None:
    label, error, detail = tool_summary(name, result)
    with st.status(label, state="error" if error else "complete", expanded=False):
        st.caption(detail)


def render_history(messages: list[ChatMessage]) -> None:
    for _turn_id, grouped in groupby(messages, key=lambda message: message.turn_id):
        turn = list(grouped)
        for message in turn:
            if message.role == "user":
                with st.chat_message("user"):
                    st.markdown(
                        "\n".join(
                            str(block["text"])
                            for block in message.content
                            if block.get("type") == "text" and isinstance(block.get("text"), str)
                        )
                    )
        if not any(message.role in {"assistant", "tool"} for message in turn):
            continue
        with st.chat_message("assistant"):
            names: dict[str, str] = {}
            for message in turn:
                for block in message.content:
                    if message.role == "assistant" and block.get("type") == "text":
                        st.markdown(str(block.get("text", "")))
                    elif block.get("type") == "tool_use":
                        names[str(block.get("id"))] = str(block.get("name", "tool"))
                    elif block.get("type") == "tool_result":
                        _show_tool(names.get(str(block.get("tool_use_id")), "tool"), block)


def stream_reply(client: ApiClient, session_id: UUID, prompt: str) -> tuple[bool, str | None]:
    statuses: dict[str, StatusContainer] = {}
    running: set[str] = set()
    names: dict[str, str] = {}
    slot: DeltaGenerator | None = None
    text = ""
    terminal = False
    failure = None
    try:
        for event in client.stream_chat(session_id, prompt):
            if event.type == "text_delta":
                if slot is None:
                    slot = st.empty()
                text += event.text or ""
                slot.markdown(text + " ▌")
            elif event.type == "tool_call_started" and event.tool_use:
                if slot is not None:
                    slot.markdown(text)
                    slot, text = None, ""
                call_id, name = str(event.tool_use["id"]), str(event.tool_use["name"])
                names[call_id] = name
                label = {
                    "query_projects": "Searching projects…",
                    "search_docs": "Searching documents…",
                }.get(name, f"Running {name.replace('_', ' ')}…")
                statuses[call_id] = st.status(label, expanded=False)
                running.add(call_id)
            elif event.type == "tool_call_finished" and event.tool_result:
                call_id = str(event.tool_result["tool_use_id"])
                label, error, detail = tool_summary(names.get(call_id, "tool"), event.tool_result)
                status = statuses.get(call_id)
                if status is None:
                    status = st.status(label, expanded=False)
                    statuses[call_id] = status
                status.update(label=label, state="error" if error else "complete", expanded=False)
                running.discard(call_id)
                with status:
                    st.caption(detail)
            elif event.type in {"turn_completed", "turn_failed"}:
                terminal = True
                if event.type == "turn_failed":
                    failure = event.message
    except ApiError as exc:
        failure = str(exc)
    finally:
        if slot is not None:
            slot.markdown(text)
    if failure:
        for call_id in running:
            statuses[call_id].update(label="Search interrupted", state="error", expanded=False)
    return terminal, failure


def _select(session_id: UUID | None) -> None:
    st.session_state.pop("turn_error", None)
    if session_id is None:
        st.query_params.pop("session", None)
    else:
        st.query_params["session"] = str(session_id)


def main() -> None:
    st.set_page_config(page_title="Project chat", page_icon="💬", initial_sidebar_state="expanded")
    st.title("Project chat")
    st.caption("Ask about company projects, clients, and timelines.")
    try:
        settings = FrontendSettings()
    except ValidationError:
        st.error("The chat interface is not configured correctly. Check its connection settings.")
        st.stop()
    with st.sidebar:
        st.header("Conversations")
        token = st.text_input("Access token", type="password", key="access_token")
        st.caption("Use your development access token to open your conversations.")
    fingerprint = hashlib.sha256(token.encode()).hexdigest() if token else None
    previous = st.session_state.get("auth_fingerprint")
    if previous is not None and previous != fingerprint:
        _select(None)
    st.session_state["auth_fingerprint"] = fingerprint
    if not token:
        st.info("Enter your access token in the sidebar to start chatting.")
        st.stop()

    with ApiClient(str(settings.backend_url), token) as client:
        try:
            sessions = client.list_sessions()
        except ApiError as exc:
            st.error(str(exc))
            st.stop()
        selected: UUID | None = None
        raw_id = st.query_params.get("session")
        if raw_id:
            try:
                selected = UUID(raw_id)
            except ValueError:
                st.warning(
                    "This conversation link is invalid. Start a new chat or choose one below."
                )
                _select(None)
        with st.sidebar:
            if st.button("New chat", key="new_chat", width="stretch"):
                _select(None)
                st.rerun()
            if st.button("Refresh conversations", key="refresh", width="stretch"):
                st.rerun()
            for session in sessions:
                title = session.title or f"Chat · {session.created_at:%b %d, %H:%M}"
                if st.button(
                    title,
                    key=f"session-{session.id}",
                    width="stretch",
                    type="primary" if session.id == selected else "secondary",
                ):
                    _select(session.id)
                    st.rerun()
        if selected is not None:
            try:
                detail = client.get_session(selected)
            except ApiError as exc:
                st.error(str(exc))
                if exc.status_code == 404:
                    _select(None)
                st.stop()
            render_history(detail.messages)
        else:
            st.info("Start a new conversation, or reopen one from the sidebar.")
        notice = st.session_state.get("turn_error")
        if notice and notice["session"] == str(selected):
            st.error(notice["message"])
        prompt = st.chat_input("Ask a question about your projects", max_chars=16_000)
        if isinstance(prompt, str) and prompt.strip():
            st.session_state.pop("turn_error", None)
            try:
                if selected is None:
                    created = client.create_session(" ".join(prompt.split())[:80])
                    selected = created.id
                    _select(selected)
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    terminal, failure = stream_reply(client, selected, prompt)
            except ApiError as exc:
                st.error(str(exc))
                st.stop()
            if failure:
                st.session_state["turn_error"] = {"session": str(selected), "message": failure}
            if terminal:
                st.rerun()
            if failure:
                st.error(failure)
                if st.button("Reload conversation", key="reload_after_failure"):
                    st.rerun()


if __name__ == "__main__":
    main()
