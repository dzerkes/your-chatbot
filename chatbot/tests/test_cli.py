"""Terminal rendering and exit behavior with the scripted provider."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from chatbot.agent import Agent
from chatbot.config import Settings
from chatbot.llm.base import LLMStop, LLMTextDelta, LLMUsage
from chatbot.llm.fake import ScriptedProvider
from chatbot.store.memory import MemoryStore
from chatbot.tools.registry import ToolRegistry
from chatbot.types import Usage

from chatbot import cli

pytestmark = pytest.mark.unit


def test_one_shot_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    store = MemoryStore()
    provider = ScriptedProvider(
        [
            [
                LLMTextDelta(text="Hello from the CLI"),
                LLMUsage(usage=Usage()),
                LLMStop(reason="end_turn"),
            ]
        ]
    )

    @asynccontextmanager
    async def build(settings: Settings) -> AsyncIterator[Agent]:
        yield Agent(store, provider, ToolRegistry(), settings, system="Test")

    monkeypatch.setattr(cli, "build_chatbot", build)
    monkeypatch.setattr(cli, "configure_logging", lambda settings: None)
    assert cli.main(["chat", "--user", "alice", "--prompt", "Hello"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "Hello from the CLI\n"
    assert "Session:" in captured.err
    assert provider.requests[0].messages[0].content[0].model_dump()["text"] == "Hello"


@pytest.mark.parametrize("ending", ["/quit", "eof", "interrupt"])
def test_interactive_exit_closes_application(
    ending: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []

    @asynccontextmanager
    async def build(settings: Settings) -> AsyncIterator[Agent]:
        try:
            yield Agent(
                MemoryStore(), ScriptedProvider([]), ToolRegistry(), settings, system="Test"
            )
        finally:
            closed.append(True)

    def read(prompt: str) -> str:
        if ending == "eof":
            raise EOFError
        if ending == "interrupt":
            raise KeyboardInterrupt
        return ending

    monkeypatch.setattr(cli, "build_chatbot", build)
    monkeypatch.setattr(cli, "configure_logging", lambda settings: None)
    monkeypatch.setattr("builtins.input", read)
    assert cli.main(["chat"]) == (130 if ending == "interrupt" else 0)
    assert closed == [True]
