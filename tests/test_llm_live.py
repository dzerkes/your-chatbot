"""Opt-in smoke test against a real Anthropic stream."""

from uuid import uuid4

import pytest
from chatbot.config import Settings
from chatbot.llm.anthropic import AnthropicProvider
from chatbot.llm.base import LLMRequest, LLMStop, LLMTextDelta, LLMUsage
from chatbot.types import Message, Text


@pytest.mark.llm
async def test_anthropic_real_stream() -> None:
    settings = Settings()
    if not settings.anthropic_api_key.get_secret_value():
        pytest.skip("ANTHROPIC_API_KEY is not configured")
    provider = AnthropicProvider(settings.llm_primary.removeprefix("anthropic:"), settings)
    try:
        request = LLMRequest(
            messages=[
                Message(
                    session_id=uuid4(),
                    turn_id=uuid4(),
                    sequence=0,
                    role="user",
                    content=[Text(text="Reply with one short greeting.")],
                )
            ],
            max_output_tokens=32,
        )
        events = [event async for event in provider.stream(request)]
        assert any(isinstance(event, LLMTextDelta) and event.text for event in events)
        assert isinstance(events[-2], LLMUsage)
        assert isinstance(events[-1], LLMStop)
        assert events[-1].reason == "end_turn"
    finally:
        await provider.aclose()
