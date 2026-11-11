"""Scripted provider for deterministic tests without API calls."""

from collections.abc import AsyncIterator, Sequence

from chatbot.errors import ProviderError
from chatbot.llm.base import LLMEvent, LLMRequest


class ScriptedProvider:
    name = "fake:scripted"

    def __init__(self, scripts: Sequence[Sequence[LLMEvent | Exception]]) -> None:
        self._scripts = [tuple(script) for script in scripts]
        self.requests: list[LLMRequest] = []

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        if not self._scripts:
            raise ProviderError("No scripted response remains")
        self.requests.append(request.model_copy(deep=True))
        for event in self._scripts.pop(0):
            if isinstance(event, Exception):
                raise event
            yield event.model_copy(deep=True)
