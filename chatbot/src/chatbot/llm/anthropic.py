"""Translate Anthropic Messages streams to the provider-neutral LLM contract."""

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

from anthropic import APIConnectionError, APIError, APIStatusError, AsyncAnthropic, not_given, omit
from anthropic.types import MessageParam, ToolParam

from chatbot.config import Settings
from chatbot.errors import ProviderError
from chatbot.llm.base import (
    LLMEvent,
    LLMRequest,
    LLMStop,
    LLMTextDelta,
    LLMToolCall,
    LLMUsage,
    StopReason,
    ToolDefinition,
)
from chatbot.types import Message, Text, ToolResult, ToolUse, Usage


def _message_params(messages: Sequence[Message]) -> list[MessageParam]:
    result: list[MessageParam] = []
    pending_results: list[dict[str, object]] = []

    def flush_results() -> None:
        if pending_results:
            # Parallel tool results belong in one user message, before any text.
            result.append(cast(MessageParam, {"role": "user", "content": pending_results.copy()}))
            pending_results.clear()

    for message in messages:
        if message.role == "system":
            raise ProviderError("System text must be passed in request.system")
        if message.role == "tool":
            for block in message.content:
                if not isinstance(block, ToolResult):
                    raise ProviderError("Tool messages must contain tool results")
                content = (
                    block.content
                    if isinstance(block.content, str)
                    else json.dumps(block.content, ensure_ascii=False, separators=(",", ":"))
                )
                pending_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.tool_use_id,
                        "content": content,
                        "is_error": block.is_error,
                    }
                )
            continue

        flush_results()
        blocks: list[dict[str, object]] = []
        for block in message.content:
            if isinstance(block, Text):
                blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolUse) and message.role == "assistant":
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.arguments,
                    }
                )
            else:
                raise ProviderError("Message contains a block unsupported for its role")
        result.append(cast(MessageParam, {"role": message.role, "content": blocks}))
    flush_results()
    return result


def _tool_params(tools: Sequence[ToolDefinition]) -> list[ToolParam]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": cast(dict[str, object], tool.input_schema),
        }
        for tool in tools
    ]


def _stop_reason(value: object) -> StopReason:
    if value == "end_turn":
        return "end_turn"
    if value == "tool_use":
        return "tool_use"
    if value == "max_tokens":
        return "max_tokens"
    if value == "stop_sequence":
        return "stop_sequence"
    if value == "pause_turn":
        return "pause_turn"
    if value == "refusal":
        return "refusal"
    if value == "model_context_window_exceeded":
        return "model_context_window_exceeded"
    return "unknown"


class _StreamTranslator:
    def __init__(self) -> None:
        self._usage: dict[str, int] | None = None
        self._stop_reason: StopReason = "unknown"
        self._tool_blocks: dict[int, dict[str, Any]] = {}
        self._stopped = False

    def feed(self, event: dict[str, Any]) -> list[LLMEvent]:
        kind = event.get("type")
        if self._stopped:
            raise ProviderError("Unexpected event after response completed")

        if kind == "message_start":
            if self._usage is not None:
                raise ProviderError("Duplicate response start")
            raw_usage = event["message"].get("usage", {})
            self._usage = {key: value for key, value in raw_usage.items() if isinstance(value, int)}
            return []

        if kind == "content_block_start":
            block = event["content_block"]
            if block.get("type") == "tool_use":
                self._tool_blocks[event["index"]] = {"block": block, "partials": []}
            elif block.get("type") == "text" and block.get("text"):
                return [LLMTextDelta(text=block["text"])]
            return []

        if kind == "content_block_delta":
            delta = event["delta"]
            if delta.get("type") == "text_delta":
                return [LLMTextDelta(text=delta["text"])]
            if delta.get("type") == "input_json_delta":
                state = self._tool_blocks.get(event["index"])
                if state is None:
                    raise ProviderError("Tool input arrived without a tool call")
                state["partials"].append(delta["partial_json"])
            return []

        if kind == "content_block_stop":
            state = self._tool_blocks.pop(event["index"], None)
            if state is None:
                return []
            block = state["block"]
            partials = state["partials"]
            try:
                arguments = json.loads("".join(partials)) if partials else block["input"]
                if not isinstance(arguments, dict):
                    raise ValueError("Tool input is not an object")
                tool_use = ToolUse(id=block["id"], name=block["name"], arguments=arguments)
            except (KeyError, ValueError, TypeError) as exc:
                raise ProviderError("Invalid streamed tool arguments") from exc
            return [LLMToolCall(tool_use=tool_use)]

        if kind == "message_delta":
            if self._usage is None:
                raise ProviderError("Usage arrived before response start")
            self._usage.update(
                {key: value for key, value in event["usage"].items() if isinstance(value, int)}
            )
            if event["delta"].get("stop_reason") is not None:
                self._stop_reason = _stop_reason(event["delta"]["stop_reason"])
            return []

        if kind == "message_stop":
            if self._usage is None:
                raise ProviderError("Response did not include usage")
            if self._tool_blocks:
                raise ProviderError("Response ended with an incomplete tool call")
            self._stopped = True
            cache_read = self._usage.get("cache_read_input_tokens", 0)
            cache_creation = self._usage.get("cache_creation_input_tokens", 0)
            usage = Usage(
                input_tokens=self._usage.get("input_tokens", 0) + cache_read + cache_creation,
                output_tokens=self._usage.get("output_tokens", 0),
                cache_read_input_tokens=cache_read,
                cache_creation_input_tokens=cache_creation,
            )
            return [LLMUsage(usage=usage), LLMStop(reason=self._stop_reason)]

        # Forward compatibility: pings, thinking and new event kinds are ignored.
        return []

    def finish(self) -> None:
        if not self._stopped:
            raise ProviderError("Incomplete response stream", retryable=True)


class AnthropicProvider:
    def __init__(
        self, model: str, settings: Settings, *, client: AsyncAnthropic | None = None
    ) -> None:
        self.model = model
        self.name = f"anthropic:{model}"
        self._owns_client = client is None
        if client is None:
            key = settings.anthropic_api_key.get_secret_value()
            if not key:
                raise ProviderError("Anthropic API key is not configured")
            client = AsyncAnthropic(
                api_key=key,
                timeout=settings.llm_timeout_seconds,
                max_retries=0,  # The gateway in C25 owns retries and fallbacks.
            )
        self._client = client

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        messages = _message_params(request.messages)
        tools = _tool_params(request.tools)
        translator = _StreamTranslator()
        try:
            raw_stream = await self._client.messages.create(
                model=self.model,
                max_tokens=request.max_output_tokens,
                messages=messages,
                tools=tools if tools else omit,
                system=request.system if request.system else omit,
                timeout=request.timeout_seconds
                if request.timeout_seconds is not None
                else not_given,
                stream=True,
            )
            async with raw_stream:
                async for event in raw_stream:
                    for normalized in translator.feed(
                        event.model_dump(mode="python", exclude_none=True)
                    ):
                        yield normalized
            translator.finish()
        except ProviderError:
            raise
        except APIConnectionError as exc:
            raise ProviderError("Anthropic connection failed", retryable=True) from exc
        except APIStatusError as exc:
            retryable = exc.status_code == 429 or exc.status_code >= 500
            raise ProviderError("Anthropic request failed", retryable=retryable) from exc
        except APIError as exc:
            raise ProviderError("Anthropic request failed") from exc
