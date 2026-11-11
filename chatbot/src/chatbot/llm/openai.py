"""Translate OpenAI Responses streams to the provider-neutral LLM contract."""

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

from openai import APIConnectionError, APIError, APIStatusError, AsyncOpenAI, not_given, omit
from openai.types.responses import ResponseInputParam, ToolParam

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


def _input_items(messages: Sequence[Message]) -> ResponseInputParam:
    items: list[dict[str, object]] = []
    for message in messages:
        if message.role == "system":
            raise ProviderError("System text must be passed in request.system")
        if message.role == "tool":
            for block in message.content:
                if not isinstance(block, ToolResult):
                    raise ProviderError("Tool messages must contain tool results")
                output: object = {"error": block.content} if block.is_error else block.content
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": block.tool_use_id,
                        "output": output
                        if isinstance(output, str)
                        else json.dumps(output, ensure_ascii=False, separators=(",", ":")),
                    }
                )
            continue

        text_parts: list[str] = []

        for block in message.content:
            if isinstance(block, Text):
                text_parts.append(block.text)
            elif isinstance(block, ToolUse) and message.role == "assistant":
                if text_parts:
                    items.append({"role": message.role, "content": "".join(text_parts)})
                    text_parts.clear()
                items.append(
                    {
                        "type": "function_call",
                        "call_id": block.id,
                        "name": block.name,
                        "arguments": json.dumps(
                            block.arguments, ensure_ascii=False, separators=(",", ":")
                        ),
                    }
                )
            else:
                raise ProviderError("Message contains a block unsupported for its role")
        if text_parts:
            items.append({"role": message.role, "content": "".join(text_parts)})
    return cast(ResponseInputParam, items)


def _tool_params(tools: Sequence[ToolDefinition]) -> list[ToolParam]:
    return [
        cast(
            ToolParam,
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
                "strict": False,
            },
        )
        for tool in tools
    ]


class _StreamTranslator:
    def __init__(self) -> None:
        self._calls: set[str] = set()
        self._completed = False
        self._refused = False

    def feed(self, event: dict[str, Any]) -> list[LLMEvent]:
        kind = event.get("type")
        if self._completed:
            raise ProviderError("Unexpected event after response completed")
        if kind == "response.output_text.delta":
            return [LLMTextDelta(text=event["delta"])]
        if kind == "response.refusal.done":
            self._refused = True
            return []
        if kind == "response.output_item.done":
            item = event["item"]
            if item.get("type") != "function_call" or item.get("status") == "incomplete":
                return []
            call_id = item["call_id"]
            if call_id in self._calls:
                raise ProviderError("Duplicate function call")
            self._calls.add(call_id)
            try:
                arguments = json.loads(item["arguments"])
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments are not an object")
                call = ToolUse(id=call_id, name=item["name"], arguments=arguments)
            except (KeyError, TypeError, ValueError) as exc:
                raise ProviderError("Invalid streamed tool arguments") from exc
            return [LLMToolCall(tool_use=call)]
        if kind in {"response.completed", "response.incomplete"}:
            response = event["response"]
            if kind == "response.completed":
                reason: StopReason = (
                    "refusal" if self._refused else "tool_use" if self._calls else "end_turn"
                )
            else:
                detail = (response.get("incomplete_details") or {}).get("reason")
                reason = (
                    "max_tokens"
                    if detail == "max_output_tokens"
                    else "refusal"
                    if detail == "content_filter"
                    else "unknown"
                )
            raw_usage = response.get("usage") or {}
            input_details = raw_usage.get("input_tokens_details") or {}
            usage = Usage(
                input_tokens=raw_usage.get("input_tokens", 0),
                output_tokens=raw_usage.get("output_tokens", 0),
                cache_read_input_tokens=input_details.get("cached_tokens", 0),
                cache_creation_input_tokens=input_details.get("cache_write_tokens", 0),
            )
            self._completed = True
            return [LLMUsage(usage=usage), LLMStop(reason=reason)]
        if kind == "response.failed":
            raise ProviderError("OpenAI response failed")
        if kind == "error":
            raise ProviderError("OpenAI stream failed")
        return []

    def finish(self) -> None:
        if not self._completed:
            raise ProviderError("Incomplete response stream", retryable=True)


class OpenAIProvider:
    def __init__(
        self, model: str, settings: Settings, *, client: AsyncOpenAI | None = None
    ) -> None:
        self.model = model
        self.name = f"openai:{model}"
        self._owns_client = client is None
        if client is None:
            key = settings.openai_api_key.get_secret_value()
            if not key:
                raise ProviderError("OpenAI API key is not configured")
            client = AsyncOpenAI(
                api_key=key,
                timeout=settings.llm_timeout_seconds,
                max_retries=0,
            )
        self._client = client

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        items = _input_items(request.messages)
        tools = _tool_params(request.tools)
        translator = _StreamTranslator()
        try:
            raw_stream = await self._client.responses.create(
                model=self.model,
                instructions=request.system if request.system else omit,
                input=items,
                tools=tools if tools else omit,
                max_output_tokens=request.max_output_tokens,
                timeout=request.timeout_seconds
                if request.timeout_seconds is not None
                else not_given,
                store=False,
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
            raise ProviderError("OpenAI connection failed", retryable=True) from exc
        except APIStatusError as exc:
            retryable = exc.status_code == 429 or exc.status_code >= 500
            raise ProviderError("OpenAI request failed", retryable=retryable) from exc
        except APIError as exc:
            raise ProviderError("OpenAI request failed") from exc
