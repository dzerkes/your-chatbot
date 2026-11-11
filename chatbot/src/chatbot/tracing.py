"""Application-owned OpenTelemetry tracing exported to Langfuse via OTLP/HTTP."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic
from typing import TYPE_CHECKING

from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.trace import INVALID_SPAN, Span, Status, StatusCode, format_trace_id

from chatbot.config import Settings
from chatbot.errors import ChatbotError, ToolError
from chatbot.llm.base import LLMEvent, LLMProvider, LLMRequest, LLMStop, LLMTextDelta, LLMUsage
from chatbot.types import ToolResult, ToolUse, Usage, User

if TYPE_CHECKING:
    from chatbot.store.base import Turn
    from chatbot.tools.base import ToolContext
    from chatbot.tools.registry import ToolRegistry

type Attributes = dict[str, str | int | float | bool | list[str]]


class TraceSpan:
    def __init__(self, span: Span) -> None:
        self.span = span
        self.failed = False

    def set(self, attributes: Attributes) -> None:
        if self.span.is_recording():
            self.span.set_attributes(attributes)

    def error(self, message: str, code: str) -> None:
        self.failed = True
        self.set(
            {
                "error.type": code,
                "langfuse.observation.level": "ERROR",
                "langfuse.observation.status_message": message,
            }
        )
        self.span.set_status(Status(StatusCode.ERROR, message))

    def usage(self, usage: Usage) -> None:
        if not self.span.is_recording():
            return
        # Flat Langfuse usage details must be exclusive: our input total includes caches.
        details = {
            "input": max(
                0,
                usage.input_tokens
                - usage.cache_read_input_tokens
                - usage.cache_creation_input_tokens,
            ),
            "output": usage.output_tokens,
            "input_cached_tokens": usage.cache_read_input_tokens,
            "input_cache_write_tokens": usage.cache_creation_input_tokens,
        }
        self.set(
            {
                "gen_ai.usage.input_tokens": usage.input_tokens,
                "gen_ai.usage.output_tokens": usage.output_tokens,
                "gen_ai.usage.cache_read.input_tokens": usage.cache_read_input_tokens,
                "gen_ai.usage.cache_creation.input_tokens": usage.cache_creation_input_tokens,
                "langfuse.observation.usage_details": json.dumps(details, separators=(",", ":")),
            }
        )


class Tracing:
    def __init__(
        self, provider: TracerProvider | None = None, *, environment: str = "local"
    ) -> None:
        self._provider = provider
        self._tracer = provider.get_tracer("chatbot", "0.1.0") if provider is not None else None
        self.environment = environment
        self._attributes: ContextVar[Attributes | None] = ContextVar(
            "chatbot_trace_attributes", default=None
        )

    @property
    def enabled(self) -> bool:
        return self._provider is not None

    @classmethod
    def from_settings(cls, settings: Settings) -> Tracing:
        if not settings.tracing_enabled:
            return cls()
        public = settings.langfuse_public_key.get_secret_value()
        secret = settings.langfuse_secret_key.get_secret_value()
        if not public or not secret:
            raise ChatbotError("Tracing requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
        auth = base64.b64encode(f"{public}:{secret}".encode()).decode()
        exporter = OTLPSpanExporter(
            endpoint=str(settings.langfuse_host).rstrip("/") + "/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {auth}", "x-langfuse-ingestion-version": "4"},
            timeout=5,
        )
        provider = TracerProvider(
            resource=Resource(
                {"service.name": "your-chatbot", "deployment.environment.name": settings.app_env}
            ),
            sampler=ALWAYS_ON,
            shutdown_on_exit=False,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        return cls(provider, environment=settings.app_env)

    async def aclose(self) -> None:
        if self._provider is not None:
            await asyncio.to_thread(self._provider.shutdown)

    @contextmanager
    def span(
        self, name: str, attributes: Attributes | None = None, *, root: bool = False
    ) -> Iterator[TraceSpan]:
        if self._tracer is None:
            yield TraceSpan(INVALID_SPAN)
            return
        started = monotonic()
        with self._tracer.start_as_current_span(
            name,
            context=Context() if root else None,
            attributes={**(self._attributes.get() or {}), **(attributes or {})},
            record_exception=False,
            set_status_on_exception=False,
        ) as raw:
            span = TraceSpan(raw)
            try:
                yield span
            except (asyncio.CancelledError, GeneratorExit):
                span.error("Operation interrupted", "cancelled")
                raise
            except ChatbotError as exc:
                span.error(str(exc), exc.code)
                raise
            except Exception:
                span.error("Operation failed", "internal_error")
                raise
            finally:
                span.set({"chatbot.latency_ms": (monotonic() - started) * 1000})

    @contextmanager
    def turn(self, turn: Turn, user: User) -> Iterator[TraceSpan]:
        if not self.enabled:
            yield TraceSpan(INVALID_SPAN)
            return
        token = self._attributes.set(
            {
                "session.id": str(turn.session_id),
                "user.id": str(user.id),
                "langfuse.session.id": str(turn.session_id),
                "langfuse.user.id": str(user.id),
                "chatbot.turn.id": str(turn.id),
                "langfuse.trace.name": "chat",
                "langfuse.environment": self.environment,
            }
        )
        try:
            # One trace per turn; Langfuse groups those traces by the stable session ID.
            with (
                self.span("session", {"langfuse.observation.type": "span"}, root=True) as session,
                self.span("turn", {"langfuse.observation.type": "agent"}) as active,
            ):
                turn.trace_id = format_trace_id(active.span.get_span_context().trace_id)
                try:
                    yield active
                finally:
                    if active.failed:
                        session.error("Turn failed", "turn_failed")
        finally:
            self._attributes.reset(token)

    def wrap_provider(self, provider: LLMProvider) -> LLMProvider:
        return TracedProvider(provider, self) if self.enabled else provider

    async def execute_tool(
        self,
        registry: ToolRegistry,
        call: ToolUse,
        context: ToolContext,
    ) -> ToolResult:
        if not self.enabled:
            return await registry.execute(call, context)
        attributes: Attributes = {
            "langfuse.observation.type": "tool",
            "gen_ai.tool.name": call.name,
            "gen_ai.tool.call.id": call.id,
            "langfuse.observation.input": json.dumps(call.arguments, ensure_ascii=False),
        }
        with self.span(f"tool:{call.name}", attributes) as span:
            try:
                result = await registry.execute(call, context)
            except ToolError:
                span.set({"chatbot.tool.result_bytes": 0, "chatbot.tool.is_error": True})
                raise
            size = len(
                json.dumps(result.content, ensure_ascii=False, separators=(",", ":")).encode()
            )
            span.set({"chatbot.tool.result_bytes": size, "chatbot.tool.is_error": result.is_error})
            if result.is_error:
                span.error("Tool returned an error result", "tool_error")
            return result


class TracedProvider:
    def __init__(self, provider: LLMProvider, tracing: Tracing) -> None:
        self._provider = provider
        self._tracing = tracing
        self.name = provider.name

    async def aclose(self) -> None:
        await self._provider.aclose()

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        provider, _, model = self.name.partition(":")
        attributes: Attributes = {
            "langfuse.observation.type": "generation",
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": provider,
            "gen_ai.request.model": model or self.name,
            "langfuse.observation.model.name": model or self.name,
            "gen_ai.request.max_tokens": request.max_output_tokens,
            "chatbot.cost_usd": 0.0,
            "chatbot.cost_calculated": False,
        }
        with self._tracing.span("llm", attributes) as span:
            stream = self._provider.stream(request)
            started = monotonic()
            first_text = True
            try:
                async for event in stream:
                    if isinstance(event, LLMUsage):
                        span.usage(event.usage)
                    elif isinstance(event, LLMStop):
                        span.set({"gen_ai.response.finish_reasons": [event.reason]})
                    elif isinstance(event, LLMTextDelta) and event.text and first_text:
                        span.set({"chatbot.time_to_first_token_ms": (monotonic() - started) * 1000})
                        first_text = False
                    yield event
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()
