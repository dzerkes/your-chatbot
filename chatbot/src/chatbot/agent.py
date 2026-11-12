"""Provider-neutral, persisted conversation loop."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import aclosing
from datetime import UTC, datetime
from time import monotonic
from typing import Literal, TypedDict
from uuid import UUID, uuid4
from weakref import WeakValueDictionary

from anyio import CancelScope

from chatbot.config import Settings
from chatbot.context import build_context_builder
from chatbot.context.base import ContextBuilder
from chatbot.errors import (
    ChatbotError,
    ContextOverflow,
    NotFoundError,
    ProviderError,
    StepLimitExceeded,
    StoreError,
    ToolError,
)
from chatbot.llm.base import (
    LLMProvider,
    LLMRequest,
    LLMStop,
    LLMTextDelta,
    LLMToolCall,
    LLMUsage,
)
from chatbot.logging import log_context
from chatbot.store.base import HistoryStore, Turn
from chatbot.tools.base import ToolContext
from chatbot.tools.registry import ToolRegistry
from chatbot.tracing import Tracing
from chatbot.types import (
    ContentBlock,
    Event,
    Message,
    Text,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    ToolResult,
    ToolUse,
    TurnCompleted,
    TurnFailed,
    Usage,
    User,
)

logger = logging.getLogger(__name__)


class _EventIDs(TypedDict):
    session_id: UUID
    turn_id: UUID
    trace_id: str | None


class Agent:
    def __init__(
        self,
        store: HistoryStore,
        provider: LLMProvider,
        tools: ToolRegistry,
        settings: Settings,
        *,
        system: str,
        context_builder: ContextBuilder | None = None,
        tracing: Tracing | None = None,
    ) -> None:
        self.store = store
        self.tracing = tracing if tracing is not None else Tracing()
        self.provider = self.tracing.wrap_provider(provider)
        self.tools = tools
        self.settings = settings
        self.system = system
        self.context_builder = (
            context_builder if context_builder is not None else build_context_builder(settings)
        )
        self._locks: WeakValueDictionary[UUID, asyncio.Lock] = WeakValueDictionary()

    async def respond(
        self, session_id: UUID, user_message: str, user: User
    ) -> AsyncGenerator[Event, None]:
        """Stream one turn. Consume or close the iterator to finish persistence/cleanup."""
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            turn = Turn(session_id=session_id, trace_id=uuid4().hex)
            with (
                self.tracing.turn(turn, user) as span,
                log_context(session_id=session_id, turn_id=turn.id, trace_id=turn.trace_id),
            ):
                async with aclosing(self._respond(turn, user_message, user)) as events:
                    async for event in events:
                        if isinstance(event, TurnFailed):
                            span.set({"chatbot.turn.status": "failed"})
                            span.error(event.message, event.code)
                        elif isinstance(event, TurnCompleted):
                            span.set(
                                {
                                    "chatbot.turn.status": "completed",
                                    "chatbot.usage.input_tokens": event.usage.input_tokens,
                                    "chatbot.usage.output_tokens": event.usage.output_tokens,
                                    "chatbot.cost_usd": float(event.cost_usd),
                                }
                            )
                        yield event

    async def _respond(self, turn: Turn, text: str, user: User) -> AsyncGenerator[Event, None]:
        started = monotonic()
        ids: _EventIDs = {
            "session_id": turn.session_id,
            "turn_id": turn.id,
            "trace_id": turn.trace_id,
        }
        try:
            if not text.strip():
                raise ChatbotError("Message must not be blank")
            if await self.store.get_session(turn.session_id, user.id) is None:
                raise NotFoundError("Session not found")
            history = await self.store.list_messages(turn.session_id, user.id)
        except ChatbotError as exc:
            yield TurnFailed(**ids, code=exc.code, message=str(exc), retryable=exc.retryable)
            return
        except Exception:
            logger.exception("Could not load conversation")
            yield TurnFailed(**ids, code=StoreError.code, message="Could not load conversation")
            return

        next_sequence = max((message.sequence for message in history), default=-1) + 1
        messages: list[Message] = []
        usage = Usage()
        pending: list[ContentBlock] = []
        step_usage: Usage | None = None

        def append(
            role: Literal["user", "assistant", "tool"],
            content: list[ContentBlock],
            *,
            model: str | None = None,
        ) -> None:
            # All messages remain in memory until save_turn commits one atomic batch.
            message = Message(
                session_id=turn.session_id,
                turn_id=turn.id,
                sequence=next_sequence + len(messages),
                role=role,
                content=content,
                model=model,
                usage=step_usage if role == "assistant" else None,
            )
            messages.append(message)

        append("user", [Text(text=text)])
        context = ToolContext(
            user=user, session_id=turn.session_id, turn_id=turn.id, trace_id=turn.trace_id
        )
        used_calls = {
            block.id
            for message in history
            for block in message.content
            if isinstance(block, ToolUse)
        }
        failure: ChatbotError | None = None
        try:
            for _step in range(self.settings.agent_max_steps):
                pending = []
                step_usage = None
                stop: LLMStop | None = None
                definitions = self.tools.definitions()
                built = self.context_builder.build(
                    history, messages, system=self.system, tools=definitions
                )
                request = LLMRequest(
                    system=built.system,
                    messages=built.messages,
                    tools=definitions,
                    max_output_tokens=self.settings.llm_max_output_tokens,
                    timeout_seconds=self.settings.llm_timeout_seconds,
                )
                stream = self.provider.stream(request)
                try:
                    async for event in stream:
                        if stop is not None:
                            raise ProviderError("Provider sent events after its stop reason")
                        if isinstance(event, LLMTextDelta):
                            if not event.text:
                                continue
                            if pending and isinstance(pending[-1], Text):
                                pending[-1].text += event.text
                            else:
                                pending.append(Text(text=event.text))
                            yield TextDelta(**ids, text=event.text)
                        elif isinstance(event, LLMToolCall):
                            if event.tool_use.id in used_calls:
                                raise ProviderError("Provider reused a tool call ID")
                            used_calls.add(event.tool_use.id)
                            pending.append(event.tool_use)
                        elif isinstance(event, LLMUsage):
                            if step_usage is not None:
                                raise ProviderError("Provider sent duplicate usage")
                            step_usage = event.usage
                            usage = Usage(
                                **{
                                    key: getattr(usage, key) + value
                                    for key, value in event.usage.model_dump().items()
                                }
                            )
                        elif isinstance(event, LLMStop):
                            stop = event
                finally:
                    close = getattr(stream, "aclose", None)
                    if close is not None:
                        await close()
                if stop is None or step_usage is None:
                    raise ProviderError("Provider returned an incomplete response", retryable=True)
                calls = [block for block in pending if isinstance(block, ToolUse)]
                if bool(calls) != (stop.reason == "tool_use"):
                    raise ProviderError("Provider tool calls do not match its stop reason")
                if stop.reason == "max_tokens":
                    raise ProviderError("Model output reached the token limit")
                if stop.reason == "model_context_window_exceeded":
                    raise ContextOverflow("Model context limit exceeded")
                if stop.reason not in {"end_turn", "stop_sequence", "refusal", "tool_use"}:
                    raise ProviderError("Provider returned an unsupported stop reason")
                if not pending:
                    if stop.reason != "refusal":
                        raise ProviderError("Provider returned an empty answer")
                    reply = "The model declined to answer this request."
                    pending.append(Text(text=reply))
                    yield TextDelta(**ids, text=reply)
                append("assistant", pending, model=self.provider.name)
                pending = []
                if not calls:
                    break
                for call in calls:
                    yield ToolCallStarted(**ids, tool_use=call)
                    try:
                        result = await self.tracing.execute_tool(self.tools, call, context)
                    except ToolError as exc:
                        result = ToolResult(
                            tool_use_id=call.id,
                            is_error=True,
                            content={
                                "error": str(exc),
                                "code": exc.code,
                                "retryable": exc.retryable,
                            },
                        )
                    append("tool", [result])
                    yield ToolCallFinished(**ids, tool_result=result)
            else:
                raise StepLimitExceeded("The agent reached its step limit")
        except (asyncio.CancelledError, GeneratorExit):
            self._finish_messages(
                messages, pending, step_usage, turn, next_sequence, "Turn was interrupted"
            )
            try:
                with CancelScope(shield=True):
                    await self._save(turn, messages, user, usage, started, failed=True)
            except Exception:
                logger.exception("Could not save interrupted conversation")
            raise
        except ChatbotError as exc:
            failure = exc
        except Exception:
            logger.exception("Agent turn failed")
            failure = ChatbotError("Could not complete the response")

        if failure is not None:
            self._finish_messages(messages, pending, step_usage, turn, next_sequence, str(failure))
        try:
            await self._save(turn, messages, user, usage, started, failed=failure is not None)
        except Exception:
            logger.exception("Could not save conversation")
            yield TurnFailed(**ids, code=StoreError.code, message="Could not save conversation")
            return
        if failure is not None:
            yield TurnFailed(
                **ids, code=failure.code, message=str(failure), retryable=failure.retryable
            )
        else:
            yield TurnCompleted(**ids, usage=usage, cost_usd=turn.cost_usd)

    def _finish_messages(
        self,
        messages: list[Message],
        pending: list[ContentBlock],
        step_usage: Usage | None,
        turn: Turn,
        next_sequence: int,
        reason: str,
    ) -> None:
        partial: list[ContentBlock] = [block for block in pending if isinstance(block, Text)]
        if partial:
            messages.append(
                Message(
                    session_id=turn.session_id,
                    turn_id=turn.id,
                    sequence=next_sequence + len(messages),
                    role="assistant",
                    content=partial,
                    model=self.provider.name,
                    usage=step_usage,
                )
            )
        completed = {
            block.tool_use_id
            for message in messages
            for block in message.content
            if isinstance(block, ToolResult)
        }
        for call in [
            block
            for message in messages
            for block in message.content
            if isinstance(block, ToolUse) and block.id not in completed
        ]:
            messages.append(
                Message(
                    session_id=turn.session_id,
                    turn_id=turn.id,
                    sequence=next_sequence + len(messages),
                    role="tool",
                    content=[
                        ToolResult(tool_use_id=call.id, is_error=True, content={"error": reason})
                    ],
                )
            )

    async def _save(
        self,
        turn: Turn,
        messages: list[Message],
        user: User,
        usage: Usage,
        started: float,
        *,
        failed: bool,
    ) -> None:
        turn.status = "failed" if failed else "completed"
        turn.completed_at = datetime.now(UTC)
        turn.updated_at = turn.completed_at
        turn.latency_ms = int((monotonic() - started) * 1000)
        for key, value in usage.model_dump().items():
            setattr(turn, key, value)
        await self.store.save_turn(turn, messages, user.id)
