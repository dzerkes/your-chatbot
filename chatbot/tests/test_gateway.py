"""LLM gateway fault injection, pricing, and per-turn cost."""

import json
from collections.abc import AsyncIterator, Sequence
from decimal import Decimal
from pathlib import Path

import pytest
from chatbot.agent import Agent
from chatbot.config import Settings
from chatbot.errors import ChatbotError, ProviderError
from chatbot.llm.base import (
    LLMEvent,
    LLMRequest,
    LLMStop,
    LLMTextDelta,
    LLMToolCall,
    LLMUsage,
)
from chatbot.llm.fake import ScriptedProvider
from chatbot.llm.gateway import LLMGateway
from chatbot.llm.pricing import ModelPrice, PriceTable
from chatbot.llm.registry import ProviderRegistry
from chatbot.store.memory import MemoryStore
from chatbot.tools.registry import ToolRegistry
from chatbot.tracing import Tracing
from chatbot.types import Message, Text, ToolUse, TurnCompleted, TurnFailed, Usage, User
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

pytestmark = pytest.mark.unit
PRICES = PriceTable(
    {
        "a:primary": ModelPrice(
            input=Decimal("2"), cached_input=Decimal("0.2"), output=Decimal("10")
        ),
        "b:fallback": ModelPrice(
            input=Decimal("1"), cached_input=Decimal("0.1"), output=Decimal("5")
        ),
    }
)
USAGE = Usage(input_tokens=1_000_000, output_tokens=100_000)


def ok(text: str = "Hello") -> list[LLMEvent | Exception]:
    return [LLMTextDelta(text=text), LLMUsage(usage=USAGE), LLMStop(reason="end_turn")]


def fail(retryable: bool) -> list[LLMEvent | Exception]:
    return [ProviderError("Provider is busy" if retryable else "Bad request", retryable=retryable)]


class FaultyProvider(ScriptedProvider):
    """Named scripted adapter that tracks resource ownership."""

    def __init__(self, name: str, scripts: Sequence[list[LLMEvent | Exception]]) -> None:
        super().__init__(scripts)
        self.name = name
        self.closed = False
        self.streams_closed = 0

    @property
    def calls(self) -> int:
        return len(self.requests)

    async def aclose(self) -> None:
        self.closed = True

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        try:
            async for event in super().stream(request):
                yield event
        finally:
            self.streams_closed += 1


def request() -> LLMRequest:
    from uuid import uuid4

    session = uuid4()
    return LLMRequest(
        messages=[
            Message(
                session_id=session,
                turn_id=uuid4(),
                sequence=0,
                role="user",
                content=[Text(text="Hi")],
            )
        ]
    )


def gateway(
    primary: FaultyProvider, *fallbacks: FaultyProvider, retries: int = 2
) -> tuple[LLMGateway, list[float]]:
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    return (
        LLMGateway(
            primary,
            fallbacks,
            prices=PRICES,
            max_retries=retries,
            base_delay=0.5,
            max_delay=1.5,
            sleep=sleep,
            jitter=lambda limit: limit,  # deterministic: the full backoff
        ),
        sleeps,
    )


async def run(gw: LLMGateway) -> list[LLMEvent]:
    return [event async for event in gw.stream(request())]


async def test_retries_with_backoff_then_succeeds_on_the_same_model() -> None:
    primary = FaultyProvider("a:primary", [fail(True), fail(True), ok()])
    backup = FaultyProvider("b:fallback", [ok()])
    gw, sleeps = gateway(primary, backup)
    events = await run(gw)
    assert (primary.calls, backup.calls, sleeps) == (3, 0, [0.5, 1.0])
    usage = next(e for e in events if isinstance(e, LLMUsage))
    assert usage.model == "a:primary"
    assert usage.cost_usd == Decimal("3.000000")  # 1M in * $2 + 100k out * $10


async def test_falls_back_when_retries_are_exhausted() -> None:
    primary = FaultyProvider("a:primary", [fail(True)] * 3)
    backup = FaultyProvider("b:fallback", [ok("From the fallback")])
    gw, sleeps = gateway(primary, backup)
    events = await run(gw)
    assert (primary.calls, backup.calls, sleeps) == (3, 1, [0.5, 1.0])
    assert [e.text for e in events if isinstance(e, LLMTextDelta)] == ["From the fallback"]
    usage = next(e for e in events if isinstance(e, LLMUsage))
    assert (usage.model, usage.cost_usd) == ("b:fallback", Decimal("1.500000"))


async def test_non_retryable_errors_fall_back_without_retrying() -> None:
    primary = FaultyProvider("a:primary", [fail(False)])
    backup = FaultyProvider("b:fallback", [ok()])
    gw, sleeps = gateway(primary, backup)
    await run(gw)
    assert (primary.calls, backup.calls, sleeps) == (1, 1, [])


async def test_unpriced_fallback_keeps_usage_and_has_unknown_cost() -> None:
    gw, _ = gateway(FaultyProvider("a:primary", [fail(False)]), FaultyProvider("b:new", [ok()]))
    usage = next(event for event in await run(gw) if isinstance(event, LLMUsage))
    assert (usage.model, usage.usage, usage.cost_usd) == ("b:new", USAGE, None)


async def test_empty_delta_allows_retry_and_each_fallback_gets_the_same_request() -> None:
    empty_failure = [LLMTextDelta(text=""), *fail(True)]
    primary = FaultyProvider("a:primary", [empty_failure, fail(True)])
    first_backup = FaultyProvider("b:fallback", [fail(True), fail(True)])
    last_backup = FaultyProvider("c:last", [ok()])
    gw, sleeps = gateway(primary, first_backup, last_backup, retries=1)
    req = request()
    events = [event async for event in gw.stream(req)]
    assert primary.requests == first_backup.requests == [req, req]
    assert last_backup.requests == [req]
    assert sleeps == [0.5, 0.5]
    assert [e.text for e in events if isinstance(e, LLMTextDelta)] == ["Hello"]


@pytest.mark.parametrize("empty", [[], [LLMTextDelta(text="")]])
async def test_empty_eof_retries_then_falls_back(empty: list[LLMEvent | Exception]) -> None:
    primary = FaultyProvider("a:primary", [empty, empty])
    backup = FaultyProvider("b:fallback", [ok()])
    gw, sleeps = gateway(primary, backup, retries=1)
    events = await run(gw)
    assert (primary.calls, backup.calls, sleeps) == (2, 1, [0.5])
    assert [e.text for e in events if isinstance(e, LLMTextDelta)] == ["Hello"]


async def test_consumer_closing_early_releases_the_active_provider_stream() -> None:
    primary = FaultyProvider("a:primary", [fail(False)])
    backup = FaultyProvider("b:fallback", [ok()])
    gw, sleeps = gateway(primary, backup)
    stream = gw.stream(request())
    assert (await anext(stream)).type == "text_delta"
    await stream.aclose()
    assert (primary.streams_closed, backup.streams_closed) == (1, 1)
    assert sleeps == []


async def test_final_usage_before_a_disconnect_is_charged_once_without_replay() -> None:
    primary = FaultyProvider("a:primary", [[LLMUsage(usage=USAGE), *fail(True)]])
    backup = FaultyProvider("b:fallback", [ok()])
    gw, sleeps = gateway(primary, backup)
    events: list[LLMEvent] = []
    with pytest.raises(ProviderError):
        async for event in gw.stream(request()):
            events.append(event)
    assert events == [LLMUsage(usage=USAGE, model=primary.name, cost_usd=Decimal("3"))]
    assert (primary.calls, backup.calls, sleeps) == (1, 0, [])


@pytest.mark.parametrize(
    "started",
    [
        [LLMTextDelta(text="Partial answer")],
        [LLMToolCall(tool_use=ToolUse(id="t1", name="lookup", arguments={}))],
    ],
)
async def test_never_retries_or_falls_back_after_output_started(started: list[LLMEvent]) -> None:
    primary = FaultyProvider(
        "a:primary", [[*started, ProviderError("Stream broke", retryable=True)]]
    )
    backup = FaultyProvider("b:fallback", [ok()])
    gw, sleeps = gateway(primary, backup)
    received: list[LLMEvent] = []
    with pytest.raises(ProviderError, match="Stream broke"):
        async for event in gw.stream(request()):
            received.append(event)
    assert received == [event.model_copy(update={"model": primary.name}) for event in started]
    assert (primary.calls, backup.calls, sleeps) == (1, 0, [])


async def test_raises_the_last_error_when_every_model_fails() -> None:
    primary = FaultyProvider("a:primary", [fail(True)] * 3)
    backup = FaultyProvider("b:fallback", [fail(False)])
    gw, _ = gateway(primary, backup)
    with pytest.raises(ProviderError, match="Bad request") as caught:
        await run(gw)
    assert not caught.value.retryable
    await gw.aclose()
    assert primary.closed and backup.closed


def test_backoff_is_capped_and_jittered() -> None:
    gw, _ = gateway(FaultyProvider("a:primary", []))
    assert [gw.delay(n) for n in range(4)] == [0.5, 1.0, 1.5, 1.5]
    real = LLMGateway(FaultyProvider("a:primary", []), prices=PRICES, base_delay=0.5, max_delay=8)
    assert all(0 <= real.delay(3) <= 4 for _ in range(50))


def test_unpriced_models_are_reported_at_startup(caplog: pytest.LogCaptureFixture) -> None:
    LLMGateway(FaultyProvider("x:unknown", []), prices=PRICES)
    assert "No price for x:unknown" in caplog.text


# --- pricing ----------------------------------------------------------------


def test_cost_splits_cached_and_cache_write_tokens() -> None:
    price = ModelPrice(
        input=Decimal("2.00"),
        cached_input=Decimal("0.10"),
        cache_write=Decimal("2.50"),
        output=Decimal("10.00"),
    )
    usage = Usage(
        input_tokens=10_000,  # includes the cached and cache-write tokens
        cache_read_input_tokens=6_000,
        cache_creation_input_tokens=1_000,
        output_tokens=500,
    )
    # 3,000 uncached * $2 + 6,000 read * $0.10 + 1,000 write * $2.50 + 500 out * $10, per 1M
    assert price.cost(usage) == Decimal("0.014100")
    no_write_price = price.model_copy(update={"cache_write": None})
    assert no_write_price.cost(usage) == Decimal("0.013600")  # writes billed as input


@pytest.mark.parametrize(
    ("input_tokens", "cost"),
    [(272_000, Decimal("0.009500")), (272_001, Decimal("0.0189752"))],
)
def test_long_context_threshold_prices_the_whole_call(input_tokens: int, cost: Decimal) -> None:
    usage = Usage(
        input_tokens=input_tokens,
        cache_read_input_tokens=200_000,
        cache_creation_input_tokens=10_000,
        output_tokens=100,
    )
    assert PriceTable.load().cost("openai:gpt-6-luna", usage) == cost


def test_numeric_override_preserves_sub_microdollar_cost(tmp_path: Path) -> None:
    override = tmp_path / "rates.toml"
    override.write_text(
        '["test:small"]\ninput = 0.1234567890123456789\ncached_input = 0\noutput = 0\n'
    )
    table = PriceTable.load(str(override))
    assert table.cost("test:small", Usage(input_tokens=1)) == Decimal("0.0000001234567890123456789")


def test_missing_price_file_reports_the_configured_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.toml"
    with pytest.raises(ChatbotError, match="Could not read model prices") as caught:
        PriceTable.load(str(missing))
    assert str(missing) in str(caught.value)


def test_shipped_price_table_covers_the_default_and_configured_models(tmp_path: Path) -> None:
    table = PriceTable.load()
    settings = Settings()
    assert not table.missing([settings.llm_primary, settings.llm_summary_model])
    assert not table.missing(["openai:gpt-6-luna", "openai:gpt-5.6-luna"])
    luna = table.cost(
        "openai:gpt-6-luna",
        Usage(input_tokens=2_000, cache_read_input_tokens=1_000, output_tokens=100),
    )
    assert luna == Decimal("0.000160")  # 1k * $0.10 + 1k * $0.01 + 100 * $0.50, per 1M
    override = tmp_path / "prices.toml"
    override.write_text(
        '["openai:gpt-6-luna"]\ninput = "1"\ncached_input = "1"\noutput = "1"\n\n'
        '["x:new"]\ninput = "0"\ncached_input = "0"\noutput = "0"\n'
    )

    merged = PriceTable.load(str(override))
    assert merged.prices["openai:gpt-6-luna"].input == Decimal("1")
    assert "x:new" in merged.prices and "anthropic:claude-sonnet-5-5" in merged.prices
    override.write_text('["x:bad"]\ninput = "-1"\ncached_input = "0"\noutput = "0"\n')
    with pytest.raises(ChatbotError, match="Invalid model prices"):
        PriceTable.load(str(override))


async def test_registry_builds_a_gateway_with_the_fallback_chain() -> None:
    settings = Settings(
        llm_primary="openai:gpt-6-luna",
        llm_fallbacks=("openai:gpt-5.6-luna", "openai:gpt-6-luna", "openai:gpt-5.6-luna"),
        openai_api_key=SecretStr("k"),
        llm_max_retries=4,
    )
    gw = ProviderRegistry(settings).primary_gateway()
    try:
        assert [p.name for p in gw.providers] == ["openai:gpt-6-luna", "openai:gpt-5.6-luna"]
        assert (gw.name, gw.max_retries) == ("openai:gpt-6-luna", 4)
    finally:
        await gw.aclose()


# --- per-turn cost ----------------------------------------------------------


async def test_each_turn_has_a_cost_on_the_turn_the_message_and_the_span() -> None:
    exporter = InMemorySpanExporter()
    tracer = TracerProvider(resource=Resource({"service.name": "test"}), shutdown_on_exit=False)
    tracer.add_span_processor(SimpleSpanProcessor(exporter))
    tracing = Tracing(tracer)
    primary = FaultyProvider("a:primary", [fail(False)])
    backup = FaultyProvider("b:fallback", [ok("Fine")])
    gw, _ = gateway(primary, backup)
    store = MemoryStore()
    agent = Agent(store, gw, ToolRegistry(), Settings(), system="Rules", tracing=tracing)
    user = await store.upsert_user(User(external_id="alice", display_name="Alice"))
    session = await store.create_session(user.id)
    events = [e async for e in agent.respond(session.id, "Hi", user)]
    done = events[-1]
    assert isinstance(done, TurnCompleted) and done.cost_usd == Decimal("1.500000")
    saved = await store.list_messages(session.id, user.id)
    assert saved[-1].model == "b:fallback"  # the model that actually answered
    turns = await store.list_turns(session.id, user.id)
    assert turns[0].cost_usd == Decimal("1.500000")
    await tracing.aclose()
    generation = next(s for s in exporter.get_finished_spans() if s.name == "llm")
    assert generation.attributes is not None
    assert generation.attributes["chatbot.cost_usd"] == 1.5
    assert generation.attributes["chatbot.cost_calculated"] is True
    assert generation.attributes["gen_ai.response.model"] == "fallback"
    assert json.loads(str(generation.attributes["langfuse.observation.cost_details"])) == {
        "total": 1.5
    }


async def test_a_turn_that_fails_everywhere_reports_it() -> None:
    gw, _ = gateway(
        FaultyProvider("a:primary", [fail(False)]), FaultyProvider("b:fallback", [fail(False)])
    )
    store = MemoryStore()
    agent = Agent(store, gw, ToolRegistry(), Settings(), system="Rules")
    user = await store.upsert_user(User(external_id="alice", display_name="Alice"))
    session = await store.create_session(user.id)
    events = [e async for e in agent.respond(session.id, "Hi", user)]
    assert isinstance(events[-1], TurnFailed) and events[-1].code == "provider_error"
