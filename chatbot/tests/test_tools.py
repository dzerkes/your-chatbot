"""Tool dispatch boundaries and pagination progress under payload limits."""

import asyncio
import logging
from uuid import uuid4

import pytest
from chatbot.errors import ToolError
from chatbot.logging import ContextFilter
from chatbot.tools.base import Tool, ToolContext, ToolInput
from chatbot.tools.pagination import (
    PaginationParams,
    ToolPage,
    build_page,
    decode_cursor,
    encode_cursor,
    paginate,
)
from chatbot.tools.registry import ToolRegistry
from chatbot.types import ToolUse, User
from pydantic import Field, JsonValue, ValidationError

pytestmark = pytest.mark.unit


class ListInput(ToolInput):
    query: str = Field(min_length=1, description="Name to find")
    pagination: PaginationParams = Field(default_factory=PaginationParams)


def context(name: str = "alice") -> ToolContext:
    return ToolContext(
        user=User(external_id=name, display_name=name),
        session_id=uuid4(),
        turn_id=uuid4(),
        trace_id=f"trace-{name}",
    )


async def test_registered_toy_tool_schema_context_and_result() -> None:
    seen: list[ToolContext] = []

    async def list_names(inputs: ListInput, ctx: ToolContext) -> ToolPage:
        seen.append(ctx)
        return paginate([{"name": inputs.query, "owner": ctx.user.external_id}], inputs.pagination)

    registry = ToolRegistry()
    registry.register(
        Tool("list_names", "Find names visible to the calling user", ListInput, list_names)
    )
    definition = registry.definitions()[0]
    assert definition.name == "list_names"
    assert definition.input_schema["additionalProperties"] is False
    properties = definition.input_schema["properties"]
    assert isinstance(properties, dict)
    assert "user" not in properties
    assert definition.input_schema["required"] == ["query"]
    ctx = context()
    result = await registry.execute(
        ToolUse(id="call-1", name="list_names", arguments={"query": "Project"}), ctx
    )
    assert seen == [ctx]
    assert result.tool_use_id == "call-1"
    assert not result.is_error
    assert result.content == {
        "items": [{"name": "Project", "owner": "alice"}],
        "total": 1,
        "next_cursor": None,
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"query": ""},
        {"query": "x", "context": {"user": "admin"}},
        {"query": "x", "user": {"external_id": "admin"}},
        {"query": "x", "pagination": {"limit": 101}},
    ],
)
async def test_invalid_arguments_never_reach_handler(arguments: dict[str, JsonValue]) -> None:
    async def handler(inputs: ListInput, ctx: ToolContext) -> ToolPage:
        pytest.fail("Invalid arguments reached the tool")

    tool = Tool("toy", "Toy tool", ListInput, handler)
    with pytest.raises(ToolError, match="Invalid tool arguments"):
        await tool.execute(arguments, context())


async def test_registry_duplicate_unknown_and_payload_limit() -> None:
    async def handler(inputs: ListInput, ctx: ToolContext) -> ToolPage:
        return ToolPage(items=["界" * 100], total=1)

    registry = ToolRegistry(max_bytes=100)
    tool = Tool("toy", "Toy tool", ListInput, handler)
    registry.register(tool)
    with pytest.raises(ToolError, match="already registered"):
        registry.register(tool)
    with pytest.raises(ToolError, match="Unknown tool"):
        registry.get("missing")
    with pytest.raises(ToolError, match="size limit"):
        await registry.execute(ToolUse(id="1", name="toy", arguments={"query": "x"}), context())


async def test_tool_item_limit_and_safe_errors() -> None:
    async def too_many(inputs: ListInput, ctx: ToolContext) -> ToolPage:
        return ToolPage(items=[1, 2], total=2)

    with pytest.raises(ToolError, match="size limit"):
        await Tool("toy", "Toy tool", ListInput, too_many).execute(
            {"query": "x"}, context(), max_items=1
        )

    async def failing(inputs: ListInput, ctx: ToolContext) -> ToolPage:
        raise RuntimeError("private database details")

    with pytest.raises(ToolError, match=r"^Tool execution failed$") as caught:
        await Tool("toy", "Toy tool", ListInput, failing).execute({"query": "x"}, context())
    assert isinstance(caught.value.__cause__, RuntimeError)


async def test_tool_trace_context_is_scoped_per_concurrent_call() -> None:
    seen: dict[str, object] = {}

    async def handler(inputs: ListInput, ctx: ToolContext) -> ToolPage:
        await asyncio.sleep(0)
        record = logging.makeLogRecord({})
        ContextFilter().filter(record)
        seen[ctx.user.external_id] = tuple(
            getattr(record, key) for key in ("session_id", "turn_id", "trace_id")
        )
        return ToolPage(items=[], total=0)

    tool = Tool("toy", "Toy tool", ListInput, handler)
    contexts = [context("alice"), context("bob")]
    await asyncio.gather(*(tool.execute({"query": "x"}, ctx) for ctx in contexts))
    for ctx in contexts:
        assert seen[ctx.user.external_id] == (str(ctx.session_id), str(ctx.turn_id), ctx.trace_id)
    record = logging.makeLogRecord({})
    ContextFilter().filter(record)
    assert record.__dict__["session_id"] is None


def test_pages_visit_every_item_once_and_stop() -> None:
    items: list[JsonValue] = [{"id": index} for index in range(7)]
    params = PaginationParams(limit=3)
    visited: list[JsonValue] = []
    while True:
        page = paginate(items, params, scope="alice:projects:active")
        visited.extend(page.items)
        assert page.total == 7
        assert len(page.items) <= 3
        if page.next_cursor is None:
            break
        params = PaginationParams(limit=3, cursor=page.next_cursor)
    assert visited == items
    assert paginate([], PaginationParams()) == ToolPage(items=[], total=0)
    assert paginate(items, PaginationParams(cursor=encode_cursor(7))).items == []


def test_byte_limit_shrinks_page_without_skipping_items() -> None:
    items: list[JsonValue] = ["界" * 60, "a" * 180, "b" * 180]
    first = paginate(items, PaginationParams(limit=3), max_bytes=450)
    assert first.items == items[:1]
    assert first.byte_size() <= 450
    assert decode_cursor(first.next_cursor) == 1
    second = paginate(items, PaginationParams(limit=3, cursor=first.next_cursor), max_bytes=450)
    assert second.items == items[1:]
    assert second.next_cursor is None
    with pytest.raises(ToolError, match="size limit"):
        paginate(["x" * 1000], PaginationParams(), max_bytes=400)


@pytest.mark.parametrize("cursor", ["", "not-base64!", "W10=", "e30=", "a" * 513])
def test_bad_cursors_are_tool_errors(cursor: str) -> None:
    with pytest.raises(ToolError, match="Invalid pagination cursor"):
        decode_cursor(cursor)


def test_cursor_scope_range_and_inconsistent_fetch() -> None:
    cursor = encode_cursor(4, scope="alice:active")
    assert decode_cursor(cursor, scope="alice:active") == 4
    with pytest.raises(ToolError, match="cursor"):
        decode_cursor(cursor, scope="bob:active")
    with pytest.raises(ToolError, match="Inconsistent"):
        paginate([1], PaginationParams(cursor=encode_cursor(2)))
    with pytest.raises(ToolError, match="did not advance"):
        build_page([], total=1)
    with pytest.raises(ValidationError):
        ToolPage(items=[1, 2], total=1)
