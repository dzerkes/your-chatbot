"""The MCP data server with fake tools: transport, identity and parity with local tools."""

from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import httpx2
import jwt
import pytest
from chatbot.config import Settings
from chatbot.errors import ToolError
from chatbot.tools.base import Tool, ToolContext, ToolInput
from chatbot.tools.identity import mint_dev_token, sign_context
from chatbot.tools.mcp_client import McpTool, McpToolset
from chatbot.tools.pagination import ToolPage
from chatbot.tools.registry import ToolRegistry
from chatbot.types import User
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import Field, SecretStr

from mcp_servers.data_server import create_app

from .server import serve

pytestmark = pytest.mark.unit
SECRET = SecretStr("s" * 40)


class EchoInput(ToolInput):
    text: str = Field(min_length=1, description="Text to echo")


def fake_registry(seen: list[ToolContext]) -> ToolRegistry:
    async def echo(inputs: EchoInput, context: ToolContext) -> ToolPage:
        seen.append(context)
        if inputs.text == "fail":
            raise ToolError("Echo refused this text")
        return ToolPage(items=[inputs.text, context.user.groups], total=2)  # type: ignore[list-item]

    registry = ToolRegistry()
    registry.register(Tool("echo", "Echo text back with the caller's groups.", EchoInput, echo))
    return registry


@pytest.fixture(scope="module")
def server() -> Iterator[tuple[str, list[ToolContext], ToolRegistry]]:
    seen: list[ToolContext] = []
    registry = fake_registry(seen)
    app = create_app(Settings(mcp_token_secret=SECRET), registry=registry)
    with serve(app) as base:
        yield base, seen, registry


def settings_for(base: str, secret: SecretStr = SECRET) -> Settings:
    return Settings(mcp_server_url=f"{base}/mcp", mcp_token_secret=secret)  # type: ignore[arg-type]


def context(*groups: str) -> ToolContext:
    user = User(external_id="alice", display_name="Alice", groups=list(groups))
    return ToolContext(user=user, session_id=uuid4(), turn_id=uuid4(), trace_id="ab" * 16)


async def test_mcp_tools_match_local_tools(
    server: tuple[str, list[ToolContext], ToolRegistry],
) -> None:
    base, seen, local = server
    async with McpToolset(settings_for(base)) as toolset:
        assert toolset.registry.definitions() == local.definitions()
        remote = toolset.registry.get("echo")
        assert isinstance(remote, McpTool)
        caller = context("employees", "finance")
        over_mcp = await remote.execute({"text": "hi"}, caller)
        in_process = await local.get("echo").execute({"text": "hi"}, caller)
    assert over_mcp == in_process == ToolPage(items=["hi", ["employees", "finance"]], total=2)
    # The server rebuilt the caller's context from the signed token, not from arguments.
    received = seen[-1]
    assert (received.user.id, received.user.groups) == (caller.user.id, ["employees", "finance"])
    assert (received.session_id, received.turn_id, received.trace_id) == (
        caller.session_id,
        caller.turn_id,
        caller.trace_id,
    )


async def test_errors_cross_the_boundary_as_tool_errors(
    server: tuple[str, list[ToolContext], ToolRegistry],
) -> None:
    base, _, _ = server
    async with McpToolset(settings_for(base)) as toolset:
        echo = toolset.registry.get("echo")
        with pytest.raises(ToolError, match="Echo refused this text"):
            await echo.execute({"text": "fail"}, context())
        # Identity cannot ride in the arguments: unknown fields are rejected.
        with pytest.raises(ToolError, match="groups: Extra inputs are not permitted"):
            await echo.execute({"text": "x", "groups": ["finance"]}, context())
        with pytest.raises(ToolError, match="size limit"):
            await echo.execute({"text": "x"}, context(), max_items=1)


async def test_client_with_the_wrong_secret_cannot_connect(
    server: tuple[str, list[ToolContext], ToolRegistry],
) -> None:
    base, _, _ = server
    with pytest.raises(Exception):  # noqa: B017 - the SDK wraps the 401 in its own error
        async with McpToolset(settings_for(base, SecretStr("x" * 40))):
            pass


@pytest.mark.parametrize("case", ["missing", "garbage", "wrong-secret", "expired", "none-alg"])
async def test_requests_without_a_valid_token_get_401(
    server: tuple[str, list[ToolContext], ToolRegistry], case: str
) -> None:
    base, _, _ = server
    tokens = {
        "missing": None,
        "garbage": "not-a-token",
        "wrong-secret": sign_context(context(), SecretStr("w" * 40)),
        "expired": sign_context(context(), SECRET, ttl_seconds=-60),
        # An unsigned token ("alg": "none") must never be accepted.
        "none-alg": jwt.encode(
            {"sub": str(uuid4()), "aud": "chatbot-tools"},
            None,  # type: ignore[arg-type]
            algorithm="none",
        ),
    }
    headers = {"Accept": "application/json, text/event-stream"}
    if tokens[case] is not None:
        headers["Authorization"] = f"Bearer {tokens[case]}"
    async with httpx2.AsyncClient() as client:
        response = await client.post(
            f"{base}/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer")


async def test_external_mcp_client_lists_and_calls_with_a_dev_token(
    server: tuple[str, list[ToolContext], ToolRegistry],
) -> None:
    """A generic MCP client (not our wrapper) with a token from `python -m mcp_servers.token`."""
    base, seen, _ = server
    token, _ = mint_dev_token(SECRET, external_id="inspector", groups=["employees"])
    async with (
        httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}) as http,
        streamable_http_client(f"{base}/mcp", http_client=http) as (read, write),
        ClientSession(read, write) as session,
    ):
        info = await session.initialize()
        listed = await session.list_tools()
        result: Any = await session.call_tool("echo", {"text": "hello"})
    assert info.server_info.name == "chatbot-data"
    assert [tool.name for tool in listed.tools] == ["echo"]
    assert result.structured_content == {
        "items": ["hello", ["employees"]],
        "total": 2,
        "next_cursor": None,
    }
    assert seen[-1].user.external_id == "inspector"


async def test_health_needs_no_token(server: tuple[str, list[ToolContext], ToolRegistry]) -> None:
    base, _, _ = server
    async with httpx2.AsyncClient() as client:
        response = await client.get(f"{base}/health")
    assert response.json() == {"status": "ok"}
