"""Tools served by an MCP server, behind the same interface as local tools.

``McpToolset`` lists the server's tools at startup and registers one
``McpTool`` per tool, so the agent loop and the tool registry do not change.
Each call opens a short stateless session whose HTTP ``Authorization`` header
carries a signed token for the calling user, so one connection never mixes two
users' identities. On localhost this costs a few milliseconds per call.

Validation, access control and size limits run on the server with the same
code as in-process tools; the client re-checks the size limits it was given.
"""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Self
from uuid import uuid4

import httpx2
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent
from pydantic import JsonValue, SecretStr, ValidationError

from chatbot.config import Settings
from chatbot.errors import ToolError
from chatbot.llm.base import ToolDefinition
from chatbot.tools.base import ToolContext
from chatbot.tools.identity import IdentityError, sign_context
from chatbot.tools.pagination import MAX_PAGE_SIZE, MAX_RESULT_BYTES, ToolPage
from chatbot.tools.registry import ToolRegistry
from chatbot.types import User

logger = logging.getLogger(__name__)
CALL_TIMEOUT_SECONDS = 60.0
# Listing tools reveals no data, but the server still requires a signed token.
_SERVICE_USER = User(external_id="chatbot-service", display_name="Chatbot service")


class McpConnection:
    def __init__(self, url: str, secret: SecretStr) -> None:
        self.url = url
        self.secret = secret

    @asynccontextmanager
    async def session(self, context: ToolContext) -> AsyncIterator[ClientSession]:
        try:
            token = sign_context(context, self.secret)
        except IdentityError as exc:
            raise ToolError("Tool server identity is not configured") from exc
        async with (
            httpx2.AsyncClient(
                headers={"Authorization": f"Bearer {token}"},
                timeout=httpx2.Timeout(CALL_TIMEOUT_SECONDS),
            ) as http,
            streamable_http_client(self.url, http_client=http) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            yield session


class McpTool:
    def __init__(
        self, connection: McpConnection, name: str, description: str, schema: dict[str, JsonValue]
    ) -> None:
        self.connection = connection
        self.name = name
        self.description = description
        self.input_schema = schema

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description, input_schema=self.input_schema
        )

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolContext,
        *,
        max_items: int = MAX_PAGE_SIZE,
        max_bytes: int = MAX_RESULT_BYTES,
    ) -> ToolPage:
        try:
            async with self.connection.session(context) as session:
                result = await session.call_tool(self.name, arguments)
        except ToolError:
            raise
        except Exception as exc:  # transport, protocol or server failure
            logger.warning("MCP call to %s failed: %s", self.name, exc)
            raise ToolError("The tool server is unavailable; try again shortly") from exc
        page = _page(result)
        if len(page.items) > max_items or page.byte_size() > max_bytes:
            raise ToolError("Tool result exceeds the size limit")
        return page


def _text(result: CallToolResult) -> str:
    return "\n".join(block.text for block in result.content if isinstance(block, TextContent))


def _page(result: object) -> ToolPage:
    if not isinstance(result, CallToolResult):
        raise ToolError("The tool server returned an unexpected result")
    if result.is_error:
        # Error text comes from the server's ToolError, which is written to be safe to show.
        raise ToolError(_text(result) or "Tool execution failed")
    try:
        if result.structured_content is not None:
            return ToolPage.model_validate(result.structured_content)
        return ToolPage.model_validate(json.loads(_text(result)))
    except (ValidationError, ValueError) as exc:
        raise ToolError("The tool server returned an invalid result") from exc


class McpToolset:
    """Discover the server's tools with ``async with`` and expose them as a registry."""

    def __init__(self, settings: Settings) -> None:
        self.connection = McpConnection(str(settings.mcp_server_url), settings.mcp_token_secret)
        self.registry = ToolRegistry()

    async def __aenter__(self) -> Self:
        context = ToolContext(user=_SERVICE_USER, session_id=uuid4(), turn_id=uuid4())
        async with self.connection.session(context) as session:
            listed = await session.list_tools()
        for tool in listed.tools:
            self.registry.register(
                McpTool(self.connection, tool.name, tool.description or "", dict(tool.input_schema))
            )
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None
