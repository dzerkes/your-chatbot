"""Composition root and lifecycle for the terminal and future web service."""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from importlib.resources import files

from chatbot.agent import Agent
from chatbot.config import Settings
from chatbot.context import build_context_builder
from chatbot.llm.registry import ProviderRegistry
from chatbot.store.postgres import PostgresStore
from chatbot.tools.local import open_local_tools
from chatbot.tools.mcp_client import McpToolset
from chatbot.tracing import Tracing


@asynccontextmanager
async def build_chatbot(settings: Settings) -> AsyncIterator[Agent]:
    context_builder = build_context_builder(settings)
    async with AsyncExitStack() as stack:
        tracing = Tracing.from_settings(settings)
        stack.push_async_callback(tracing.aclose)
        provider = ProviderRegistry(settings).get_primary()
        stack.push_async_callback(provider.aclose)
        store = await stack.enter_async_context(PostgresStore(settings))
        if settings.tools_transport == "mcp":
            tools = (await stack.enter_async_context(McpToolset(settings))).registry
        else:
            tools = await open_local_tools(settings, stack)
        system = files("chatbot").joinpath("prompts/system.md").read_text(encoding="utf-8")
        yield Agent(
            store,
            provider,
            tools,
            settings,
            system=system,
            context_builder=context_builder,
            tracing=tracing,
        )
