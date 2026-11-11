"""Composition root and lifecycle for the terminal and future web service."""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from importlib.resources import files

from chatbot.agent import Agent
from chatbot.config import Settings
from chatbot.context import build_context_builder
from chatbot.llm.registry import ProviderRegistry
from chatbot.store.postgres import PostgresStore
from chatbot.tools.query_projects import QueryProjects
from chatbot.tools.registry import ToolRegistry


@asynccontextmanager
async def build_chatbot(settings: Settings) -> AsyncIterator[Agent]:
    context_builder = build_context_builder(settings)
    async with AsyncExitStack() as stack:
        provider = ProviderRegistry(settings).get_primary()
        stack.push_async_callback(provider.aclose)
        store = await stack.enter_async_context(PostgresStore(settings))
        projects = await stack.enter_async_context(QueryProjects(settings))
        tools = ToolRegistry()
        tools.register(projects.tool())
        system = files("chatbot").joinpath("prompts/system.md").read_text(encoding="utf-8")
        yield Agent(
            store, provider, tools, settings, system=system, context_builder=context_builder
        )
