"""The data tools run in this process: used in-process and by the MCP data server."""

from contextlib import AsyncExitStack

from chatbot.config import Settings
from chatbot.tools.query_projects import QueryProjects
from chatbot.tools.registry import ToolRegistry
from chatbot.tools.search_docs import SearchDocs


async def open_local_tools(settings: Settings, stack: AsyncExitStack) -> ToolRegistry:
    """Open the tools' pools and models on ``stack`` and register them."""
    projects = await stack.enter_async_context(QueryProjects(settings))
    documents = await stack.enter_async_context(SearchDocs(settings))
    tools = ToolRegistry()
    tools.register(projects.tool())
    tools.register(documents.tool())
    return tools
