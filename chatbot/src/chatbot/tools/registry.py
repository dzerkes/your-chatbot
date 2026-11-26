"""Registration, schema discovery and bounded dispatch for local and MCP tools."""

from chatbot.errors import ToolError
from chatbot.llm.base import ToolDefinition
from chatbot.tools.base import ToolContext, ToolLike
from chatbot.tools.pagination import MAX_PAGE_SIZE, MAX_RESULT_BYTES
from chatbot.types import ToolResult, ToolUse


class ToolRegistry:
    def __init__(
        self, *, max_items: int = MAX_PAGE_SIZE, max_bytes: int = MAX_RESULT_BYTES
    ) -> None:
        if not 1 <= max_items <= MAX_PAGE_SIZE or max_bytes < 1:
            raise ValueError("Invalid tool result limits")
        self.max_items = max_items
        self.max_bytes = max_bytes
        self._tools: dict[str, ToolLike] = {}

    def register(self, tool: ToolLike) -> None:
        if tool.name in self._tools:
            raise ToolError(f"Tool is already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolLike:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(f"Unknown tool: {name}") from exc

    def tools(self) -> list[ToolLike]:
        return list(self._tools.values())

    def definitions(self) -> list[ToolDefinition]:
        return [tool.definition() for tool in self._tools.values()]

    async def execute(self, call: ToolUse, context: ToolContext) -> ToolResult:
        page = await self.get(call.name).execute(
            call.arguments, context, max_items=self.max_items, max_bytes=self.max_bytes
        )
        return ToolResult(tool_use_id=call.id, content=page.model_dump(mode="json"))
