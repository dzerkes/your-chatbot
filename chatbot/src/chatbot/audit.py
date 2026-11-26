"""Write compliance records using the application role, independently of tracing."""

from typing import Literal

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import JsonValue

from chatbot.tools.base import ToolContext
from chatbot.types import ToolResult, ToolUse


def returned_sources(name: str, result: ToolResult | None) -> list[dict[str, JsonValue]]:
    if result is None or result.is_error or not isinstance(result.content, dict):
        return []
    items = result.content.get("items")
    if not isinstance(items, list):
        return []
    sources: list[dict[str, JsonValue]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if name == "search_docs" and isinstance(item.get("source"), str):
            sources.append({k: item[k] for k in ("source", "chunk_id") if k in item})
        elif name in {"query_projects", "memory"} and "id" in item:
            sources.append(
                {
                    "source": "curated.projects" if name == "query_projects" else "user_memory",
                    "id": item["id"],
                }
            )
    return sources


class AuditLog:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self.pool = pool

    async def turn(self, context: ToolContext, question: str) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                """INSERT INTO audit_log
                       (user_id, session_id, turn_id, trace_id, event_type, question)
                   VALUES (%s, %s, %s, %s, 'turn', %s)""",
                (context.user.id, context.session_id, context.turn_id, context.trace_id, question),
            )

    async def tool(
        self,
        context: ToolContext,
        call: ToolUse,
        result: ToolResult | None,
        outcome: Literal["returned", "failed", "interrupted"],
    ) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                """INSERT INTO audit_log (user_id, session_id, turn_id, trace_id, event_type,
                       call_id, tool_name, arguments, outcome, sources)
                   VALUES (%s, %s, %s, %s, 'tool', %s, %s, %s, %s, %s)""",
                (
                    context.user.id,
                    context.session_id,
                    context.turn_id,
                    context.trace_id,
                    call.id,
                    call.name,
                    Jsonb(call.arguments),
                    outcome,
                    Jsonb(returned_sources(call.name, result)),
                ),
            )
