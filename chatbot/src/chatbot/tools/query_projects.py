"""Query approved project fields using the dedicated read-only database login."""

# Transaction and cursor scopes depend on the acquired connection.
# ruff: noqa: SIM117

import json
from datetime import date
from typing import Literal, Self

from psycopg import Error
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import Field, JsonValue, PositiveInt, field_validator, model_validator

from chatbot.config import Settings
from chatbot.errors import ToolError
from chatbot.tools.base import Tool, ToolContext, ToolInput
from chatbot.tools.pagination import PaginationParams, ToolPage, build_page, decode_cursor
from chatbot.types import DomainModel

type ProjectStatus = Literal["planned", "active", "on_hold", "completed", "cancelled"]


class QueryProjectsInput(ToolInput):
    status: ProjectStatus | None = None
    client_id: PositiveInt | None = None
    client: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Case-insensitive literal substring of the client name",
    )
    start_date_from: date | None = Field(default=None, description="Inclusive project start date")
    start_date_to: date | None = Field(default=None, description="Inclusive project start date")
    text: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Case-insensitive literal substring of project name or description",
    )
    pagination: PaginationParams = Field(default_factory=PaginationParams)

    @field_validator("client", "text")
    @classmethod
    def nonblank_filter(cls, value: str | None) -> str | None:
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("Search filters cannot be blank")
        return value

    @model_validator(mode="after")
    def valid_date_range(self) -> Self:
        if (
            self.start_date_from
            and self.start_date_to
            and self.start_date_from > self.start_date_to
        ):
            raise ValueError("Start date range is reversed")
        return self


class ProjectSummary(DomainModel):
    id: int
    name: str
    client_id: int
    client_name: str
    description: str
    status: ProjectStatus
    start_date: date
    end_date: date | None


def _literal_pattern(value: str) -> str:
    # User wildcard characters are data; only our outer '%' performs matching.
    return "%" + value.replace("!", "!!").replace("%", "!%").replace("_", "!_") + "%"


def _filters(inputs: QueryProjectsInput) -> tuple[str, list[str | int | date]]:
    clauses: list[str] = []
    params: list[str | int | date] = []
    for column, value, operator in (
        ("status", inputs.status, "="),
        ("client_id", inputs.client_id, "="),
        ("start_date", inputs.start_date_from, ">="),
        ("start_date", inputs.start_date_to, "<="),
    ):
        if value is not None:
            clauses.append(f"{column} {operator} %s")
            params.append(value)
    if inputs.client is not None:
        clauses.append("client_name ILIKE %s ESCAPE '!'")
        params.append(_literal_pattern(inputs.client))
    if inputs.text is not None:
        clauses.append("(name ILIKE %s ESCAPE '!' OR description ILIKE %s ESCAPE '!')")
        pattern = _literal_pattern(inputs.text)
        params.extend([pattern, pattern])
    return " AND ".join(clauses) if clauses else "TRUE", params


class QueryProjects:
    """Own the read-only pool with ``async with`` and register ``tool()``."""

    def __init__(self, settings: Settings) -> None:
        self._pool = AsyncConnectionPool(
            conninfo=make_conninfo(
                host=settings.postgres_host,
                port=settings.postgres_port,
                dbname=settings.postgres_db,
                user=settings.ro_db_user,
                password=settings.ro_db_password.get_secret_value(),
            ),
            min_size=1,
            max_size=5,
            open=False,
        )

    async def __aenter__(self) -> Self:
        await self._pool.open(wait=True)
        return self

    async def __aexit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        await self._pool.close()

    def tool(self) -> Tool[QueryProjectsInput]:
        return Tool(
            "query_projects",
            "Find projects by status, client, inclusive start-date range, or text in the name "
            "and description. Returns approved fields only, ordered by project ID. "
            "Use next_cursor with the same filters to retrieve more results.",
            QueryProjectsInput,
            self.query,
        )

    async def query(self, inputs: QueryProjectsInput, context: ToolContext) -> ToolPage:
        scope = json.dumps(
            {
                "tool": "query_projects",
                "user": str(context.user.id),
                "filters": inputs.model_dump(mode="json", exclude={"pagination"}),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        offset = decode_cursor(inputs.pagination.cursor, scope=scope)
        where, params = _filters(inputs)
        try:
            async with self._pool.connection() as conn:
                async with conn.transaction():
                    # Count and slice must see one snapshot, even during a concurrent seed/update.
                    await conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
                    async with conn.cursor(row_factory=dict_row) as cursor:
                        await cursor.execute(
                            f"SELECT count(*) AS total FROM curated.projects WHERE {where}", params
                        )
                        count = await cursor.fetchone()
                        assert count is not None
                        total = int(count["total"])
                        await cursor.execute(
                            "SELECT id, name, client_id, client_name, description, status, "
                            f"start_date, end_date FROM curated.projects WHERE {where} "
                            "ORDER BY id LIMIT %s OFFSET %s",
                            [*params, inputs.pagination.limit, offset],
                        )
                        rows = await cursor.fetchall()
        except Error as exc:
            raise ToolError("Project query failed") from exc
        items: list[JsonValue] = [
            ProjectSummary.model_validate(row).model_dump(mode="json") for row in rows
        ]
        return build_page(
            items, total=total, offset=offset, limit=inputs.pagination.limit, scope=scope
        )
