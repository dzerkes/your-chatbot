"""Opaque offset cursors and JSON pages bounded by item count and UTF-8 bytes."""

import base64
import binascii
import hashlib
import json
from collections.abc import Sequence
from typing import Self

from pydantic import Field, JsonValue, NonNegativeInt, model_validator

from chatbot.errors import ToolError
from chatbot.types import DomainModel, NonEmptyStr

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
MAX_RESULT_BYTES = 16_384


class PaginationParams(DomainModel):
    limit: int = Field(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE)
    cursor: str | None = Field(default=None, min_length=1, max_length=512)


class ToolPage(DomainModel):
    items: list[JsonValue]
    total: NonNegativeInt
    next_cursor: NonEmptyStr | None = None

    @model_validator(mode="after")
    def valid_total(self) -> Self:
        if self.total < len(self.items):
            raise ValueError("Total cannot be smaller than the page")
        return self

    def byte_size(self) -> int:
        return len(self.model_dump_json().encode("utf-8"))


def encode_cursor(offset: int, *, scope: str = "") -> str:
    if type(offset) is not int or not 0 <= offset <= 2**63 - 1:
        raise ValueError("Offset must be a nonnegative 64-bit integer")
    payload = {"v": 1, "offset": offset, "scope": hashlib.sha256(scope.encode()).hexdigest()}
    return base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()


def decode_cursor(cursor: str | None, *, scope: str = "") -> int:
    if cursor is None:
        return 0
    try:
        if not cursor or len(cursor) > 512:
            raise ValueError("Invalid cursor length")
        payload = json.loads(base64.b64decode(cursor, altchars=b"-_", validate=True))
        if not isinstance(payload, dict) or set(payload) != {"v", "offset", "scope"}:
            raise ValueError("Invalid cursor shape")
        offset = payload["offset"]
        if (
            type(payload["v"]) is not int
            or payload["v"] != 1
            or type(offset) is not int
            or not 0 <= offset <= 2**63 - 1
            or payload["scope"] != hashlib.sha256(scope.encode()).hexdigest()
        ):
            raise ValueError("Invalid cursor values")
        return offset
    except (ValueError, TypeError, UnicodeError, binascii.Error) as exc:
        raise ToolError("Invalid pagination cursor") from exc


def build_page(
    items: Sequence[JsonValue],
    *,
    total: int,
    offset: int = 0,
    limit: int = DEFAULT_PAGE_SIZE,
    max_bytes: int = MAX_RESULT_BYTES,
    scope: str = "",
) -> ToolPage:
    """Bound a fetched slice starting at offset; keep the cursor at the next omitted item."""
    if not 1 <= limit <= MAX_PAGE_SIZE or max_bytes < 1:
        raise ValueError("Invalid page limits")
    if offset < 0 or total < offset or len(items) > total - offset:
        raise ToolError("Inconsistent pagination result")
    selected = list(items[:limit])
    if not selected and offset < total:
        raise ToolError("Pagination did not advance")
    while True:
        end = offset + len(selected)
        page = ToolPage(
            items=selected,
            total=total,
            next_cursor=encode_cursor(end, scope=scope) if end < total else None,
        )
        if page.byte_size() <= max_bytes:
            return page
        if len(selected) <= 1:
            raise ToolError("Tool result exceeds the size limit")
        selected = selected[:-1]


def paginate(
    items: Sequence[JsonValue],
    params: PaginationParams,
    *,
    max_bytes: int = MAX_RESULT_BYTES,
    scope: str = "",
) -> ToolPage:
    """Page an already ordered in-memory sequence. Database tools should fetch a slice."""
    offset = decode_cursor(params.cursor, scope=scope)
    return build_page(
        items[offset : offset + params.limit],
        total=len(items),
        offset=offset,
        limit=params.limit,
        max_bytes=max_bytes,
        scope=scope,
    )
