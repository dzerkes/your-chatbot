"""Shared types for the search legs: filters, candidates and the leg interface.

Access control is part of the filters, so every leg applies it inside its SQL
query. Filtering after ranking would let restricted chunks take the top places
and push permitted ones out, and would make one forgotten check a data leak.
"""

import json
from dataclasses import dataclass
from typing import Protocol

from psycopg_pool import AsyncConnectionPool

type SqlParam = str | int | list[str]


@dataclass(frozen=True)
class SearchFilters:
    """What the caller may see and what it asked to narrow to.

    ``groups`` comes from the authenticated user, never from the model. A chunk
    matches when it shares at least one group with the caller; a caller with no
    groups matches nothing.
    """

    groups: tuple[str, ...]
    source_prefix: str | None = None  # e.g. "engineering/"
    tags: tuple[str, ...] = ()  # the document must carry all of these

    def where(self) -> tuple[str, list[SqlParam]]:
        """SQL conditions over ``c`` (rag.chunks) and ``d`` (rag.documents)."""
        clauses = ["c.allowed_groups && %s::text[]"]
        params: list[SqlParam] = [sorted(set(self.groups))]
        if self.source_prefix:
            escaped = (
                self.source_prefix.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
            )
            clauses.append("d.source LIKE %s")
            params.append(f"{escaped}%")
        if self.tags:
            clauses.append("d.metadata -> 'tags' @> %s::jsonb")
            params.append(json.dumps(sorted(set(self.tags))))
        return " AND ".join(clauses), params


@dataclass(frozen=True)
class Candidate:
    chunk_id: int
    document_id: int
    source: str
    title: str
    heading: str
    text: str
    score: float  # leg-specific; only comparable within one leg's results


CANDIDATE_COLUMNS = "c.id, c.document_id, d.source, d.title, c.heading, c.text"


def candidate(row: tuple[int, int, str, str, str, str, float]) -> Candidate:
    return Candidate(*row[:6], score=float(row[6]))


class SearchLeg(Protocol):
    name: str

    async def search(
        self, pool: AsyncConnectionPool, query: str, filters: SearchFilters, limit: int
    ) -> list[Candidate]:
        """Best ``limit`` chunks for ``query`` among those ``filters`` allow, best first."""
        ...
