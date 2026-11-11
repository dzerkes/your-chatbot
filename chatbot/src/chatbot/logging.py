"""Standard-library logging with scoped context for concurrent turns."""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import TextIO
from uuid import UUID

from chatbot.config import Settings

_context: ContextVar[dict[str, str] | None] = ContextVar("chatbot_log_context", default=None)


@contextmanager
def log_context(
    *,
    session_id: UUID | str | None = None,
    turn_id: UUID | str | None = None,
    trace_id: str | None = None,
) -> Iterator[None]:
    """Bind IDs for this scope, inherit omitted IDs, and restore context on exit."""
    updates = {
        key: str(value)
        for key, value in {
            "session_id": session_id,
            "turn_id": turn_id,
            "trace_id": trace_id,
        }.items()
        if value is not None
    }
    token = _context.set({**(_context.get() or {}), **updates})
    try:
        yield
    finally:
        _context.reset(token)


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        context = _context.get() or {}
        for key in ("session_id", "turn_id", "trace_id"):
            setattr(record, key, context.get(key))
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "session_id": getattr(record, "session_id", None),
            "turn_id": getattr(record, "turn_id", None),
            "trace_id": getattr(record, "trace_id", None),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(settings: Settings, *, stream: TextIO | None = None) -> None:
    """Configure the process at startup; repeated calls replace the previous setup."""
    handler = logging.StreamHandler(stream)
    handler.addFilter(ContextFilter())
    handler.setFormatter(
        JsonFormatter()
        if settings.log_format == "json"
        else logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s "
            "session_id=%(session_id)s turn_id=%(turn_id)s trace_id=%(trace_id)s %(message)s"
        )
    )
    logging.basicConfig(level=settings.log_level, handlers=[handler], force=True)
