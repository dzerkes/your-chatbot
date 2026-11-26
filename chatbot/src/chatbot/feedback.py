"""Turn feedback and delivery of trace scores to Langfuse."""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated, Literal
from uuid import UUID

import httpx
from pydantic import AwareDatetime, StringConstraints

from chatbot.config import Settings
from chatbot.types import DomainModel

if TYPE_CHECKING:
    from chatbot.store.base import HistoryStore

logger = logging.getLogger(__name__)


class FeedbackInput(DomainModel):
    rating: Literal["up", "down"]
    comment: Annotated[str, StringConstraints(strip_whitespace=True, max_length=2000)] | None = None


class Feedback(FeedbackInput):
    turn_id: UUID
    trace_id: str | None
    created_at: AwareDatetime


type ScoreDelivery = Literal["accepted", "retry", "rejected"]
type SendScore = Callable[[Feedback], Awaitable[ScoreDelivery]]


async def send_score(client: httpx.AsyncClient, feedback: Feedback) -> ScoreDelivery:
    score = {
        "id": str(feedback.turn_id),
        "name": "user-feedback",
        "timestamp": feedback.created_at.isoformat(),
        "traceId": feedback.trace_id,
        "dataType": "BOOLEAN",
        "value": int(feedback.rating == "up"),
        "comment": feedback.comment or "",
    }
    try:
        response = await client.post("api/public/scores", json=[score])
        if response.status_code == 207:
            # A batch contains exactly one score. Partial acceptance rejects this item.
            logger.error("Langfuse rejected the feedback for turn %s", feedback.turn_id)
            return "rejected"
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning(
            "Langfuse score delivery failed for turn %s (%s)", feedback.turn_id, type(exc).__name__
        )
        return "retry"
    return "accepted"


@asynccontextmanager
async def score_exporter(settings: Settings, store: "HistoryStore") -> AsyncIterator[asyncio.Event]:
    wake = asyncio.Event()
    if not settings.tracing_enabled:
        yield wake
        return
    async with httpx.AsyncClient(
        base_url=str(settings.langfuse_host),
        auth=(
            settings.langfuse_public_key.get_secret_value(),
            settings.langfuse_secret_key.get_secret_value(),
        ),
        timeout=5,
    ) as client:

        async def publish(feedback: Feedback) -> ScoreDelivery:
            return await send_score(client, feedback)

        async def run() -> None:
            while True:
                wake.clear()
                try:
                    while await store.export_next_feedback(publish):
                        pass
                except Exception:
                    logger.exception("Could not process pending feedback")
                try:
                    await asyncio.wait_for(wake.wait(), timeout=30)
                except TimeoutError:
                    continue

        task = asyncio.create_task(run())
        try:
            yield wake
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
