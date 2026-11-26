from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest
from chatbot.feedback import Feedback, send_score

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("status", "outcome"), [(202, "accepted"), (503, "retry"), (207, "rejected")]
)
async def test_score_delivery_distinguishes_acceptance_from_rejection(
    status: int, outcome: str
) -> None:
    feedback = Feedback(
        turn_id=uuid4(), trace_id=uuid4().hex, created_at=datetime.now(UTC), rating="up"
    )
    async with httpx.AsyncClient(
        base_url="https://receiver.test",
        transport=httpx.MockTransport(lambda _: httpx.Response(status)),
    ) as client:
        assert await send_score(client, feedback) == outcome
