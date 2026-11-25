"""Admission limits for authenticated chat requests."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from chatbot.errors import ChatbotError


class RateLimitExceeded(ChatbotError):
    code = "rate_limit_exceeded"


class CostCapExceeded(ChatbotError):
    code = "cost_cap_exceeded"


@dataclass(frozen=True)
class ChatLimits:
    requests_per_minute: int
    daily_usd: Decimal

    def check(self, requests: int, spend: Decimal) -> None:
        if spend >= self.daily_usd:
            raise CostCapExceeded(
                "Your daily chat spending limit has been reached. Try again after midnight UTC.",
                retryable=True,
            )
        if requests >= self.requests_per_minute:
            raise RateLimitExceeded(
                "You have reached your chat request limit. Try again in a minute.", retryable=True
            )


def window_starts(now: datetime) -> tuple[datetime, datetime]:
    midnight = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return now - timedelta(minutes=1), midnight
