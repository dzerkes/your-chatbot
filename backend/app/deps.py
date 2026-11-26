"""Typed access to resources created by the application lifespan."""

import asyncio
from dataclasses import dataclass
from typing import Annotated

from chatbot.agent import Agent
from chatbot.config import Settings
from chatbot.limits import ChatLimits
from fastapi import Depends, HTTPException, Request


@dataclass(frozen=True)
class DevToken:
    token: bytes
    identity: str
    groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class Runtime:
    settings: Settings
    chatbot: Agent
    dev_tokens: tuple[DevToken, ...]
    score_wake: asyncio.Event


def get_runtime(request: Request) -> Runtime:
    runtime = getattr(request.app.state, "runtime", None)
    if not isinstance(runtime, Runtime):
        raise HTTPException(status_code=503, detail="Service is not ready")
    return runtime


RuntimeDep = Annotated[Runtime, Depends(get_runtime)]


def chat_limits(runtime: RuntimeDep) -> ChatLimits:
    return ChatLimits(
        runtime.settings.rate_limit_requests_per_minute,
        runtime.settings.cost_cap_usd_per_user_per_day,
    )


ChatLimitsDep = Annotated[ChatLimits, Depends(chat_limits)]
