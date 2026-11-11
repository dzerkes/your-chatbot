"""Typed access to resources created by the application lifespan."""

from dataclasses import dataclass
from typing import Annotated

from chatbot.agent import Agent
from chatbot.config import Settings
from fastapi import Depends, HTTPException, Request


@dataclass(frozen=True)
class Runtime:
    settings: Settings
    chatbot: Agent
    dev_tokens: tuple[tuple[bytes, str], ...]


def get_runtime(request: Request) -> Runtime:
    runtime = getattr(request.app.state, "runtime", None)
    if not isinstance(runtime, Runtime):
        raise HTTPException(status_code=503, detail="Service is not ready")
    return runtime


RuntimeDep = Annotated[Runtime, Depends(get_runtime)]
