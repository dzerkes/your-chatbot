"""FastAPI application factory; the core owns the chatbot's behavior."""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from chatbot.agent import Agent
from chatbot.app import build_chatbot
from chatbot.config import Settings
from chatbot.errors import ChatbotError, ConflictError, NotFoundError
from chatbot.logging import configure_logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.auth import parse_dev_tokens
from app.deps import Runtime
from app.routes import chat, health, sessions

type ChatbotFactory = Callable[[Settings], AbstractAsyncContextManager[Agent]]
logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    chatbot_factory: ChatbotFactory = build_chatbot,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        configured = settings if settings is not None else Settings()
        tokens = parse_dev_tokens(configured.auth_dev_tokens)
        configure_logging(configured)
        async with chatbot_factory(configured) as chatbot:
            application.state.runtime = Runtime(configured, chatbot, tokens)
            try:
                yield
            finally:
                application.state.runtime = None

    application = FastAPI(title="Your Chatbot API", version="0.1.0", lifespan=lifespan)
    application.include_router(health.router)
    application.include_router(sessions.router)
    application.include_router(chat.router)

    @application.exception_handler(ChatbotError)
    async def chatbot_error(request: Request, exc: ChatbotError) -> JSONResponse:
        status = (
            404
            if isinstance(exc, NotFoundError)
            else 409
            if isinstance(exc, ConflictError)
            else 500
        )
        return JSONResponse(status_code=status, content={"detail": str(exc), "code": exc.code})

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error("HTTP request failed", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return application


app = create_app()
