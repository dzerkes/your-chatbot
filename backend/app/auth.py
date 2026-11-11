"""Local development bearer tokens behind one replaceable identity dependency."""

from secrets import compare_digest
from typing import Annotated

from chatbot.errors import ChatbotError
from chatbot.types import User
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import SecretStr

from app.deps import RuntimeDep

bearer = HTTPBearer(auto_error=False)


def parse_dev_tokens(value: SecretStr) -> tuple[tuple[bytes, str], ...]:
    result: list[tuple[bytes, str]] = []
    seen: set[bytes] = set()
    for entry in value.get_secret_value().split(","):
        token, separator, identity = entry.strip().partition(":")
        token, identity = token.strip(), identity.strip()
        encoded = token.encode()
        if not separator or not token or not identity or encoded in seen:
            raise ChatbotError("Invalid AUTH_DEV_TOKENS configuration")
        seen.add(encoded)
        result.append((encoded, identity))
    return tuple(result)


async def current_user(
    runtime: RuntimeDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> User:
    identity = None
    supplied = credentials.credentials.encode() if credentials is not None else b""
    for token, candidate in runtime.dev_tokens:
        if compare_digest(token, supplied):
            identity = candidate
    if identity is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await runtime.chatbot.store.upsert_user(
        User(external_id=identity, display_name=identity)
    )


CurrentUser = Annotated[User, Depends(current_user)]
