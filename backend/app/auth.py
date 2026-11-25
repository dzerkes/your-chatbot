"""Local development bearer tokens behind one replaceable identity dependency.

``AUTH_DEV_TOKENS`` is a comma-separated list of ``token:identity[:groups]``,
where groups are separated by ``|``, for example
``dev-token-alice:alice:employees|engineering``. Groups decide which documents
search_docs may return; SSO will supply them from the identity provider later.
"""

import re
from secrets import compare_digest
from typing import Annotated

from chatbot.errors import ChatbotError
from chatbot.types import User
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import SecretStr

from app.deps import DevToken, RuntimeDep

bearer = HTTPBearer(auto_error=False)
_GROUP = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def parse_dev_tokens(value: SecretStr) -> tuple[DevToken, ...]:
    result: list[DevToken] = []
    seen: set[bytes] = set()
    for entry in value.get_secret_value().split(","):
        token, separator, rest = entry.strip().partition(":")
        identity, _, group_list = rest.partition(":")
        token, identity = token.strip(), identity.strip()
        groups = tuple(sorted({g.strip() for g in group_list.split("|") if g.strip()}))
        encoded = token.encode()
        if (
            not separator
            or not token
            or not identity
            or encoded in seen
            or any(not _GROUP.match(group) for group in groups)
        ):
            raise ChatbotError("Invalid AUTH_DEV_TOKENS configuration")
        seen.add(encoded)
        result.append(DevToken(encoded, identity, groups))
    return tuple(result)


async def current_user(
    runtime: RuntimeDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> User:
    match = None
    supplied = credentials.credentials.encode() if credentials is not None else b""
    for candidate in runtime.dev_tokens:
        if compare_digest(candidate.token, supplied):
            match = candidate
    if match is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Groups come from configuration on every request, so a change takes effect at once.
    return await runtime.chatbot.store.upsert_user(
        User(external_id=match.identity, display_name=match.identity, groups=list(match.groups))
    )


CurrentUser = Annotated[User, Depends(current_user)]
