"""Signed identity for tool calls that cross a process boundary (MCP).

The backend authenticates the user, then signs a short-lived token carrying who
is calling and in which conversation. The tool server verifies the signature,
audience and expiry before running any tool, and builds ``ToolContext`` only
from the token, never from tool arguments, which the model writes.

Tokens are HS256 JWTs signed with ``MCP_TOKEN_SECRET``, a secret shared by the
backend and the tool server. Anyone holding the secret can mint identities, so
it is a server-side secret like a database password. An asymmetric key
(backend signs, server only verifies) is the upgrade path when the servers are
run by different teams.
"""

import time
from datetime import UTC, datetime
from uuid import UUID, uuid4

import jwt
from pydantic import SecretStr, ValidationError

from chatbot.errors import ChatbotError
from chatbot.tools.base import ToolContext
from chatbot.types import User

AUDIENCE = "chatbot-tools"
ISSUER = "chatbot-backend"
ALGORITHM = "HS256"
DEFAULT_TTL_SECONDS = 300
MIN_SECRET_LENGTH = 32


class IdentityError(ChatbotError):
    """A tool-call identity token is missing, malformed, expired or forged."""

    code = "identity_error"


def _key(secret: SecretStr) -> str:
    value = secret.get_secret_value()
    if len(value) < MIN_SECRET_LENGTH:
        raise IdentityError(f"MCP_TOKEN_SECRET must be at least {MIN_SECRET_LENGTH} characters")
    return value


def sign_context(
    context: ToolContext, secret: SecretStr, *, ttl_seconds: int = DEFAULT_TTL_SECONDS
) -> str:
    now = int(time.time())
    user = context.user
    claims = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": str(user.id),
        "iat": now,
        "exp": now + ttl_seconds,
        "jti": uuid4().hex,
        "external_id": user.external_id,
        "name": user.display_name,
        "groups": sorted(set(user.groups)),
        "session_id": str(context.session_id),
        "turn_id": str(context.turn_id),
        "trace_id": context.trace_id,
    }
    return jwt.encode(claims, _key(secret), algorithm=ALGORITHM)


def verify_token(token: str, secret: SecretStr) -> ToolContext:
    try:
        claims = jwt.decode(
            token,
            _key(secret),
            algorithms=[ALGORITHM],  # never accept "none" or another algorithm
            audience=AUDIENCE,
            issuer=ISSUER,
            options={"require": ["exp", "iat", "sub", "aud", "iss"]},
            leeway=5,
        )
        groups = claims.get("groups", [])
        if not isinstance(groups, list) or not all(isinstance(g, str) for g in groups):
            raise IdentityError("Invalid groups claim")
        user = User(
            id=UUID(claims["sub"]),
            external_id=claims["external_id"],
            display_name=claims["name"],
            groups=groups,
        )
        trace_id = claims.get("trace_id")
        return ToolContext(
            user=user,
            session_id=UUID(claims["session_id"]),
            turn_id=UUID(claims["turn_id"]),
            trace_id=trace_id if isinstance(trace_id, str) else None,
        )
    except IdentityError:
        raise
    except (jwt.PyJWTError, KeyError, TypeError, ValueError, ValidationError) as exc:
        raise IdentityError("Invalid or expired tool identity token") from exc


def mint_dev_token(
    secret: SecretStr,
    *,
    external_id: str,
    groups: list[str],
    ttl_seconds: int = 3600,
) -> tuple[str, datetime]:
    """A token for an external MCP client in development, outside any chat session."""
    user = User(external_id=external_id, display_name=external_id, groups=groups)
    context = ToolContext(user=user, session_id=uuid4(), turn_id=uuid4())
    token = sign_context(context, secret, ttl_seconds=ttl_seconds)
    return token, datetime.fromtimestamp(time.time() + ttl_seconds, UTC)
