"""Signed identity tokens for tool calls across process boundaries."""

import base64
import json
import time
from uuid import uuid4

import jwt
import pytest
from chatbot.tools.base import ToolContext
from chatbot.tools.identity import IdentityError, mint_dev_token, sign_context, verify_token
from chatbot.types import User
from pydantic import SecretStr

pytestmark = pytest.mark.unit
SECRET = SecretStr("k" * 32)


def context() -> ToolContext:
    user = User(external_id="bob", display_name="Bob", groups=["finance", "employees", "finance"])
    return ToolContext(user=user, session_id=uuid4(), turn_id=uuid4(), trace_id="0" * 32)


def test_round_trip_keeps_user_groups_and_conversation() -> None:
    original = context()
    restored = verify_token(sign_context(original, SECRET), SECRET)
    assert restored.user.id == original.user.id
    assert restored.user.external_id == "bob"
    assert restored.user.groups == ["employees", "finance"]
    assert (restored.session_id, restored.turn_id, restored.trace_id) == (
        original.session_id,
        original.turn_id,
        original.trace_id,
    )


def test_tampered_groups_break_the_signature() -> None:
    header, payload, signature = sign_context(context(), SECRET).split(".")
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    claims["groups"] = ["hr", "leadership"]
    forged = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    with pytest.raises(IdentityError, match="Invalid or expired"):
        verify_token(f"{header}.{forged}.{signature}", SECRET)


@pytest.mark.parametrize(
    "claims",
    [
        {"aud": "someone-else"},
        {"iss": "someone-else"},
        {"exp": int(time.time()) - 60},
        {"groups": "finance"},
        {"session_id": "not-a-uuid"},
    ],
)
def test_wrong_claims_are_rejected(claims: dict[str, object]) -> None:
    valid = jwt.decode(
        sign_context(context(), SECRET),
        SECRET.get_secret_value(),
        algorithms=["HS256"],
        audience="chatbot-tools",
    )
    token = jwt.encode({**valid, **claims}, SECRET.get_secret_value(), algorithm="HS256")
    with pytest.raises(IdentityError):
        verify_token(token, SECRET)


def test_short_secrets_are_refused() -> None:
    with pytest.raises(IdentityError, match="at least 32"):
        sign_context(context(), SecretStr("short"))
    with pytest.raises(IdentityError, match="at least 32"):
        verify_token("x.y.z", SecretStr(""))


def test_dev_tokens_carry_the_given_groups_and_expire() -> None:
    token, expires = mint_dev_token(
        SECRET, external_id="inspector", groups=["employees"], ttl_seconds=60
    )
    restored = verify_token(token, SECRET)
    assert (restored.user.external_id, restored.user.groups) == ("inspector", ["employees"])
    assert 50 < expires.timestamp() - time.time() <= 60
