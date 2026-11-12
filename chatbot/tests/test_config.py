from decimal import Decimal
from pathlib import Path

import pytest
from chatbot.config import Settings
from pydantic import SecretStr, ValidationError

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_example_env_is_supported(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text((ROOT / ".env.example").read_text())
    settings = Settings()
    assert settings.postgres_user == "chatbot_owner"
    assert settings.llm_fallbacks == ()
    assert settings.embedding_dim is None
    assert settings.embedding_model is None
    assert settings.memory_enabled is False
    assert settings.cost_cap_usd_per_user_per_day == Decimal("2.00")


def test_settings_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "POSTGRES_PORT=55432\nLOG_LEVEL=DEBUG\nPOSTGRES_ADMIN_PASSWORD=unused\n"
    )
    assert Settings().postgres_port == 55432
    monkeypatch.setenv("POSTGRES_PORT", "55433")
    assert Settings().postgres_port == 55433
    assert Settings(postgres_port=55434).postgres_port == 55434
    assert Settings().log_level == "DEBUG"


def test_blank_optional_values_and_booleans(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_DIM", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("MEMORY_ENABLED", "true")
    monkeypatch.setenv("TRACING_ENABLED", "false")
    settings = Settings()
    assert settings.embedding_dim is None
    assert settings.anthropic_api_key.get_secret_value() == ""
    assert settings.memory_enabled is True
    assert settings.tracing_enabled is False


def test_fallbacks_are_comma_separated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_FALLBACKS", " openai:model-one, anthropic:model-two , ")
    assert Settings().llm_fallbacks == ("openai:model-one", "anthropic:model-two")
    assert Settings(llm_fallbacks=("local:demo",)).llm_fallbacks == ("local:demo",)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("POSTGRES_PORT", "65536"),
        ("LLM_TIMEOUT_SECONDS", "0"),
        ("LLM_MAX_RETRIES", "-1"),
        ("AGENT_MAX_STEPS", "0"),
        ("CONTEXT_STRATEGY", "unknown"),
        ("EMBEDDING_DIM", "-1"),
        ("COST_CAP_USD_PER_USER_PER_DAY", "0"),
        ("BACKEND_URL", "not-a-url"),
        ("LLM_PRIMARY", "no-provider"),
        ("LLM_FALLBACKS", "openai:valid,broken"),
    ],
)
def test_invalid_settings_fail_early(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings()


def test_secrets_are_masked() -> None:
    secret = "private-test-password"
    settings = Settings(postgres_password=SecretStr(secret), openai_api_key=SecretStr(secret))
    assert settings.postgres_password.get_secret_value() == secret
    assert secret not in repr(settings)
    assert secret not in settings.model_dump_json()


def test_validation_errors_hide_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_PORT", "private-invalid-input")
    with pytest.raises(ValidationError) as error:
        Settings()
    assert "private-invalid-input" not in str(error.value)
