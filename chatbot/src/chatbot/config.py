"""Validated application settings; the only application module that reads the environment."""

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import (
    Field,
    HttpUrl,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    field_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Port = Annotated[int, Field(ge=1, le=65535)]
type ContextStage = Literal["full", "sliding_window", "compact_tools", "summary"]


class Settings(BaseSettings):
    """Load defaults, then .env in the working directory, then environment and init overrides.

    Blank environment values use defaults. No API keys are needed until their
    provider is used. Construct explicitly and pass settings to consumers.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",  # The shared .env also contains Compose-only settings.
        hide_input_in_errors=True,
        frozen=True,
    )

    app_env: Literal["local", "ci", "prod"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["console", "json"] = "console"

    postgres_host: str = "localhost"
    postgres_port: Port = 5432
    postgres_db: str = "chatbot"
    postgres_user: str = "chatbot_owner"
    postgres_password: SecretStr = SecretStr("change-me")
    app_db_user: str = "chatbot_app"
    app_db_password: SecretStr = SecretStr("change-me")
    ro_db_user: str = "chatbot_ro"
    ro_db_password: SecretStr = SecretStr("change-me")
    seed_demo_data: bool = False  # Used only by the explicit database bootstrap job.

    llm_primary: str = "anthropic:claude-sonnet-5-5"
    llm_fallbacks: Annotated[tuple[str, ...], NoDecode] = ()
    llm_summary_model: str = "anthropic:claude-haiku-5-5"
    anthropic_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    llm_max_output_tokens: PositiveInt = 2048
    llm_timeout_seconds: PositiveFloat = 60
    llm_max_retries: NonNegativeInt = 2
    agent_max_steps: PositiveInt = 8

    # Comma-separated pipeline, applied in this order: summary, compact_tools,
    # sliding_window. "full" (no trimming) and "sliding_window" may stand alone.
    context_strategy: Annotated[tuple[ContextStage, ...], NoDecode] = ("summary", "compact_tools")
    context_max_input_tokens: PositiveInt = 60000
    context_keep_last_turns: PositiveInt = 10  # stored turns kept verbatim
    # Summarize once this many turns beyond keep_last_turns are unsummarized, in one
    # large step, so the cached prompt prefix changes rarely.
    context_summary_trigger_turns: PositiveInt = 20
    prompt_caching: bool = True

    # The vector column size is fixed by the rag migration from embedding_dim, so
    # changing the model to one with another dimension needs a schema change too.
    embedding_model: str = "local:BAAI/bge-small-en-v1.5"
    embedding_dim: PositiveInt = 384
    embedding_cache_dir: str = ".cache/fastembed"
    reranker: Literal["none", "local", "api"] = "local"
    # local: a fastembed cross-encoder; api: a Cohere-compatible /rerank endpoint.
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    reranker_api_url: HttpUrl | None = None
    reranker_api_key: SecretStr = SecretStr("")
    rerank_candidates: PositiveInt = 20  # fused hits rescored by the reranker
    retrieval_candidates: PositiveInt = 40
    retrieval_page_size: Annotated[int, Field(ge=1, le=10)] = 5

    # inprocess: tools run inside the chatbot process. mcp: the agent calls them on
    # the MCP data server, which alone holds the read-only role and the models.
    tools_transport: Literal["inprocess", "mcp"] = "inprocess"
    mcp_server_url: HttpUrl = HttpUrl("http://localhost:8001/mcp")
    mcp_server_port: Port = 8001
    mcp_token_secret: SecretStr = SecretStr("")  # shared by backend and MCP server

    memory_enabled: bool = False
    tracing_enabled: bool = False
    langfuse_host: HttpUrl = HttpUrl("https://cloud.langfuse.com")
    langfuse_public_key: SecretStr = SecretStr("")
    langfuse_secret_key: SecretStr = SecretStr("")

    rate_limit_requests_per_minute: PositiveInt = 20
    cost_cap_usd_per_user_per_day: Annotated[Decimal, Field(gt=0)] = Decimal("2.00")

    backend_port: Port = 8000
    backend_url: HttpUrl = HttpUrl("http://localhost:8000")
    frontend_port: Port = 8501
    auth_mode: Literal["dev"] = "dev"
    auth_dev_tokens: SecretStr = SecretStr(
        "dev-token-alice:alice:employees|engineering,dev-token-bob:bob:employees|finance|leadership"
    )

    @field_validator("context_strategy", mode="before")
    @classmethod
    def parse_context_strategy(cls, value: object) -> object:
        if isinstance(value, str):
            value = tuple(part.strip() for part in value.split(",") if part.strip())
        return value

    @field_validator("context_strategy")
    @classmethod
    def validate_context_strategy(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("context_strategy needs one or more distinct stages")
        if "full" in value and len(value) > 1:
            raise ValueError("full keeps everything and cannot be combined with other stages")
        return value

    @field_validator("llm_fallbacks", mode="before")
    @classmethod
    def parse_fallbacks(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value

    @field_validator("llm_primary", "llm_summary_model", "embedding_model")
    @classmethod
    def validate_model(cls, value: str | None) -> str | None:
        if value is not None:
            provider, separator, model = value.partition(":")
            if not separator or not provider.strip() or not model.strip():
                raise ValueError("model must use provider:model notation")
        return value

    @field_validator("llm_fallbacks")
    @classmethod
    def validate_fallbacks(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for model in value:
            cls.validate_model(model)
        return value
