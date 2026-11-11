"""Frontend-only settings; the UI has no dependency on the chatbot core."""

from typing import Annotated

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrontendSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        hide_input_in_errors=True,
    )
    backend_url: HttpUrl = HttpUrl("http://localhost:8000")
    frontend_port: Annotated[int, Field(ge=1, le=65535)] = 8501
