"""Application configuration, loaded from environment variables (and an optional .env file)."""

from datetime import datetime
from functools import lru_cache
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True)

    environment: Literal["development", "production", "test"] = "development"
    port: int = 8000
    log_level: str = "INFO"

    # Facility timezone: "today", "last week" etc. are resolved in this zone.
    app_timezone: str = "UTC"

    # Data source
    data_source: Literal["memory", "mongodb"] = "memory"
    mongodb_uri: str | None = None
    mongodb_db: str = "care_app"
    mongodb_collection: str = "residents"
    vocab_refresh_seconds: int = Field(default=300, ge=10)

    # Which language model translates questions into query plans.
    # "auto" picks Gemini if GEMINI_API_KEY is set, otherwise Anthropic if ANTHROPIC_API_KEY is set.
    llm_provider: Literal["auto", "gemini", "anthropic"] = "auto"

    # Google Gemini (google-genai SDK). GOOGLE_API_KEY is accepted as an alias.
    gemini_api_key: str | None = Field(default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_thinking_level: Literal["minimal", "low", "medium", "high", "default"] = "minimal"
    gemini_max_tokens: int = Field(default=1000, ge=128, le=16000)
    gemini_timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    gemini_max_retries: int = Field(default=2, ge=0, le=5)

    # Anthropic Claude. The SDK reads ANTHROPIC_API_KEY itself.
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: Literal["low", "medium", "high"] = "low"
    anthropic_max_tokens: int = Field(default=4000, ge=256, le=16000)
    anthropic_timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    anthropic_max_retries: int = Field(default=2, ge=0, le=5)
    anthropic_refusal_fallbacks: bool = True

    # Plan cache and conversation memory (per process; use Redis for multiple replicas)
    cache_max_entries: int = Field(default=5000, ge=0)
    cache_ttl_seconds: int = Field(default=86400, ge=60)
    session_ttl_seconds: int = Field(default=1800, ge=60)
    session_max_entries: int = Field(default=10000, ge=100)

    # HTTP
    rate_limit_per_minute: int = Field(default=60, ge=1)
    api_token: str | None = Field(default=None, min_length=16)
    cors_origins: str | None = None
    enable_data_browser: bool = False
    static_dir: str | None = None

    @field_validator("app_timezone")
    @classmethod
    def _valid_timezone(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone {v!r}") from exc
        return v

    @model_validator(mode="after")
    def _check_dependencies(self) -> "Settings":
        if self.data_source == "mongodb" and not self.mongodb_uri:
            raise ValueError("MONGODB_URI is required when DATA_SOURCE=mongodb")
        return self

    def today(self) -> str:
        return datetime.now(ZoneInfo(self.app_timezone)).date().isoformat()


@lru_cache
def get_settings() -> Settings:
    return Settings()
