"""Application configuration, loaded from environment variables (and an optional .env file)."""

from datetime import datetime
from functools import lru_cache
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True)

    environment: Literal["development", "production", "test"] = "development"
    port: int = 8000
    log_level: str = "INFO"

    # Facility timezone: "today", "last week" etc. are resolved in this zone.
    app_timezone: str = "UTC"

    # Data source
    # mcp = MongoDB MCP Server tools (read-only; needs Node.js) · mongodb = PyMongo driver · memory = tests only
    data_source: Literal["memory", "mongodb", "mcp"] = "mcp"
    mongodb_uri: str | None = None
    mongodb_db: str = "care_app"
    mongodb_collection: str = "residents"
    vocab_refresh_seconds: int = Field(default=300, ge=10)
    mcp_server_command: str = "mongodb-mcp-server"  # always started with --readOnly and telemetry disabled
    mcp_node_path: str | None = None  # Node.js binary for the MCP server; default: first suitable one found
    mcp_max_documents: int = Field(default=5000, ge=100)
    mcp_timeout_seconds: float = Field(default=20.0, gt=0, le=120)

    # Local query-translation model: Qwen3-0.6B (GGUF Q4_K_M, non-thinking mode) served by llama.cpp `llama-server` on the
    # private network. Model path, context size, threads and parallel slots are llama-server flags (deploy/); these
    # settings are the client side. LLM_ENABLED=false → questions the deterministic planner cannot answer get a
    # controlled "not available" reply. There is no external model and no fallback to one.
    llm_enabled: bool = True
    llm_server_url: str = "http://127.0.0.1:8081"
    llm_model_name: str = "qwen3-4b-instruct-2507-q4_k_m"  # previous: "qwen3-0.6b-q4_k_m" (until 2026-11-16)
    llm_max_tokens: int = Field(default=256, ge=32, le=1024)
    llm_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    llm_max_concurrency: int = Field(default=2, ge=1, le=32)  # match llama-server --parallel
    # Sampling. Default = greedy (deterministic plans). Qwen3 thinking mode recommends temperature 0.6, top_p 0.95,
    # top_k 20 (only for evaluation; thinking is not the default — see README "Thinking vs non-thinking").
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    llm_top_k: int = Field(default=1, ge=0, le=200)
    llm_seed: int = 42
    # Extra variables for the model's chat template, as JSON. Qwen3 needs {"enable_thinking": false} (non-thinking
    # mode, required for structured query translation); Qwen2.5's template ignores it. Selecting a model =
    # LLM_SERVER_URL (the server loaded with that GGUF) + LLM_MODEL_NAME + this; prompt, schema and validation are
    # identical for every model.
    llm_chat_template_kwargs: dict[str, bool | str | int] = Field(default_factory=lambda: {"enable_thinking": False})

    # Local topic check (free embedding model on this server): stops clearly off-topic questions
    # before they reach the language model. Margin 0.02 was tuned on held-out questions (0 resident
    # questions blocked); raise it to block less, lower it to block more.
    topic_gate_enabled: bool = True
    topic_gate_model: str = "BAAI/bge-small-en-v1.5"
    topic_gate_cache_dir: str | None = ".models"
    topic_gate_margin: float = Field(default=0.02, ge=0.0, le=0.5)

    # Semantic router (Aurelio semantic-router, local encoder): picks the kind of question for the deterministic
    # planner. Routes are defined in a YAML file (default: app/data/intent_routes.yaml).
    semantic_router_enabled: bool = True
    semantic_router_routes_file: str | None = None
    # Answer confident, simple questions directly with MongoDB MCP tools (no model call); rules in intent_routes.yaml
    direct_answers_enabled: bool = True

    # Plan cache and conversation memory (per process; use Redis for multiple replicas)
    cache_max_entries: int = Field(default=5000, ge=0)
    cache_ttl_seconds: int = Field(default=86400, ge=60)
    session_ttl_seconds: int = Field(default=1800, ge=60)
    session_max_entries: int = Field(default=10000, ge=100)

    # HTTP
    rate_limit_per_minute: int = Field(default=60, ge=1)
    rate_limit_per_day: int = Field(default=2000, ge=1)  # per IP, on POST /api/query
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

    def today(self) -> str:
        return datetime.now(ZoneInfo(self.app_timezone)).date().isoformat()


@lru_cache
def get_settings() -> Settings:
    return Settings()
