"""Application configuration, loaded from environment variables (and an optional .env file)."""

import json
from datetime import datetime
from functools import lru_cache
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
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
    llm_max_tokens: int = Field(default=512, ge=32, le=1024)  # 256 truncated multi-condition plans (2026-11-17)
    # One deadline per model call: waiting for a free slot + inference (qwen_translator._post). Measured 2026-11-17 with
    # Qwen3-4B Q4_K_M on the 4-core dev CPU (i5-1335U, --threads 4, --parallel 2): ~3.3 generated tokens/s, ~6 new
    # prompt tokens/s (the ~1,420-token system prompt is cached); single requests took 6–25 s, growing with plan length
    # (53-token plan: 24.6 s), so 15 s cut off difficult questions. Use a 60 s deadline for slower or queued calls;
    # it still shares one budget between waiting for a slot and inference. Re-measure on the production host.
    # 120 s (was 60): with the prompt cached, questions took 20-90 s on the dev CPU (2026-11-19), so 60 s cut off
    # longer plans. Cold prompts are handled by the startup warm-up, not by this limit.
    llm_timeout_seconds: float = Field(default=120.0, gt=0, le=300)
    # Time budget for ONE question's model calls, counted from when the question arrives and shared by the first call
    # and any retry (None = llm_timeout_seconds). Before,
    # each call got a fresh llm_timeout_seconds, so with a retry one question took 127-153 s at a 120 s setting
    # (benchmark 2026-11-17). A retry is skipped when less than 1 s of the budget is left.
    llm_request_budget_seconds: float | None = Field(default=None, gt=0, le=600)
    llm_max_concurrency: int = Field(default=1, ge=1, le=32)  # match llama-server --parallel (script default 1)
    # Startup warm-up: the first read of the ~1,800-token prompt took ~220-240 s on the 4-core dev CPU (2026-11-19).
    # It must be allowed to FINISH (an abandoned read is not cached), so it gets its own long limit, retried 3 times.
    llm_warmup_timeout_seconds: float = Field(default=600.0, gt=0, le=3600)
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
    # Question brief (nlu/brief.py): when the fast path cannot answer, the local model also gets what the deterministic
    # analysis understood (bge-small route, residents, attributes, values, unrecognised words).
    # Off by default: measured +4–6 s per model call with Qwen3-4B on the 4-core dev CPU (2026-11-17).
    llm_question_brief: bool = False
    # Answer phrasing (engine/phraser.py): the same local model rewrites the template summary in the style the question
    # asked for; the rewrite is shown only if it passes a fact check, otherwise the template answer is kept.
    # Off by default: it adds one model call (~5–15 s on a 4-core CPU) to every data answer, including cached ones.
    llm_response_phrasing: bool = False
    llm_phrasing_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    llm_phrasing_max_tokens: int = Field(default=160, ge=32, le=512)

    # Which client the model questions go to (app/main.py build_translator):
    #   local   = the Qwen model on llama-server above (LLM_SERVER_URL / LLM_MODEL_NAME); the GGUF stays installed.
    #   bedrock = Gemma 4 31B on AWS Bedrock, bedrock-mantle OpenAI-compatible endpoint (app/nlu/bedrock_translator.py).
    # There is never an automatic fallback from one to the other: a Bedrock failure is reported like a local failure.
    # The LLM_* timeout/concurrency/sampling/brief/phrasing settings apply to both (top_k, seed, chat template: local only).
    llm_provider: Literal["local", "bedrock"] = "local"
    bedrock_region: str = "us-west-2"
    bedrock_model_id: str = "google.gemma-4-31b"
    # Gemma 4 is served under /openai/v1 on bedrock-mantle (NOT /v1). None = https://bedrock-mantle.{region}.api.aws/openai/v1
    bedrock_base_url: str | None = None
    # Amazon Bedrock API key (Bearer token). Read from BEDROCK_API_KEY or the AWS-standard AWS_BEARER_TOKEN_BEDROCK.
    # Keep it in backend/.env (git-ignored) or the process environment; it is never logged.
    bedrock_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("bedrock_api_key", "aws_bearer_token_bedrock")
    )
    # Output constraint sent to Bedrock. "none" (default): the prompt asks for the RequestIR JSON object and the normal
    # RequestIR validation + retry decide. Structured outputs are not listed for Gemma 4 on bedrock-mantle and Bedrock's
    # JSON-Schema subset rejects parts of the RequestIR schema, so json_object / json_schema are opt-in experiments.
    bedrock_response_format: Literal["none", "json_object", "json_schema"] = "none"
    # Family-member NAMES as a third kind of person (family_N / person_N placeholders; app/nlu/vocabulary.py). On by
    # default in every environment (decided 2026-11-20); this setting only exists as an off switch, nothing needs to be
    # configured. Note: masking protects names from the model, it does not decide who may see them. Until the API has
    # trusted caller identity, every caller of the shared API_TOKEN gets the CallerContext built from the settings
    # below (all facilities, the configured family permissions).
    family_name_resolution_enabled: bool = True
    # Caller defaults while the API has no per-user identity (app/engine/caller.py). Separate permissions: names and
    # relationships vs phone and email. caller_facility_ids: restrict every query to these resident facilityIds (tests,
    # staging); None = unrestricted (today's behaviour).
    family_names_permitted: bool = True
    family_contacts_permitted: bool = True
    caller_facility_ids: list[str] | None = None
    # Gemma 4 reasons by default; Chat Completions does not return the reasoning, only its latency. None = model default.
    bedrock_reasoning_effort: Literal["low", "medium", "high"] | None = None

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

    # Failure cache and conversation memory (per process; use Redis for multiple replicas). CACHE_MAX_ENTRIES bounds the
    # failure cache (questions whose model plan failed validation). The exact-fingerprint successful-plan cache and its
    # CACHE_TTL_SECONDS were removed on 2026-11-18 in favour of the semantic template cache below.
    cache_max_entries: int = Field(default=5000, ge=0)
    session_ttl_seconds: int = Field(default=1800, ge=60)
    session_max_entries: int = Field(default=10000, ge=100)

    # Semantic query-plan template cache (app/engine/semantic_cache.py): checked plan templates in a DEDICATED MongoDB
    # Atlas database with a Vector Search index, reused for re-worded questions after slot filling and the existing
    # validator. Needs its OWN restricted credential (readWrite on PLAN_CACHE_DATABASE only), never MONGODB_URI.
    # Off until the Atlas user and index exist (README "Semantic plan cache"); off = no successful-plan caching at all.
    plan_cache_enabled: bool = False
    plan_cache_mongodb_uri: str | None = None
    plan_cache_database: str = "sal_staging_vector"
    plan_cache_collection: str = "residents_ai"
    plan_cache_vector_index: str = "query_plan_embedding_index"
    # EXPERIMENTAL cosine threshold for NOMINATING candidates. Not a safety boundary: BGE-small scores opposite
    # questions ("older than 75" / "younger than 75": 0.984) above true paraphrases (0.915–0.927); safety comes from
    # slot shapes, logic words and check_plan. Calibrate with tests/test_semantic_cache.py on real questions.
    plan_cache_similarity_threshold: float = Field(default=0.88, ge=0.5, le=1.0)
    plan_cache_top_k: int = Field(default=5, ge=1, le=20)  # candidates returned by $vectorSearch
    plan_cache_max_candidates: int = Field(default=50, ge=1, le=1000)  # $vectorSearch numCandidates (ANN breadth)
    plan_cache_timeout_seconds: float = Field(default=2.0, gt=0, le=30)  # embedding / Atlas call; over → cache miss
    # Each template is deleted automatically this long after it was FIRST stored (MongoDB TTL index on expires_at; the
    # server's TTL monitor runs about once a minute, and lookups skip expired templates themselves). Reuse does not extend it.
    plan_cache_ttl_hours: float = Field(default=12.0, gt=0, le=24 * 365)
    plan_cache_store_templates: bool = True  # store checked, executed plans as reusable templates (re-validated on reuse)

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

    @model_validator(mode="after")
    def _plan_cache_config(self) -> "Settings":
        """Fail at startup, with a clear message, when the semantic cache is enabled but not safely configured."""
        if not self.plan_cache_enabled:
            return self
        if not self.plan_cache_mongodb_uri:
            raise ValueError("PLAN_CACHE_ENABLED=true needs PLAN_CACHE_MONGODB_URI (a dedicated, restricted credential)")
        if self.mongodb_uri and self.plan_cache_mongodb_uri.strip() == self.mongodb_uri.strip():
            raise ValueError("PLAN_CACHE_MONGODB_URI must not reuse MONGODB_URI: create a user with readWrite on "
                             "PLAN_CACHE_DATABASE only")  # fmt: skip
        if self.plan_cache_database == self.mongodb_db:
            raise ValueError("PLAN_CACHE_DATABASE must not be the resident database (MONGODB_DB)")
        for name in ("plan_cache_database", "plan_cache_collection", "plan_cache_vector_index"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name.upper()} must not be empty")
        return self

    @model_validator(mode="after")
    def _bedrock_config(self) -> "Settings":
        """LLM_PROVIDER=bedrock without a key fails at startup with a clear message (never a silent switch to Qwen)."""
        if self.llm_enabled and self.llm_provider == "bedrock":
            if self.bedrock_api_key is None or not self.bedrock_api_key.get_secret_value().strip():
                raise ValueError("LLM_PROVIDER=bedrock needs BEDROCK_API_KEY (or AWS_BEARER_TOKEN_BEDROCK)")
            if not self.bedrock_model_id.strip() or not self.bedrock_region.strip():
                raise ValueError("LLM_PROVIDER=bedrock needs BEDROCK_MODEL_ID and BEDROCK_REGION")
        return self

    @property
    def bedrock_endpoint(self) -> str:
        return (self.bedrock_base_url or f"https://bedrock-mantle.{self.bedrock_region}.api.aws/openai/v1").rstrip("/")

    @property
    def model_version_tag(self) -> str:
        """The active model's identity, for cache compatibility. Unchanged for local: '<LLM_MODEL_NAME>:<template kwargs>'."""
        if self.llm_provider == "bedrock":
            return f"bedrock:{self.bedrock_region}:{self.bedrock_model_id}"
        return f"{self.llm_model_name}:{json.dumps(self.llm_chat_template_kwargs, sort_keys=True)}"

    def today(self) -> str:
        return datetime.now(ZoneInfo(self.app_timezone)).date().isoformat()


@lru_cache
def get_settings() -> Settings:
    return Settings()
