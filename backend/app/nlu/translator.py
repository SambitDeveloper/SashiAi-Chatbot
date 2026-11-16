"""QueryTranslator contract: masked question + permitted schema context → raw query-plan candidate (JSON dict).

The application depends only on this interface, never on a model runtime. The one implementation is the local
Qwen2.5-0.5B-Instruct model served by llama.cpp (app/nlu/qwen_translator.py). A candidate plan is never trusted:
it always goes through the resolver, Pydantic validation and the grounding checks before anything touches the data.
"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ModelUsage:
    """Local inference accounting (informational: local tokens have no per-token cost)."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    inference_ms: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "ModelUsage") -> "ModelUsage":
        return ModelUsage(
            self.calls + other.calls,
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.inference_ms + other.inference_ms,
        )


@dataclass
class SchemaContext:
    """Permitted schema metadata for one request, read from the live database vocabulary (never hard-coded)."""

    statuses: list[str]
    care_types: list[str]
    wings: list[str]  # only the wings/facilities relevant to THIS question (never the whole list: prompt size)
    caregivers: list[str] = field(default_factory=list)
    # A fixed real wing for the few-shot example, the same for every request so the prompt prefix stays cacheable.
    example_wing: str | None = None


@dataclass
class TranslationRequest:
    masked_question: str
    today: str
    context: SchemaContext
    previous_plan_json: str | None = None
    date_hints: list[str] = field(default_factory=list)  # e.g. 'last month' = 2026-10-01 to 2026-10-31
    hint: str | None = None  # completeness / grounding feedback on a retry


@dataclass
class Translation:
    plan: dict
    usage: ModelUsage
    model: str


class TranslatorUnavailable(Exception):
    """The model could not be reached or returned an unusable response (invalid JSON, timeout, overload)."""


class QueryTranslator(Protocol):
    provider: str
    model_name: str

    async def translate(self, request: TranslationRequest) -> Translation: ...
