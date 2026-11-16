import copy

import pytest

from app.config import Settings
from app.data.memory_repository import MemoryResidentRepository
from app.engine.service import QueryService
from app.nlu.translator import ModelUsage, Translation, TranslationRequest, TranslatorUnavailable

TODAY = "2026-11-09"


class FakeTranslator:
    """Stands in for the local Qwen model: returns a fixed plan per masked question and records every call."""

    provider = "fake"
    model_name = "fake-local-model"

    def __init__(self, plans: dict[str, dict] | None = None, fail: bool = False):
        self.plans = {k.lower(): v for k, v in (plans or {}).items()}
        self.fail = fail
        self.calls: list[dict] = []

    async def health(self) -> bool:
        return not self.fail

    async def translate(self, request: TranslationRequest):
        masked_question = request.masked_question
        self.calls.append(
            {
                "question": masked_question,
                "today": request.today,
                "wings": request.context.wings,
                "previous": request.previous_plan_json,
                "hint": request.hint,
                "dates": request.date_hints,
            }
        )
        if self.fail:
            raise TranslatorUnavailable("connection")
        plan = self.plans.get(masked_question.lower())
        if isinstance(plan, list):  # successive answers for the same question (e.g. a retry)
            plan = plan.pop(0) if len(plan) > 1 else plan[0]
        if plan is None:
            raise AssertionError(f"no fake plan for masked question: {masked_question!r}")
        return Translation(
            plan=copy.deepcopy(plan),
            usage=ModelUsage(calls=1, prompt_tokens=700, completion_tokens=40, inference_ms=5),
            model="fake",
        )


class PinnedDateSettings(Settings):
    """Settings with "today" fixed, so relative-date behaviour is reproducible."""

    def today(self) -> str:
        return TODAY


@pytest.fixture
def settings() -> Settings:
    return PinnedDateSettings(
        _env_file=None,
        environment="test",
        rate_limit_per_minute=1000,
        topic_gate_enabled=False,
        semantic_router_enabled=False,
    )


@pytest.fixture
def repo() -> MemoryResidentRepository:
    return MemoryResidentRepository()


@pytest.fixture
def make_service(repo, settings):
    async def _make(translator) -> QueryService:
        service = QueryService(repo, translator, settings)
        await service.refresh_vocabulary()
        return service

    return _make
