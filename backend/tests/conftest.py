import copy

import pytest

from app.config import Settings
from app.data.memory_repository import MemoryResidentRepository
from app.engine.service import QueryService
from app.nlu.translator import TokenUsage, Translation, TranslatorUnavailable

TODAY = "2026-11-09"


class FakeTranslator:
    """Stands in for Claude: returns a fixed plan per masked question and records every call."""

    def __init__(self, plans: dict[str, dict] | None = None, fail: bool = False):
        self.plans = {k.lower(): v for k, v in (plans or {}).items()}
        self.fail = fail
        self.calls: list[dict] = []

    async def translate(self, masked_question, today, wings, previous_plan_json):
        self.calls.append({"question": masked_question, "today": today, "wings": wings, "previous": previous_plan_json})
        if self.fail:
            raise TranslatorUnavailable("connection")
        plan = self.plans.get(masked_question.lower())
        if plan is None:
            raise AssertionError(f"no fake plan for masked question: {masked_question!r}")
        return Translation(
            plan=copy.deepcopy(plan),
            usage=TokenUsage(input_tokens=120, output_tokens=40, cache_read_input_tokens=600),
            model="fake",
        )


class PinnedDateSettings(Settings):
    """Settings with "today" fixed, so relative-date behaviour is reproducible."""

    def today(self) -> str:
        return TODAY


@pytest.fixture
def settings() -> Settings:
    return PinnedDateSettings(_env_file=None, environment="test", anthropic_api_key=None, rate_limit_per_minute=1000)


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
