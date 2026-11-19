"""Question brief: what the deterministic analysis understood, handed to the local model when the fast path gives up."""

from datetime import date

import pytest

from app.data.memory_repository import MemoryResidentRepository
from app.engine.service import QueryService
from app.nlu.brief import QuestionBrief, RouteGuess, describe_route
from app.nlu.direct_query import DirectQueryBuilder
from app.nlu.intent_router import IntentScore, SemanticIntentRouter
from app.nlu.prompt import build_user_message
from app.nlu.vocabulary import EntityIndex
from tests.conftest import FakeTranslator, PinnedDateSettings

PIN = date(2026, 11, 9)


@pytest.fixture
async def index():
    return EntityIndex(await MemoryResidentRepository().vocabulary())


@pytest.fixture
def builder():
    return DirectQueryBuilder(SemanticIntentRouter().config)


def describe(builder, index, question, scores=None):
    intent = None
    if scores:
        best = max(scores, key=scores.get)
        rest = sorted(scores.values(), reverse=True)
        intent = IntentScore(best, scores[best], rest[0] - (rest[1] if len(rest) > 1 else 0), scores)
    return builder.describe(question, index.mask(question), intent, index, PIN)


async def test_lookup_brief_has_resident_attribute_value_and_unknown_word(builder, index):
    b = describe(builder, index, "is Eleanor Vance already discharged ?", {"lookup": 0.89, "count": 0.80})
    assert b.residents == ["resident_1"]
    assert "discharge_date" in b.attributes
    assert b.values.get("status") == ["discharged"]
    assert b.unknown_words == ["already"]
    assert b.routes[0].name == "lookup"
    text = b.to_text()
    assert "likely action: lookup (0.89)" in text
    assert "yes/no question: lookup with the mentioned values as conditions" in text
    assert "Eleanor" not in text and "Vance" not in text  # names never in the brief


async def test_attribute_written_as_two_words_is_recognised(builder, index):
    b = describe(builder, index, "who is the care giver of Eleanor Vance")
    assert "primary_caregiver" in b.attributes
    assert "giver" not in b.unknown_words


async def test_unrecognised_name_is_flagged_for_lookups(builder, index):
    b = describe(builder, index, "who is the caregiver of Zz Qq", {"lookup": 0.76, "list": 0.75})
    assert b.residents == []
    text = b.to_text()
    assert "none found in the records (do not invent resident_N)" in text
    assert "likely action: lookup (0.76); or list" in text  # router's own ranking
    assert "yes/no" not in text  # a wh-question, and no values
    assert b.unknown_words == ["zz", "qq"]


async def test_dates_numbers_and_database_values_are_not_unknown(builder, index):
    b = describe(builder, index, "how many female residents over 80 in memory care admitted last month", {"count": 0.9})
    assert b.unknown_words == []
    assert b.values["gender"] == ["Female"] and b.values["care_type"] == ["memory_care"]


def test_route_description_uses_plan_terms():
    assert describe_route({"action": "breakdown", "group_by": ["status"]}) == "breakdown group_by status"
    assert describe_route({"action": "list", "sort": {"field": "age", "direction": "desc"}, "limit": 1}) == (
        "list sort age desc limit 1"
    )
    assert describe_route({"action": "aggregate", "op": "avg"}) == "aggregate avg age"


def test_empty_brief_adds_nothing():
    assert QuestionBrief().to_text() is None
    assert "Analysis" not in build_user_message("q", "2026-11-09", [], None)


def test_brief_is_placed_after_the_question():
    brief = QuestionBrief(routes=[RouteGuess("count", 0.9, {"action": "count"})]).to_text()
    msg = build_user_message("how many residents", "2026-11-09", [], None, brief=brief)
    assert msg.index("Question: how many residents") < msg.index("Analysis")


# ---- through the service ----------------------------------------------------------------------------------------


class BriefRecorder(FakeTranslator):
    def __init__(self, plans):
        super().__init__(plans)
        self.briefs: list[str | None] = []

    async def translate(self, request):
        self.briefs.append(request.brief)
        return await super().translate(request)


Q = "is Eleanor Vance already discharged ?"
PLAN = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["status"]}}


async def _service(repo, enabled):
    settings = PinnedDateSettings(
        _env_file=None, environment="test", topic_gate_enabled=False, semantic_router_enabled=False,
        llm_question_brief=enabled,
    )  # fmt: skip
    fake = BriefRecorder({"is resident_1 already discharged ?": PLAN})
    service = QueryService(repo, fake, settings, intent_router=SemanticIntentRouter())  # router not started
    await service.refresh_vocabulary()
    return service, fake


async def test_model_gets_the_brief(repo):
    service, fake = await _service(repo, True)
    result = await service.handle(Q)
    assert result["tier"] == "llm"
    assert fake.briefs and fake.briefs[0].startswith("Analysis")
    assert "residents named: resident_1" in fake.briefs[0] and "already" in fake.briefs[0]


async def test_brief_can_be_switched_off(repo):
    service, fake = await _service(repo, False)
    await service.handle(Q)
    assert fake.briefs and all(b is None for b in fake.briefs)  # first call and any retry


async def test_retry_gets_the_same_brief(repo):
    service, fake = await _service(repo, True)
    await service.handle(Q)
    assert len(fake.briefs) >= 1 and len(set(fake.briefs)) == 1
