"""Reference resolution: pronouns within the current question vs references to the previous conversation turn."""

import pytest
import yaml

from app.nlu.intent_router import DEFAULT_ROUTES_FILE
from app.nlu.references import ReferenceResolver
from tests.conftest import FakeTranslator
from tests.test_direct_query import build, make_service

RESOLVER = ReferenceResolver(yaml.safe_load(DEFAULT_ROUTES_FILE.read_text()))


@pytest.mark.parametrize(
    ("question", "has_previous", "previous", "local"),
    [
        ("List active residents and show their rooms.", True, False, 1),  # their → residents (this question)
        ("List residents admitted in August and show their caregivers", True, False, 1),
        ("How many of them are female?", True, True, 0),  # them → the previous answer
        ("What about their rooms?", True, True, 0),  # continuation + pronoun without antecedent
        ("And how many of these residents are in memory care?", True, True, 0),  # demonstrative → previous answer
        ("Who is resident_1's caregiver and what is her room?", True, False, 1),  # her → resident_1
        ("Is she in memory care?", True, True, 0),
        ("How many residents are active?", True, False, 0),  # no reference at all
        ("list those residents admitted in august", False, False, 0),  # fresh chat: part of this question
        ("and how many are in memory care", False, False, 0),  # opener means nothing without a previous turn
    ],
)
def test_resolution(question, has_previous, previous, local):
    r = RESOLVER.resolve(question, has_previous)
    assert (r.refers_to_previous, len(r.local_spans)) == (previous, local)


def test_reference_without_a_previous_turn_is_ambiguous():
    r = RESOLVER.resolve("How many of them are female?", has_previous=False)
    assert r.refers_to_previous and r.ambiguous(has_previous=False) and r.trigger == "them"


# ---------------------------------------------------------------- fast path
async def test_local_pronoun_is_understood_by_the_fast_path(builder, index):
    r = build(builder, index, "Who is Eleanor Vance's caregiver and what is her room?", "lookup")
    assert r.plan == {
        "action": "lookup",
        "lookup": {"residents": ["RES-1001"], "attributes": ["primary_caregiver", "room_number"]},
    }


async def test_previous_turn_reference_goes_to_the_model(builder, index):
    r = builder.build(*_args(index, "How many of them are female?"), has_previous=True)
    assert r.plan is None and r.reason == "refers_to_previous"


def _args(index, q):
    from datetime import date

    from app.nlu.intent_router import IntentScore

    return q, index.mask(q), IntentScore("count", 0.95, 0.2, {"count": 0.95}), index, date(2026, 11, 9)


# ---------------------------------------------------------------- service: conversation context
ACTIVE_AL = {"action": "count", "filters": {"status": {"include": ["active"]}, "care_type": {"include": ["assisted_living"]}}}


async def test_follow_up_keeps_the_previous_filters():
    keeps = {"action": "count", "filters": {**ACTIVE_AL["filters"], "gender": {"include": ["Male"]}}, "uses_previous": True}
    fake = FakeTranslator({"and how many of them are male?": keeps})
    service = await make_service({"count": 0.93, "list": 0.80}, fake)
    first = await service.handle("How many active residents are in assisted living?", session_id="s1")
    assert first["tier"] == "direct"
    second = await service.handle("and how many of them are male?", session_id="s1")
    assert second["tier"] == "llm" and second["intent"]["category"] == "RESIDENT_COUNT"
    assert "Gender: Male" in second["interpretation"] and "Status: Active" in second["interpretation"]


async def test_follow_up_that_drops_previous_filters_is_not_answered():
    drops = {"action": "count", "filters": {"gender": {"include": ["Male"]}}}
    fake = FakeTranslator({"and how many of them are male?": drops})
    service = await make_service({"count": 0.93, "list": 0.80}, fake)
    await service.handle("How many active residents are in assisted living?", session_id="s1")
    second = await service.handle("and how many of them are male?", session_id="s1")
    assert second["response"].get("needsClarification") and second["mcpCall"] is None


async def test_local_pronoun_never_inherits_previous_filters():
    """The reported failure: "their" in a self-contained question pulled the previous question's caregiver condition."""
    fake = FakeTranslator({})
    service = await make_service({"count": 0.93, "list": 0.80, "lookup": 0.93}, fake)
    await service.handle("How many residents does Care Aide Elena Rostova look after?", session_id="s1")
    r = await service.handle("Who is Eleanor Vance's caregiver and what is her room?", session_id="s1")
    assert r["tier"] == "direct" and r["modelCalls"] == 0 and fake.calls == []
    assert r["intent"]["parameters"].get("filters", {}) == {}  # no caregiver FILTER carried over from turn 1


async def test_reference_without_conversation_asks_who_is_meant():
    fake = FakeTranslator({"How many of them are female?": {"action": "count", "filters": {"gender": {"include": ["Female"]}}}})
    service = await make_service({"count": 0.93, "list": 0.80}, fake)
    r = await service.handle("How many of them are female?", session_id="fresh")
    assert r["response"].get("needsClarification") and "'them'" in r["response"]["summaryText"]
    assert r["mcpCall"] is None  # nothing executed on a guess


@pytest.fixture
async def index():
    from app.data.memory_repository import MemoryResidentRepository
    from app.nlu.vocabulary import EntityIndex

    return EntityIndex(await MemoryResidentRepository().vocabulary())


@pytest.fixture
def builder():
    from app.nlu.direct_query import DirectQueryBuilder
    from app.nlu.intent_router import SemanticIntentRouter

    return DirectQueryBuilder(SemanticIntentRouter().config)
