"""Regressions from staff questions that failed on real data (questions 21–30 of the staff test round).

Each test pins a GENERAL rule, with the test data or a small synthetic vocabulary — never a real resident or a
question-specific condition:
  • dates: day dates written out, a bare year after a comparison word, exclusive "before"/"after";
  • names: a name followed by punctuation, staff recorded under a single word, facility names with a bracketed tag;
  • columns: a list of shown attributes that starts with the resident's name, a display verb before "their";
  • questions the plan format cannot express are answered with a clear message, without the model.
"""

import pytest

from app.data.repository import Vocabulary
from app.nlu.vocabulary import EntityIndex
from tests.conftest import FakeTranslator
from tests.test_direct_query import build


@pytest.fixture
async def index():
    from app.data.memory_repository import MemoryResidentRepository

    return EntityIndex(await MemoryResidentRepository().vocabulary())


@pytest.fixture
def builder():
    from app.nlu.direct_query import DirectQueryBuilder
    from app.nlu.intent_router import SemanticIntentRouter

    return DirectQueryBuilder(SemanticIntentRouter().config)


def _vocab(**kw) -> EntityIndex:
    return EntityIndex(Vocabulary(**kw))


# ---------------------------------------------------------------- names
def test_name_at_end_of_sentence_is_still_inside_the_care_type():
    """ "Skilled Nursing." is the care type, not the staff member "Director Nursing" (trailing full stop)."""
    idx = _vocab(caregivers=["Director Nursing"], care_types=["skilled_nursing"])
    assert idx.mentioned_caregivers("Find residents older than 90 who belong to Skilled Nursing.") == {}


def test_staff_recorded_under_one_word_are_recognised_as_names():
    idx = _vocab(caregivers=["Vik", "Jerry", "Dr. Michael Chen"])
    q = "Find residents who have Vik or Jerry listed as their primary caregiver."
    assert idx.mentioned_caregivers(q) == {"Vik": ["Vik"], "Jerry": ["Jerry"]}
    # only when written as a name: not lower-case, not the sentence's first word
    assert idx.mentioned_caregivers("residents of vik") == {}
    assert idx.mentioned_caregivers("Jerry rigged question about residents") == {}


def test_one_word_staff_name_is_not_masked_as_a_resident():
    """A word that is both a resident's surname and a staff record stays visible as the caregiver."""
    idx = _vocab(
        caregivers=["Vik"],
        residents=[{"residentId": "RES-9001", "firstName": "Jon", "lastName": "Vik"}],
    )
    assert idx.mask("Find residents who have Vik as their caregiver").placeholders == {}


# ---------------------------------------------------------------- fast path: the failed questions' shapes
async def test_day_date_with_exclusive_after(builder, index):
    r = build(builder, index, "Find all residents admitted after August 1, 2026, who are currently Active.", "list")
    assert r.plan["filters"] == {"status": {"include": ["active"]}, "admission_date": {"start": "2026-08-02"}}


async def test_bare_year_before(builder, index):
    r = build(builder, index, "Find all residents who are Active and have an admission date before 2026.", "list")
    assert r.plan["filters"] == {"status": {"include": ["active"]}, "admission_date": {"end": "2025-12-31"}}


async def test_columns_after_along_with_their(builder, index):
    q = "List all residents who are currently Discharged, along with their admission dates and room numbers."
    r = build(builder, index, q, "list")
    assert r.plan == {
        "action": "list",
        "filters": {"status": {"include": ["discharged"]}},
        "fields": ["admission_date", "room_number"],
    }


async def test_column_list_starting_with_name(builder, index):
    q = "Show the name, age, room number and admission date of every resident admitted in June 2026."
    r = build(builder, index, q, "list")
    assert r.plan == {
        "action": "list",
        "filters": {"admission_date": {"start": "2026-06-01", "end": "2026-06-30"}},
        "fields": ["age", "room_number", "admission_date"],  # the name is always shown: no extra field
    }


async def test_age_and_care_type_with_belong_to(builder, index):
    r = build(builder, index, "Find all residents older than 90 years who belong to Skilled Nursing.", "list")
    assert r.plan["filters"] == {"care_type": {"include": ["skilled_nursing"]}, "age": {"min": 91}}


# ---------------------------------------------------------------- grounding of model plans
async def test_inconsistent_model_date_bounds_are_rejected_without_execution():
    """The model read "before 2026" as "during 2026"; the inconsistent plan must not execute."""
    from tests.test_direct_query import make_service

    q = "how many residents were admitted before 2026?"
    fake = FakeTranslator({q: {"action": "count", "filters": {"admission_date": {"start": "2026-01-01"}}}})
    service = await make_service({"count": 0.50, "list": 0.40}, fake)  # router unsure: the model plans
    r = await service.handle(q)
    assert r["tier"] == "llm"
    assert r["response"]["needsClarification"] is True
    assert r["mcpCall"] is None


async def test_model_placeholder_for_an_unnamed_resident_is_rejected(make_service):
    """No resident is named, yet the model looked up "resident_1": never executed."""
    q = "Find residents who have Nurse Sarah Jenkins listed as their caregiver"
    bad = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["primary_caregiver"]}}
    fake = FakeTranslator({q.replace("Sarah Jenkins", "person_1"): [bad, bad]})  # the model reads person_N
    service = await make_service(fake)
    r = await service.handle(q)
    assert r["response"].get("needsClarification") and r["mcpCall"] is None


# ---------------------------------------------------------------- unsupported: comparing residents with each other
async def test_comparison_between_residents_is_answered_without_the_model(make_service):
    fake = FakeTranslator({})
    service = await make_service(fake)  # no router at all: the capability check still runs
    r = await service.handle("List all residents who share the same room number, including their names and IDs.")
    assert fake.calls == [] and r["modelCalls"] == 0 and r["mcpCall"] is None
    assert r["response"]["needsClarification"] and "isn't supported yet" in r["response"]["summaryText"]
    assert r["fallbackReason"].startswith("unsupported:comparison")


def test_function_word_before_a_name_is_not_part_of_the_name():
    """ "as" is a substring of "aakash": it must not be masked together with the name ("same room as <name>")."""
    idx = _vocab(residents=[{"residentId": "R1", "firstName": "Bansal", "lastName": "Aakash"},
                            {"residentId": "R2", "firstName": "Ann", "lastName": "Lee"}])  # fmt: skip
    m = idx.mask("residents in the same room as Bansal Aakash")
    assert m.text == "residents in the same room as resident_1" and m.originals == {"resident_1": "Bansal Aakash"}
    assert idx.mask("what is Ann Lee's room").originals == {"resident_1": "Ann Lee"}  # short real names still match
