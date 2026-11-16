"""Grounding check (shared validator): preserved filters, invented filters, deterministic date/age canonicalisation."""

from datetime import date

import pytest
import yaml

from app.data.memory_repository import MemoryResidentRepository
from app.engine.grounding import check_plan, scan_question
from app.models.plan import QueryPlan
from app.nlu.intent_router import DEFAULT_ROUTES_FILE
from app.nlu.vocabulary import EntityIndex

TODAY = date(2026, 11, 9)
ANCHORS = yaml.safe_load(DEFAULT_ROUTES_FILE.read_text())["direct"]["date_fields"]


@pytest.fixture
async def index():
    return EntityIndex(await MemoryResidentRepository().vocabulary())


def ground(index, question, plan: dict, previous=None):
    mask = index.mask(question)
    return check_plan(
        QueryPlan.model_validate(plan), question, mask, index, scan_question(mask.text, TODAY), ANCHORS, previous
    )


async def test_complete_plan_passes(index):
    g = ground(
        index,
        "How many active female residents are in memory care?",
        {
            "action": "count",
            "filters": {
                "status": {"include": ["active"]},
                "gender": {"include": ["Female"]},
                "care_type": {"include": ["memory_care"]},
            },
        },
    )
    assert g.ok and not g.fixes


async def test_dropped_filter_is_missing(index):
    g = ground(
        index,
        "How many active female residents are in memory care?",
        {"action": "count", "filters": {"status": {"include": ["active"]}, "gender": {"include": ["Female"]}}},
    )
    assert not g.ok and g.missing == ['care type "memory care"']


@pytest.mark.parametrize(
    ("question", "filters", "invented"),
    [
        ("How many residents are active?", {"status": {"include": ["active"]}, "gender": {"include": ["Male"]}}, "gender Male"),
        ("How many residents are active?", {"status": {"include": ["active"]}, "age": {"min": 80}}, "age range"),
        (
            "How many residents are active?",
            {"status": {"include": ["active"]}, "admission_date": {"start": "2026-10-01"}},
            "admission date",
        ),
        ("How many residents are active?", {"status": {"include": ["active"]}, "wing": {"include": ["North Wing"]}}, "facility"),
    ],
)
async def test_invented_filters_are_rejected(index, question, filters, invented):
    g = ground(index, question, {"action": "count", "filters": filters})
    assert not g.ok and any(invented in x for x in g.invented), g.invented


async def test_synonyms_justify_values(index):
    g = ground(
        index,
        "how many dementia patients are women",
        {"action": "count", "filters": {"care_type": {"include": ["memory_care"]}, "gender": {"include": ["Female"]}}},
    )
    assert g.ok


async def test_date_bounds_are_canonicalised_from_the_question(index):
    # the model got the calendar arithmetic wrong; the deterministic resolution wins
    g = ground(
        index,
        "How many residents were admitted last month?",
        {"action": "count", "filters": {"admission_date": {"start": "2026-10-09", "end": "2026-11-09"}}},
    )
    assert g.ok and g.plan.filters.admission_date.start == date(2026, 10, 1)
    assert g.plan.filters.admission_date.end == date(2026, 10, 31)


async def test_date_field_follows_the_verb(index):
    g = ground(
        index,
        "how many residents were discharged last month",
        {
            "action": "count",
            "filters": {
                "status": {"include": ["discharged"]},
                "admission_date": {"start": "2026-10-01", "end": "2026-10-31"},
            },
        },
    )
    assert g.ok and g.plan.filters.discharge_date is not None and g.plan.filters.admission_date is None


async def test_admitted_with_a_date_is_not_the_status(index):
    plan = {"action": "count", "filters": {"admission_date": {"start": "2026-10-01", "end": "2026-10-31"}}}
    assert ground(index, "How many residents were admitted last month?", plan).ok


async def test_dropped_date_is_missing(index):
    g = ground(index, "How many residents were admitted last month?", {"action": "count"})
    assert not g.ok and g.missing and "last month" in g.missing[0]


async def test_age_bounds_are_canonicalised(index):
    g = ground(index, "how many residents are over 80", {"action": "count", "filters": {"age": {"min": 80}}})
    assert g.ok and g.plan.filters.age.min == 81


async def test_follow_up_may_keep_previous_conditions(index):
    previous = {"status": {"include": ["active"]}}
    plan = {"action": "count", "filters": {"status": {"include": ["active"]}, "care_type": {"include": ["memory_care"]}}}
    assert ground(index, "and how many of them are in memory care?", plan, previous).ok
    assert not ground(index, "and how many of them are in memory care?", plan, None).ok


async def test_unknown_resident_in_lookup_is_rejected(index):
    g = ground(index, "Tell me about Eleanor Vance", {"action": "lookup", "lookup": {"residents": ["RES-1002"]}})
    assert not g.ok and any("Arthur Pendleton" in x for x in g.invented)


async def test_non_data_plans_are_not_checked(index):
    assert ground(index, "hi", {"action": "help"}).ok


def test_out_of_scope_topics_are_detected():
    from app.engine.grounding import out_of_scope_terms

    assert out_of_scope_terms("active residents with diabetes") == ["diabetes"]
    assert out_of_scope_terms("What medications is resident_1 taking?") == ["medications"]
    assert out_of_scope_terms("how many active residents are in memory care") == []


@pytest.mark.parametrize(
    ("question", "plan", "check"),
    [
        # wrong sort direction → canonicalised from "oldest"; number → limit
        ("show me the three oldest residents", {"action": "list", "sort": {"field": "age", "direction": "asc"}, "limit": 3},
         lambda p: (p.sort.field, p.sort.direction, p.limit) == ("age", "desc", 3)),
        # aggregate op follows the question's word
        ("minimum age in respite care", {"action": "aggregate", "aggregate": {"op": "avg"}, "filters": {"care_type": {"include": ["respite_care"]}}},
         lambda p: p.aggregate.op == "min"),
        # an ordering/limit nobody asked for is dropped
        ("list the active residents", {"action": "list", "filters": {"status": {"include": ["active"]}}, "sort": {"field": "created_date", "direction": "asc"}, "limit": 10},
         lambda p: p.sort is None and p.limit is None),
    ],
)
async def test_structure_is_canonicalised(index, question, plan, check):
    g = ground(index, question, plan)
    assert g.ok and check(g.plan), (g.problems(), g.plan)


@pytest.mark.parametrize(
    ("question", "plan"),
    [
        ("average age per wing", {"action": "aggregate", "aggregate": {"op": "avg"}}),  # dropped grouping
        ("caregiver breakdown of residents", {"action": "breakdown", "group_by": ["gender"]}),  # wrong grouping
        ("minimum age of residents in respite care", {"action": "list", "filters": {"care_type": {"include": ["respite_care"]}}}),
        ("how many residents are active", {"action": "aggregate", "aggregate": {"op": "avg"}, "filters": {"status": {"include": ["active"]}}}),
    ],
)
async def test_wrong_structure_is_rejected(index, question, plan):
    assert not ground(index, question, plan).ok


async def test_recency_question_cannot_carry_a_guessed_date_or_status(index):
    plan = {
        "action": "list",
        "filters": {"status": {"include": ["admitted"]}, "admission_date": {"start": "2026-11-09"}},
        "sort": {"field": "admission_date", "direction": "desc"},
        "limit": 1,
    }
    g = ground(index, "who was admitted most recently", plan)
    assert not g.ok and any("status admitted" in x for x in g.invented) and any("admission date" in x for x in g.invented)


async def test_a_bare_number_does_not_justify_a_date_range(index):
    plan = {
        "action": "list",
        "filters": {"care_type": {"include": ["assisted_living"]}, "admission_date": {"start": "2026-11-09", "end": "2026-11-09"}},
        "sort": {"field": "admission_date", "direction": "desc"},
        "limit": 5,
    }
    g = ground(index, "the 5 most recently admitted residents in assisted living", plan)
    assert not g.ok and any("admission date" in x for x in g.invented)


async def test_follow_up_must_keep_the_previous_conditions(index):
    previous = {"status": {"include": ["active"]}, "care_type": {"include": ["memory_care"]}}
    dropped = {"action": "count", "filters": {"gender": {"include": ["Female"]}}}
    kept = {"action": "count", "filters": {**previous, "gender": {"include": ["Female"]}}}
    q = "and how many of them are female?"
    g = ground(index, q, dropped, previous)
    assert not g.ok and any("earlier condition on status" in m for m in g.missing)
    assert ground(index, q, kept, previous).ok
    # no previous question in this conversation: "them" refers to nothing → clarify, never guess
    assert not ground(index, q, dropped, None).ok


async def test_yes_no_question_needs_the_condition_not_just_the_attribute(index):
    q = "does Arthur Pendleton receive memory care"
    as_attribute = {"action": "lookup", "lookup": {"residents": ["RES-1002"], "attributes": ["care_type"]}}
    as_check = {**as_attribute, "filters": {"care_type": {"include": ["memory_care"]}}}
    assert not ground(index, q, as_attribute).ok
    assert ground(index, q, as_check).ok
    # asking for the attribute without naming a value is fine
    assert ground(index, "what is Arthur Pendleton's care type", as_attribute).ok



async def test_exclusion_needs_a_negation_in_the_question(index):
    # thinking mode produced exactly this for "the 5 oldest active residents"
    bad = {
        "action": "list",
        "filters": {"status": {"exclude": ["active"]}, "age": {"max": 18}},
        "sort": {"field": "age", "direction": "desc"},
        "limit": 5,
    }
    g = ground(index, "the 5 oldest active residents", bad)
    assert not g.ok
    assert any("exclusion of status active" in x for x in g.invented) and "age range" in g.invented


async def test_negated_question_needs_an_exclusion(index):
    q = "How many residents are not active?"
    assert not ground(index, q, {"action": "count", "filters": {"status": {"include": ["active"]}}}).ok
    assert ground(index, q, {"action": "count", "filters": {"status": {"exclude": ["active"]}}}).ok
    assert ground(index, "how many residents aren't in memory care",
                  {"action": "count", "filters": {"care_type": {"exclude": ["memory_care"]}}}).ok


async def test_a_bare_number_is_not_an_age_range(index):
    plan = {"action": "list", "filters": {"age": {"min": 5}}, "sort": {"field": "age", "direction": "desc"}, "limit": 5}
    assert "age range" in ground(index, "the 5 oldest residents", plan).invented


async def test_resident_name_word_is_not_a_missing_caregiver():
    """Regression: resident "Allison Akilah" + a staff member surnamed Allison → the correct lookup was rejected as
    "missing caregiver Allison", fell through to the model, which then reused the previous resident."""
    from app.data.repository import Vocabulary

    idx = EntityIndex(
        Vocabulary(
            residents=[{"residentId": "RES-2001", "firstName": "Allison", "lastName": "Akilah"}],
            caregivers=["Nurse Bob Allison, RN"],
            statuses=["active"],
        )
    )
    q = "How old is Allison Akilah?"
    mask = idx.mask(q)
    assert list(mask.placeholders.values()) == [["RES-2001"]]
    plan = QueryPlan.model_validate({"action": "lookup", "lookup": {"residents": ["RES-2001"], "attributes": ["age"]}})
    g = check_plan(plan, q, mask, idx, scan_question(mask.text, TODAY), ANCHORS, None)
    assert g.ok, g.problems()
    # the staff member is still caught when actually named as a caregiver
    q2 = "How many residents does Bob Allison look after?"
    m2 = idx.mask(q2)
    g2 = check_plan(QueryPlan(action="count"), q2, m2, idx, scan_question(m2.text, TODAY), ANCHORS, None)
    assert not g2.ok and any("Allison" in x for x in g2.missing)
