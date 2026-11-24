"""Requested display fields: equivalent wordings give the same fields, conditions are never mistaken for display
requests, and grounding stays strict both ways (a requested field may not be dropped, an unrequested one not added)."""

import pytest
import yaml

from app.nlu.intent_router import DEFAULT_ROUTES_FILE
from app.nlu.projection import ProjectionDetector
from tests.conftest import FakeTranslator

DETECTOR = ProjectionDetector(yaml.safe_load(open(DEFAULT_ROUTES_FILE, encoding="utf-8")))


def shown(question: str) -> set[str]:
    return {m.attribute for m in DETECTOR.scan(question.lower()) if m.attribute}


@pytest.mark.parametrize(
    ("question", "fields"),
    [
        # the entity word as a modifier or possessive between the cue and the attribute
        ("Include the resident names and room numbers in each status group.", {"room_number"}),
        ("Display each resident's name and room.", {"room_number"}),
        ("Show the residents' room numbers.", {"room_number"}),
        ("Give me each resident's room number and care level.", {"room_number", "care_type"}),
        ("Show resident names, room numbers and caregivers.", {"room_number", "primary_caregiver"}),
        ("List memory care residents with the residents' rooms and ages.", {"room_number", "age"}),
        # wordings that already worked: unchanged
        ("Show residents with their room numbers.", {"room_number"}),
        ("List the names along with their room numbers.", {"room_number"}),
        ("List active residents and their caregivers.", {"primary_caregiver"}),
        ("List residents including room and age.", {"room_number", "age"}),
    ],
)
def test_equivalent_display_requests_give_the_same_fields(question, fields):
    assert shown(question) == fields


@pytest.mark.parametrize(
    "question",
    [
        "How many residents are in room 101?",  # a value: condition
        "List residents whose room numbers start with A.",
        "List residents with room numbers starting with A.",  # a text-match condition, not "show room numbers"
        "List residents with a facility containing Garden.",
        "List residents in memory care admitted last month.",
        "Show the resident count by status.",
        "List the residents in each facility.",
        "Which resident has the most visitors?",
        "List the residents.",
    ],
)
def test_conditions_and_plain_questions_request_no_display_field(question):
    assert shown(question) == set()


# ------------------------------------------------------------------------------ grounding, through the real service

Q2 = ("Among residents admitted after August 1, 2026, show the count of Active, Discharged, and Away residents "
      "separately. Include the resident names and room numbers in each status group.")  # fmt: skip
Q2_PLAN = {"action": "breakdown", "filters": {"admission_date": {"start": "2026-08-02"}}, "group_by": ["status"],
           "fields": ["room_number"]}  # fmt: skip  (Gemma 4 31B's live plan, 2026-11-19)


async def test_q2_correct_plan_is_accepted(make_service):
    fake = FakeTranslator({Q2: Q2_PLAN})
    r = await (await make_service(fake)).handle(Q2)
    assert len(fake.calls) == 1 and not r["response"].get("needsClarification")
    assert r["tier"] == "llm" and r["mcpCall"] is not None


async def test_q2_plan_that_drops_the_requested_rooms_is_caught(make_service):
    dropped = {k: v for k, v in Q2_PLAN.items() if k != "fields"}
    fake = FakeTranslator({Q2: [dropped, dropped]})
    r = await (await make_service(fake)).handle(Q2)
    assert len(fake.calls) == 2 and "show room number" in (fake.calls[1]["hint"] or "")
    assert r["response"]["needsClarification"]  # never answered without the requested field


async def test_q2_plan_that_adds_an_unrequested_field_is_caught(make_service):
    extra = {**Q2_PLAN, "fields": ["room_number", "age"]}
    fake = FakeTranslator({Q2: [extra, Q2_PLAN]})
    r = await (await make_service(fake)).handle(Q2)
    assert len(fake.calls) == 2 and "show age" in (fake.calls[1]["hint"] or "")
    assert not r["response"].get("needsClarification")


async def test_named_caregiver_is_a_condition_not_a_display_request(make_service):
    q = "List residents with caregiver Sarah Jenkins."
    plan = {"action": "list", "filters": {"primary_caregiver": {"include": ["person_1"]}}}  # the model reads person_N
    fake = FakeTranslator({q.replace("Sarah Jenkins", "person_1"): plan})
    r = await (await make_service(fake)).handle(q)
    assert len(fake.calls) <= 1 and not r["response"].get("needsClarification")


async def test_text_match_on_rooms_needs_no_room_column(make_service):
    q = 'List residents with room numbers starting with "SN".'
    fake = FakeTranslator({q: {"action": "list", "filters": {"room_number": {"starts_with": ["SN"]}}}})
    r = await (await make_service(fake)).handle(q)
    assert len(fake.calls) == 1 and r["response"]["count"] == 13


async def test_multiple_fields_requested_by_possessive_entity(make_service):
    q = "List active residents with each resident's room number and care level."
    plan = {"action": "list", "filters": {"status": {"include": ["active"]}}, "fields": ["room_number", "care_type"]}
    fake = FakeTranslator({q: plan})
    r = await (await make_service(fake)).handle(q)
    assert not r["response"].get("needsClarification") and len(fake.calls) <= 1
