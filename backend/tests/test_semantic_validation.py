"""Stage 1 semantic validation: a plan must answer THE question, not just contain valid conditions.

Negative cases from the questions 31–40 evaluation, written generically with the test data:
  • the model returns only the reference resident (a lookup) for a question about other residents;
  • the operation is not the one asked for;
  • a literal date appears where the question compares with another resident's date;
  • a resident is used as a caregiver (a planning error — not "resident not found");
  • the model's own clarify/unsupported wording is never shown;
  • comparison questions are answered "not supported yet" and never executed, with or without the pre-check.
"""

from datetime import date

import pytest
import yaml

from app.data.memory_repository import MemoryResidentRepository
from app.engine.grounding import check_plan, scan_question
from app.engine.resolver import resolve_plan
from app.nlu.intent_router import DEFAULT_ROUTES_FILE
from app.nlu.projection import ProjectionDetector
from app.nlu.references import ReferenceResolver
from app.nlu.structure import StructureAnalyzer
from app.nlu.vocabulary import EntityIndex
from tests.conftest import FakeTranslator

ROUTES = yaml.safe_load(DEFAULT_ROUTES_FILE.read_text())
STRUCTURE = StructureAnalyzer(ROUTES)
TODAY = date(2026, 11, 9)
DATE_ANCHORS = {f: [str(w) for w in ws] for f, ws in ROUTES["direct"]["date_fields"].items()}


@pytest.fixture
async def index():
    return EntityIndex(await MemoryResidentRepository().vocabulary())


def validate(index, question: str, raw: dict):
    mask = index.mask(question)
    outcome = resolve_plan(raw, index, mask, question)
    if outcome.problem:
        return outcome, None
    g = check_plan(
        outcome.plan, question, mask, index, scan_question(mask.text, TODAY), DATE_ANCHORS,
        reference_resolver=ReferenceResolver(ROUTES), projection_detector=ProjectionDetector(ROUTES),
        structure=STRUCTURE.analyse(mask.text),
    )  # fmt: skip
    return outcome, g


# ---------------------------------------------------------------- the model returns only the reference resident
async def test_lookup_of_the_reference_is_rejected(index):
    q = "How many residents were admitted on the same day as Eleanor Vance?"
    _, g = validate(index, q, {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["admission_date"]}})
    assert not g.ok
    assert any("compared with, not asked about" in x for x in g.invented)  # the reference is never the answer
    assert any("asks about many residents" in m for m in g.missing)  # the operation is about residents in general
    assert any("not supported yet" in b for b in g.blocking)  # Stage 1: comparisons are not executed at all


async def test_lookup_of_a_subject_is_still_accepted(index):
    _, g = validate(index, "What is Eleanor Vance's room number?",
                    {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["room_number"]}})  # fmt: skip
    assert g.ok


# ---------------------------------------------------------------- the operation must be the one asked for
@pytest.mark.parametrize(
    ("question", "action", "ok"),
    [
        ("Find all residents who are active", "count", False),  # which/find → list
        ("How many residents are active?", "list", False),  # how many → count
        ("How many residents are active?", "count", True),
        ("Which residents are active and how many are there?", "list", True),  # both asked: list carries the count
    ],
)
async def test_operation_must_match(index, question, action, ok):
    _, g = validate(index, question, {"action": action, "filters": {"status": {"include": ["active"]}}})
    assert g.ok is ok, g.problems()


# ---------------------------------------------------------------- invented dates in a comparison
async def test_literal_date_for_a_relative_date_is_rejected(index):
    """"The same month as <resident>" names no date: a literal date in the plan was invented."""
    q = "Which residents were admitted in the same month and year as Eleanor Vance?"
    _, g = validate(index, q, {"action": "list", "filters": {"admission_date": {"start": "2026-11-09"}}})
    assert not g.ok and any("admission date" in x for x in g.invented)


# ---------------------------------------------------------------- a resident used as a caregiver
async def test_resident_as_caregiver_is_a_planning_error_not_a_missing_record(index):
    q = "Who is the caregiver of Eleanor Vance?"
    outcome, _ = validate(index, q, {"action": "list", "filters": {"primary_caregiver": {"include": ["resident_1"]}}})
    assert outcome.problem == "role:primary_caregiver"
    assert "couldn't find" not in outcome.plan.message and "Eleanor Vance" in outcome.plan.message
    assert "is a resident, not a primary caregiver" in outcome.hint


async def test_unknown_caregiver_is_still_reported_as_not_found(index):
    outcome, _ = validate(index, "Residents of Nurse Nobody Atall",
                          {"action": "list", "filters": {"primary_caregiver": {"include": ["Nobody Atall"]}}})  # fmt: skip
    assert outcome.problem == "unknown" and "couldn't find" in outcome.plan.message


async def test_role_error_gets_one_retry_with_feedback(make_service):
    q = "Who is the caregiver of resident_1?"
    wrong = {"action": "list", "filters": {"primary_caregiver": {"include": ["resident_1"]}}}
    right = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["primary_caregiver"]}}
    fake = FakeTranslator({q: [wrong, right]})
    service = await make_service(fake)
    r = await service.handle("Who is the caregiver of Eleanor Vance?")
    assert len(fake.calls) == 2 and "is a resident" in fake.calls[1]["hint"]
    assert r["response"]["headline"].startswith("Nurse Sarah Jenkins")


# ---------------------------------------------------------------- the model's own wording is never shown
async def test_model_unsupported_text_is_replaced(make_service):
    q = "What does resident_1 prefer?"
    fake = FakeTranslator({q: {"action": "unsupported", "message": "resident_1 is not a valid resident reference."}})
    service = await make_service(fake)
    r = await service.handle("What does Eleanor Vance prefer?")
    text = r["response"]["summaryText"]
    assert "not a valid resident" not in text and "resident_1" not in text
    assert r["response"]["needsClarification"] and r["mcpCall"] is None


# ---------------------------------------------------------------- comparison questions: never executed in Stage 1
COMPARISON = "How many residents were admitted on the same day as Eleanor Vance?"


async def test_comparison_is_not_supported_without_a_model_call(make_service):
    fake = FakeTranslator({})
    service = await make_service(fake)
    r = await service.handle(COMPARISON)
    assert fake.calls == [] and r["mcpCall"] is None and r["fallbackReason"] == "unsupported:comparison:reference"
    assert "isn't supported yet" in r["response"]["summaryText"]


async def test_comparison_model_plan_is_rejected_when_the_precheck_is_off(make_service):
    """Model-only mode: even when the question reaches the model, the reference lookup it returns is not executed."""
    lookup = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["admission_date"]}}
    fake = FakeTranslator({"How many residents were admitted on the same day as resident_1?": lookup})
    service = await make_service(fake)
    service.precheck_comparisons = False
    r = await service.handle(COMPARISON)
    assert len(fake.calls) == 1  # blocking: no retry
    assert r["mcpCall"] is None and "isn't supported yet" in r["response"]["summaryText"]


async def test_ambiguous_reference_in_a_comparison_is_not_executed(make_service):
    fake = FakeTranslator({})
    service = await make_service(fake)
    r = await service.handle("Which residents were admitted before Roosevelt?")  # "Roosevelt" matches several
    assert fake.calls == [] and r["mcpCall"] is None and "isn't supported yet" in r["response"]["summaryText"]


async def test_comparison_among_residents_is_not_supported(make_service):
    fake = FakeTranslator({})
    service = await make_service(fake)
    r = await service.handle("List all residents who share the same room number")
    assert fake.calls == [] and r["fallbackReason"] == "unsupported:comparison:cross_record"


# ---------------------------------------------------------------- polarity and grouping (found by the benchmarks)
async def test_negation_applies_only_to_what_follows_it(index):
    q = "memory care residents not in Garden Terrace"
    right = {"care_type": {"include": ["memory_care"]}, "facility": {"exclude": ["Garden Terrace"]}}
    inverted = {"care_type": {"exclude": ["memory_care"]}, "facility": {"exclude": ["Garden Terrace"]}}
    assert validate(index, q, {"action": "list", "filters": right})[1].ok
    _, g = validate(index, q, {"action": "list", "filters": inverted})
    assert not g.ok and any("exclusion of care type memory_care" in x for x in g.invented)


async def test_negation_before_the_value_still_allows_the_exclusion(index):
    _, g = validate(index, "active residents who are not in memory care",
                    {"action": "list", "filters": {"status": {"include": ["active"]}, "care_type": {"exclude": ["memory_care"]}}})  # fmt: skip
    assert g.ok, g.problems()


async def test_dropped_grouping_in_a_list_is_caught(index):
    _, g = validate(index, "residents per care type in North Wing",
                    {"action": "list", "filters": {"facility": {"include": ["North Wing"]}}})  # fmt: skip
    assert not g.ok and any("grouping by care type" in m for m in g.missing)
