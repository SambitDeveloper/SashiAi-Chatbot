"""Text matching (starts_with / contains on room_number, facility, primary_caregiver) at every layer.

plan model · one shared semantics (memory ≡ MongoDB ≡ MCP) · compiler · prompt/schema · resolver · grounding ·
describe · service end to end · semantic-cache safety. Negative tests prove unsupported fields/operators are rejected.
"""

import json
import re

import pytest
from pydantic import ValidationError

from app.data import text_match
from app.data.memory_repository import MemoryResidentRepository
from app.data.repository import FIELD_MAP, matches
from app.data.resident_view import ResidentView, to_mongo_filter
from app.data.schema_config import missing_markers
from app.engine.describe import describe_filters
from app.models.plan import TEXT_MATCH_FIELDS, Filters, QueryPlan
from app.nlu.prompt import build_output_schema, build_system_prompt
from app.nlu.request_ir import (
    CONDITION_FIELDS,
    TEXT_OPERATORS,
    IRCompilationError,
    PredicateOperator,
    compile_request_ir,
    operators_for,
    request_ir_from_plan,
)
from app.nlu.translator import SchemaContext
from tests.conftest import FakeTranslator
from tests.test_semantic_cache import ask, build

ROOMS = FIELD_MAP["room_number"]  # roomNumber (scalar)
WING = FIELD_MAP["facility"]  # wing (scalar, "Unknown" = not recorded)
STAFF = FIELD_MAP["primary_caregiver"]  # primaryCaregiver (list field)


def P(evidence, field, op, value, status="supported"):
    return {"kind": "predicate", "status": status, "evidence": evidence, "field": field, "operator": op, "value": value}


def IR(where, action="list", evidence="List residents", **kw):
    return {"action": action, "action_status": "supported", "action_evidence": evidence, "where": where,
            "unsupported": [], "unresolved": [], "ambiguous": [], **kw}  # fmt: skip


# ======================================================================== 1. plan model


def test_set_filter_accepts_text_on_approved_fields_only():
    for name in TEXT_MATCH_FIELDS:
        Filters.model_validate({name: {"starts_with": ["A"]}})
        Filters.model_validate({name: {"contains": ["x"]}})
    for name in ("status", "care_type", "gender"):  # fixed-vocabulary fields: exact values only
        with pytest.raises(ValidationError, match="apply only to"):
            Filters.model_validate({name: {"starts_with": ["a"]}})


@pytest.mark.parametrize(
    "body",
    [
        {"starts_with": [""]},  # empty
        {"starts_with": ["   "]},  # blank
        {"contains": ["x" * 61]},  # too long
        {"contains": ["a\nb"]},  # control character
        {"starts_with": []},  # empty list
        {"starts_with": ["A"], "missing": True},  # a missing value never matches a text
        {"starts_with": ["A"], "missing": False},  # implied by the text match
    ],
)
def test_set_filter_rejects_bad_text_conditions(body):
    with pytest.raises(ValidationError):
        Filters.model_validate({"room_number": body})


def test_needles_are_trimmed_and_deduplicated():
    f = Filters.model_validate({"room_number": {"starts_with": [" A ", "A", "B"]}})
    assert f.room_number.starts_with == ["A", "B"] and f.has_text_match


# ======================================================================== 2. one semantics: memory ≡ MongoDB (≡ MCP)


def _mongo_eval(doc: dict, query: dict) -> bool:
    """A tiny evaluator for exactly the MongoDB operators to_mongo_filter emits, with MongoDB's array semantics."""

    def field_ok(value, cond) -> bool:
        if not isinstance(cond, dict) or not any(k.startswith("$") for k in cond):
            return value == cond
        for op, arg in cond.items():
            if op == "$options":
                continue
            if op == "$regex":
                flags = re.IGNORECASE if "i" in cond.get("$options", "") else 0
                candidates = value if isinstance(value, list) else [value]
                if not any(isinstance(v, str) and re.search(arg, v, flags) for v in candidates):
                    return False
            elif op == "$in":
                candidates = value if isinstance(value, list) else [value]
                if not any(v in arg for v in candidates):
                    return False
            elif op == "$nin":
                candidates = value if isinstance(value, list) else [value]
                if any(v in arg for v in candidates):
                    return False
            elif op == "$elemMatch":
                if not isinstance(value, list) or not any(field_ok(v, arg) for v in value):
                    return False
            elif op == "$not":
                if field_ok(value, arg):
                    return False
            else:
                raise AssertionError(f"evaluator does not know {op}")
        return True

    for key, cond in query.items():
        if key == "$and":
            if not all(_mongo_eval(doc, q) for q in cond):
                return False
        elif key == "$or":
            if not any(_mongo_eval(doc, q) for q in cond):
                return False
        elif not field_ok(doc.get(key), cond):
            return False
    return True


VALUES = [
    "AL-201", "al-203", "A", "B-1", "XA-1", "", None, 101, ["AL-1"], "A.1", "AX1", "a(b", "Ünit-1", "ünit-2",
]  # fmt: skip
FILTERS = [
    {"room_number": {"starts_with": ["A"]}},
    {"room_number": {"starts_with": ["a"]}},  # case-insensitive
    {"room_number": {"contains": ["L-2"]}},
    {"room_number": {"starts_with": ["A", "B"]}},  # OR of needles
    {"room_number": {"starts_with": ["A"], "contains": ["1"]}},  # AND of lists
    {"room_number": {"starts_with": ["A.1"]}},  # "." is literal, not "any character"
    {"room_number": {"contains": ["a(b"]}},  # regex metacharacters are literal
    {"room_number": {"starts_with": ["ü"]}},  # unicode case folding
    {"room_number": {"starts_with": ["A"], "include": ["AL-201", "B-1"]}},
    {"room_number": {"starts_with": ["A"], "exclude": ["AL-201"]}},
    {"room_number": {"starts_with": ["1"]}},  # a number value never matches text
]


@pytest.mark.parametrize("plan_filters", FILTERS, ids=[json.dumps(f) for f in FILTERS])
def test_memory_and_mongo_agree_on_every_value(plan_filters):
    f = Filters.model_validate(plan_filters)
    query = to_mongo_filter(f)
    for value in VALUES:
        doc = {ROOMS: value}
        assert matches(doc, f) == _mongo_eval(doc, query), (value, query)


def test_expected_matches_for_the_semantics():
    f = Filters.model_validate({"room_number": {"starts_with": ["A"]}})
    hits = [v for v in VALUES if matches({ROOMS: v}, f)]
    # never None, "" or 101; a list stored in a scalar field matches on any element, like the existing $in semantics
    assert hits == ["AL-201", "al-203", "A", ["AL-1"], "A.1", "AX1", "a(b"]
    dot = Filters.model_validate({"room_number": {"starts_with": ["A.1"]}})
    assert [v for v in VALUES if matches({ROOMS: v}, dot)] == ["A.1"]  # "AX1" is NOT matched: no wildcard


def test_missing_placeholders_never_match_and_mongo_agrees():
    assert "Unknown" in missing_markers(WING)
    f = Filters.model_validate({"facility": {"contains": ["unk"]}})
    for value in ("Unknown", "Garden Unknown Terrace", None, ""):
        doc = {WING: value}
        assert matches(doc, f) == _mongo_eval(doc, to_mongo_filter(f))
    assert not matches({WING: "Unknown"}, f) and matches({WING: "Garden Unknown Terrace"}, f)


def test_list_field_matches_any_real_element_like_elem_match():
    f = Filters.model_validate({"primary_caregiver": {"contains": ["chen"]}})
    q = to_mongo_filter(f)
    for value in (["Nurse Sarah Jenkins, RN", "Care Aide Marcus Chen"], ["Nurse Sarah Jenkins, RN"], [], None,
                  "Care Aide Marcus Chen"):  # fmt: skip  (a scalar in a list field: $elemMatch never matches it)
        doc = {STAFF: value}
        assert matches(doc, f) == _mongo_eval(doc, q), value
    assert matches({STAFF: ["x", "Care Aide Marcus Chen"]}, f)
    assert "$elemMatch" in json.dumps(q)


def test_user_text_is_escaped_never_a_regex():
    assert text_match.regex_source("starts_with", "A.*") == "^A\\.\\*"
    assert text_match.regex_source("contains", "(a|b)") == "\\(a\\|b\\)"
    f = Filters.model_validate({"room_number": {"contains": [".*"]}})
    assert not matches({ROOMS: "AL-201"}, f) and matches({ROOMS: "x.*y"}, f)
    with pytest.raises(ValueError):
        text_match.regex_source("regex", "x")


def test_plans_without_text_build_the_same_mongo_query_as_before():
    f = Filters.model_validate({"status": {"include": ["active"]}, "room_number": {"include": ["AL-201"]}})
    assert to_mongo_filter(f) == {"status": {"$in": ["active"]}, ROOMS: {"$in": ["AL-201"]}}


def test_text_only_filter_has_no_empty_value_condition():
    q = to_mongo_filter(Filters.model_validate({"room_number": {"starts_with": ["A"]}}))
    assert q == {"$and": [{ROOMS: {"$regex": "^A", "$options": "i", "$nin": missing_markers(ROOMS)}}]}


def test_mcp_and_mongo_run_the_same_pipeline_clause():
    """The MCP and PyMongo repositories both execute ResidentView.pipeline → to_mongo_filter: one code path."""
    f = Filters.model_validate({"room_number": {"starts_with": ["A"]}})
    pipeline = ResidentView().pipeline(f)
    assert {"$match": to_mongo_filter(f)} in pipeline


async def test_memory_repository_executes_text_filters():
    repo = MemoryResidentRepository()
    f = Filters.model_validate({"room_number": {"starts_with": ["al"]}, "status": {"include": ["active"]}})
    rows = await repo.find(f)
    assert rows and all(r["roomNumber"].upper().startswith("AL") and r["status"] == "active" for r in rows)


# ======================================================================== 3. compiler


Q = 'List residents whose room numbers start with "A", who are currently Active, and whose care level is Skilled Nursing.'


def test_compiles_the_original_question():
    where = {"kind": "and", "status": "supported", "evidence": "List residents", "children": [
        P('room numbers start with "A"', "room_number", "starts_with", "A"),
        P("currently Active", "status", "eq", "active"),
        P("care level is Skilled Nursing", "care_type", "eq", "skilled_nursing")]}  # fmt: skip
    _, plan = compile_request_ir(IR(where), Q, ["active"], ["skilled_nursing"])
    assert plan["filters"] == {"room_number": {"starts_with": ["A"]}, "status": {"include": ["active"]},
                               "care_type": {"include": ["skilled_nursing"]}}  # fmt: skip


def test_same_field_or_of_one_text_operator_is_a_needle_list():
    q = "rooms starting with A or B"
    where = {"kind": "or", "status": "supported", "evidence": "A or B",
             "children": [P("starting with A", "room_number", "starts_with", "A"), P("B", "room_number", "starts_with", "B")]}  # fmt: skip
    _, plan = compile_request_ir(IR(where, evidence="rooms"), q)
    assert plan["filters"] == {"room_number": {"starts_with": ["A", "B"]}}


def test_integer_needle_is_its_text():
    _, plan = compile_request_ir(IR(P("room 1", "room_number", "contains", 1), evidence="room"), "room 1")
    assert plan["filters"]["room_number"] == {"contains": ["1"]}


@pytest.mark.parametrize(
    ("where", "code", "capability"),
    [
        (P("x", "status", "starts_with", "act"), "operator", "text matching on status"),  # not an approved field
        (P("x", "age", "contains", "8"), "operator", "text matching on age"),
        ({"kind": "not", "status": "supported", "evidence": "x", "child": P("x", "room_number", "starts_with", "A")},
         "capability", "excluding by a text match"),
        ({"kind": "and", "status": "supported", "evidence": "x", "children": [
            P("x", "room_number", "starts_with", "A"), P("x", "room_number", "starts_with", "B")]},
         "capability", "requiring several"),
        ({"kind": "or", "status": "supported", "evidence": "x", "children": [
            P("x", "room_number", "starts_with", "A"), P("x", "room_number", "eq", "B-1")]},
         "capability", "combining a text match"),
        ({"kind": "and", "status": "supported", "evidence": "x", "children": [
            P("x", "room_number", "starts_with", "A"), P("x", "room_number", "missing", None)]},
         "invalid", None),
    ],
)  # fmt: skip
def test_unsupported_text_uses_are_rejected_with_a_precise_reason(where, code, capability):
    with pytest.raises(IRCompilationError) as exc:
        compile_request_ir(IR(where, evidence="x"), "x")
    assert exc.value.code == code
    assert (capability is None and exc.value.capability is None) or capability in (exc.value.capability or "")
    assert exc.value.details


def test_unknown_operator_is_a_retryable_structure_error_naming_the_allowed_ones():
    with pytest.raises(IRCompilationError) as exc:
        compile_request_ir(IR(P("x", "room_number", "like", "A%"), evidence="x"), "x")
    e = exc.value
    assert e.code == "structure" and e.retryable
    assert "operator" in e.details[0] and "'like'" in e.details[0] and "starts_with" in e.details[0]


def test_plan_round_trip_keeps_text_conditions():
    plan = {"action": "list", "filters": {"room_number": {"include": ["AL-201", "AL-203"], "starts_with": ["A", "B"],
                                                         "contains": ["2"]}}}  # fmt: skip
    q = "List residents"
    _, compiled = compile_request_ir(request_ir_from_plan(plan, q), q)
    assert QueryPlan.model_validate(compiled).filters == QueryPlan.model_validate(plan).filters


# ======================================================================== 4. prompt and output schema (generated)


def test_operator_table_is_generated_from_the_compiler():
    ctx = SchemaContext(["active"], ["memory_care"], [])
    full = build_system_prompt(ctx, operator_table=True)
    for name in ("status", "room_number", "age", "facility"):
        line = next(ln for ln in full.splitlines() if re.search(rf"\b{name}\b.*:\s*eq|\b{name}\b.*:\s*gt", ln))
        assert line.rstrip().endswith(" | ".join(operators_for(name))), line
    compact = build_system_prompt(ctx)
    text_fields = [f for f in CONDITION_FIELDS if f in TEXT_MATCH_FIELDS]
    assert sorted(text_fields) == sorted(TEXT_MATCH_FIELDS)
    assert "starts_with/contains: only " + ", ".join(text_fields) in compact


def test_output_schema_operator_enum_is_the_compiler_enum():
    schema = build_output_schema(SchemaContext(["active"], ["memory_care"], []))
    enum = schema["$defs"]["IRNode"]["properties"]["operator"]["anyOf"][0]["enum"]
    assert set(enum) == set(PredicateOperator.__args__) and set(TEXT_OPERATORS) <= set(enum)


# ======================================================================== 5. resolver, describe


def test_describe_shows_the_literal_text():
    f = Filters.model_validate({"room_number": {"starts_with": ["A", "B"]}, "facility": {"contains": ["Garden"]}})
    chips = describe_filters(f)
    assert 'Room: starts with "A" or "B"' in chips and any('contains "Garden"' in c for c in chips)


async def test_resolver_never_looks_a_needle_up_in_the_vocabulary(make_service):
    """A needle is not a room/facility name: no 'couldn't find "A"', no fuzzy replacement."""
    q = 'Which residents have a room number starting with "zz"?'
    service = await make_service(FakeTranslator({q: {"action": "list", "filters": {"room_number": {"starts_with": ["zz"]}}}}))
    r = await service.handle(q)
    assert r["tier"] == "llm" and not r["response"].get("needsClarification")
    assert r["response"]["count"] == 0 and 'Room: starts with "zz"' in r["interpretation"]


# ======================================================================== 6. grounding and service, end to end


async def test_original_question_is_answered(make_service):
    q = 'List residents whose room numbers start with "a", who are currently Active, and whose care level is Assisted Living.'
    plan = {"action": "list", "filters": {"room_number": {"starts_with": ["a"]}, "status": {"include": ["active"]},
                                          "care_type": {"include": ["assisted_living"]}}}  # fmt: skip
    service = await make_service(FakeTranslator({q: plan}))
    r = await service.handle(q)
    data = json.load(open("tests/data/residents.json"))
    expected = {d["residentId"] for d in data if (d.get("roomNumber") or "").lower().startswith("a")
                and d["status"] == "active" and d["careType"] == "assisted_living"}  # fmt: skip
    assert r["tier"] == "llm" and r["modelCalls"] == 1 and r["response"]["count"] == len(expected) > 0


async def test_a_dropped_text_condition_is_caught_and_retried(make_service):
    q = 'How many residents have a room number starting with "SN"?'
    fake = FakeTranslator({q: [{"action": "count"}, {"action": "count", "filters": {"room_number": {"starts_with": ["SN"]}}}]})
    r = await (await make_service(fake)).handle(q)
    assert len(fake.calls) == 2 and "starts with" in fake.calls[1]["hint"]
    assert r["response"]["count"] == 13  # 12 active + 1 admitted SN rooms in tests/data/residents.json


async def test_an_invented_needle_is_caught(make_service):
    q = 'How many residents have a room number starting with "SN"?'
    wrong = {"action": "count", "filters": {"room_number": {"starts_with": ["MC"]}}}
    fake = FakeTranslator({q: [wrong, wrong]})
    r = await (await make_service(fake)).handle(q)
    assert len(fake.calls) == 2 and r["response"].get("needsClarification")  # never executes "MC"


async def test_facility_mention_is_covered_by_a_text_match(make_service):
    q = 'How many residents live in a facility containing "Garden"?'
    fake = FakeTranslator({q: {"action": "count", "filters": {"facility": {"contains": ["Garden"]}}}})
    r = await (await make_service(fake)).handle(q)
    assert len(fake.calls) == 1 and r["response"]["count"] == 27


# ======================================================================== 7. semantic template cache safety


async def test_text_match_plans_are_never_stored_as_templates(repo, settings):
    q = 'How many residents have a room number starting with "SN"?'
    service, fake, store = await build(repo, settings, {q: {"action": "count", "filters": {"room_number": {"starts_with": ["SN"]}}}})
    r = await ask(service, q)
    assert r["response"]["count"] == 13 and store.docs == {}  # "SN" would otherwise be kept literally


async def test_a_template_is_never_served_when_the_question_has_a_text_match(repo, settings):
    plain = "How many active residents are there?"
    text_q = 'How many active residents have a room number starting with "SN"?'
    service, fake, store = await build(repo, settings, {
        plain: {"action": "count", "filters": {"status": {"include": ["active"]}}},
        text_q: {"action": "count", "filters": {"status": {"include": ["active"]}, "room_number": {"starts_with": ["SN"]}}},
    })  # fmt: skip
    await ask(service, plain)
    assert len(store.docs) == 1
    r = await ask(service, text_q)  # the stored template lacks the text condition: grounding rejects it
    assert r["tier"] == "llm" and r["response"]["count"] == 12
