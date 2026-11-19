import re

import pytest
from pydantic import ValidationError

from app.models.plan import QueryPlan
from app.nlu.prompt import build_output_schema, build_system_prompt, conditions_to_filters, filters_to_conditions
from app.nlu.request_ir import RequestIR
from app.nlu.translator import SchemaContext

# Schema/prompt are built from the live vocabulary; this context stands in for one database's values.
CTX = SchemaContext(
    statuses=["active", "discharged", "pending"],
    care_types=["memory_care", "assisted_living"],
    wings=["North Wing", "Garden Terrace"],
    caregivers=["Nurse Sarah Jenkins, RN"],
)
OUTPUT_SCHEMA = build_output_schema(CTX)
SYSTEM_PROMPT = build_system_prompt(CTX)


def test_valid_combined_plan():
    plan = QueryPlan.model_validate(
        {
            "action": "count",
            "filters": {
                "status": {"include": ["active"]},
                "primary_caregiver": {"include": ["Nurse Sarah Jenkins, RN"]},
            },
        }
    )
    assert plan.filters.status.include == ["active"]


def test_breakdown_can_project_resident_fields():
    plan = QueryPlan.model_validate(
        {
            "action": "breakdown",
            "filters": {"status": {"include": ["active", "discharged"]}},
            "group_by": ["status"],
            "fields": ["room_number"],
        }
    )
    assert plan.output_fields == ["room_number"]
    with pytest.raises(ValidationError):
        QueryPlan.model_validate({"action": "count", "group_by": ["status"]})


@pytest.mark.parametrize(
    "bad",
    [
        {"action": "count", "filters": {"status": {"include": ["alive"]}}},  # unknown enum
        {"action": "count", "filters": {"diagnosis": {"include": ["x"]}}},  # field not in schema
        {"action": "count", "filters": {"admission_date": {"start": "2026-02-31"}}},  # impossible date
        {"action": "count", "filters": {"admission_date": {"start": "2026-05-01", "end": "2026-04-01"}}},  # reversed
        {"action": "count", "filters": {"age": {"min": 90, "max": 80}}},
        {"action": "count", "filters": {"status": {}}},  # empty filter
        {"action": "breakdown"},  # missing group_by
        {"action": "aggregate"},  # missing aggregate
        {"action": "lookup"},  # missing residents
        {"action": "list", "limit": 100000},
        {"action": "delete_all"},
        {"action": "count", "extra": True},
    ],
)
def test_invalid_plans_rejected(bad):
    with pytest.raises(ValidationError):
        QueryPlan.model_validate(bad)


def _walk(schema, path="$"):
    yield path, schema
    for key in ("properties",):
        for name, sub in schema.get(key, {}).items():
            yield from _walk(sub, f"{path}.{name}")
    if "items" in schema:
        yield from _walk(schema["items"], f"{path}[]")
    for i, sub in enumerate(schema.get("anyOf", [])):
        yield from _walk(sub, f"{path}|{i}")


def test_output_schema_objects_forbid_extra_properties():
    for path, node in _walk(OUTPUT_SCHEMA):
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False, path


def test_output_schema_mirrors_pydantic_model():
    assert set(OUTPUT_SCHEMA["properties"]) == set(RequestIR.model_fields)
    assert OUTPUT_SCHEMA["properties"]["where"]["anyOf"]
    assert OUTPUT_SCHEMA["$defs"]["IRNode"]["properties"]["kind"]["enum"] == [
        "predicate",
        "and",
        "or",
        "not",
        "relationship",
    ]


def test_request_ir_compiler_enforces_live_values_beyond_schema_shape():
    from app.nlu.request_ir import IRCompilationError, compile_request_ir, request_ir_from_plan

    raw = request_ir_from_plan(
        {"action": "count", "filters": {"status": {"include": ["away"]}}},
        "How many residents are away?",
    )
    with pytest.raises(IRCompilationError, match="live vocabulary"):
        compile_request_ir(raw, "How many residents are away?", CTX.statuses, CTX.care_types)
    assert "active | discharged | pending" in SYSTEM_PROMPT and "memory_care | assisted_living" in SYSTEM_PROMPT


def test_conditions_convert_to_filters_in_any_order():
    conds = [
        {"field": "primary_caregiver", "include": ["Elena Rostova"]},
        {"field": "admission_date", "start": "2026-10-01", "end": "2026-10-31"},
        {"field": "status", "include": ["active"]},
        {"field": "status", "include": ["pending"]},
        {"field": "age", "min": 80},
    ]
    f = conditions_to_filters(conds)
    assert f == {
        "primary_caregiver": {"include": ["Elena Rostova"]},
        "admission_date": {"start": "2026-10-01", "end": "2026-10-31"},
        "status": {"include": ["active", "pending"]},
        "age": {"min": 80},
    }
    assert conditions_to_filters(filters_to_conditions(f)) == f


def test_malformed_conditions_fail_validation():
    for bad in (
        [{"field": "diagnosis", "include": ["x"]}],
        [{"field": "status", "start": "2026-01-01"}],
        [{"include": ["a"]}],
    ):
        with pytest.raises(ValidationError):
            QueryPlan.model_validate({"action": "count", "filters": conditions_to_filters(bad)})


def test_system_prompt_is_static_for_prompt_caching():
    assert not re.search(r"\d{4}-\d{2}-\d{2}", SYSTEM_PROMPT), (
        "dates must go in the user message, not the cached prefix"
    )
    assert "resident in a breakdown" in SYSTEM_PROMPT
    assert "never invent resident placeholders" in SYSTEM_PROMPT


def test_retry_candidate_is_distinct_from_conversation_memory():
    from app.nlu.prompt import build_user_message

    message = build_user_message(
        "Question text",
        "2026-11-09",
        [],
        previous_ir_json='{"action":"count"}',
        retry_ir_json='{"action":"list"}',
        hint="Use a breakdown.",
    )
    assert 'Previous RequestIR (use only if the question refers back to it): {"action":"count"}' in message
    assert 'Retry candidate RequestIR (same question): {"action":"list"}' in message


def test_prompt_uses_one_compositional_interpretation_not_a_requirements_ledger():
    from app.nlu.prompt import build_messages, build_user_message

    user = build_user_message("Question text", "2026-11-09", [], None)
    assert "RequestRequirements" not in user
    examples = build_messages(CTX, "Question text")
    dialogue = "\n".join(message["content"] for message in examples)
    assert "Count September 2026 admissions per care type" in dialogue
    assert '"kind":"and"' in dialogue
    assert "resident_1" not in dialogue
    assert "Sarah Jenkins" not in dialogue
