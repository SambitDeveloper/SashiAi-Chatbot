import re

import pytest
from pydantic import ValidationError

from app.models.plan import Filters, QueryPlan
from app.nlu.prompt import build_output_schema, build_system_prompt, conditions_to_filters, filters_to_conditions
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
    # options are server-generated; uses_previous is a cache hint; the model writes filters as a list of conditions
    model_side = set(OUTPUT_SCHEMA["properties"]) - {"uses_previous", "conditions"} | {"filters"}
    assert model_side == set(QueryPlan.model_fields) - {"options"}
    condition_fields = set()
    for option in OUTPUT_SCHEMA["properties"]["conditions"]["items"]["anyOf"]:
        f = option["properties"]["field"]
        condition_fields |= {f["const"]} if "const" in f else set(f["enum"])
    assert condition_fields == set(Filters.model_fields)


def test_output_schema_values_come_from_the_database_vocabulary():
    options = {
        o["properties"]["field"].get("const"): o for o in OUTPUT_SCHEMA["properties"]["conditions"]["items"]["anyOf"]
    }
    assert options["status"]["properties"]["include"]["items"]["enum"] == CTX.statuses
    assert options["care_type"]["properties"]["include"]["items"]["enum"] == CTX.care_types
    assert options["facility"]["properties"]["include"]["items"]["enum"] == CTX.wings
    # values the database does not contain are not even expressible by the model
    assert "away" not in str(options["status"]) and "respite_care" not in str(options["care_type"])
    assert "active | discharged | pending" in SYSTEM_PROMPT and "away" not in SYSTEM_PROMPT


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
