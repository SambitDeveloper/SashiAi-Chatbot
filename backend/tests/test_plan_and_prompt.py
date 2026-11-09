import re

import pytest
from pydantic import ValidationError

from app.models.plan import Filters, QueryPlan
from app.nlu.prompt import OUTPUT_SCHEMA, SYSTEM_PROMPT


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


def test_output_schema_objects_forbid_extra_properties():
    for path, node in _walk(OUTPUT_SCHEMA):
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False, path


def test_output_schema_mirrors_pydantic_model():
    # options are server-generated; uses_previous is a cache hint consumed by the service, not part of the plan
    assert set(OUTPUT_SCHEMA["properties"]) - {"uses_previous"} == set(QueryPlan.model_fields) - {"options"}
    assert set(OUTPUT_SCHEMA["properties"]["filters"]["properties"]) == set(Filters.model_fields)


def test_system_prompt_is_static_for_prompt_caching():
    assert not re.search(r"\d{4}-\d{2}-\d{2}", SYSTEM_PROMPT), (
        "dates must go in the user message, not the cached prefix"
    )
