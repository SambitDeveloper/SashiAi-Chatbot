"""`facility` is the canonical plan field (the data has no wing: it is the resident's facility, facilityId → facilities).
The earlier name `wing` is accepted wherever a plan is read and normalised, so there is one way to express it."""

import pytest
from pydantic import ValidationError

from app.models.plan import QueryPlan
from app.nlu.prompt import conditions_to_filters


def test_legacy_wing_is_read_as_facility_everywhere():
    plan = QueryPlan.model_validate({
        "action": "list",
        "filters": {"wing": {"include": ["North Wing"]}},
        "fields": ["wing", "room_number"],
    })  # fmt: skip
    assert plan.filters.facility.include == ["North Wing"] and plan.fields == ["facility", "room_number"]
    grouped = QueryPlan.model_validate({"action": "breakdown", "group_by": ["wing"]})
    assert grouped.group_by == ["facility"]
    looked_up = QueryPlan.model_validate({"action": "lookup", "lookup": {"residents": ["RES-1"], "attributes": ["wing"]}})
    assert looked_up.lookup.attributes == ["facility"]
    assert conditions_to_filters([{"field": "wing", "include": ["North Wing"]}]) == {"facility": {"include": ["North Wing"]}}


def test_both_names_at_once_is_rejected():
    with pytest.raises(ValidationError):
        QueryPlan.model_validate({"action": "count", "filters": {"wing": {"include": ["A"]}, "facility": {"include": ["B"]}}})


def test_canonical_output_uses_facility():
    plan = QueryPlan.model_validate({"action": "count", "filters": {"wing": {"include": ["North Wing"]}}})
    assert "facility" in plan.model_dump(exclude_none=True)["filters"] and "wing" not in plan.model_dump()["filters"]
