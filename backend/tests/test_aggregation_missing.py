"""Aggregation with missing or malformed data: the defined behaviour, and "incomplete data never crashes a valid query".

Defined behaviour (repository.MISSING_GROUP, GroupAggregateRow), identical in MongoDB and in memory:
  • Grouping (breakdowns and grouped aggregates): a resident whose grouped field is not recorded — absent, null, "",
    an empty list, or a placeholder declared in resident_schema.yaml (status "unknown", facility "Unknown") — is counted
    in ONE group, "Not recorded". It is never dropped, never split into variants, never turned back into a filter.
  • A list field (several caregivers) puts the resident in the group of each value, like MongoDB's $unwind.
  • Aggregates (avg/min/max of age) use only residents whose age is a real number; others are counted in the group
    but not in the value. `n_values` says how many were used; a group with none has value None ("age not recorded").
  • An ungrouped aggregate over residents that all lack an age says the age is not recorded — not "no residents".
"""

import copy
import json

import pytest

from app.data.memory_repository import SEED_FILE, MemoryResidentRepository
from app.data.repository import MISSING_GROUP
from app.engine.executor import execute
from app.engine.responder import build_response
from app.models.plan import Filters, QueryPlan


def rec(rid, status="active", age=80, care="memory_care", wing="North Wing", cg="Nurse A"):
    return {"residentId": rid, "id": rid, "firstName": "Test", "lastName": rid, "status": status, "careType": care,
            "age": age, "wing": wing, "primaryCaregiver": cg, "roomNumber": "101", "gender": "Female",
            "admissionDate": "2026-01-01", "createdDate": "2026-01-01", "emergencyContact": {}}  # fmt: skip


async def agg(records, group_by, op="avg"):
    rows = await MemoryResidentRepository(records).group_aggregate(Filters(), group_by, op, "age")
    return {r.key: (r.count, r.value, r.n_values) for r in rows}


# ---------------------------------------------------------------- the defined behaviour
async def test_every_form_of_missing_status_is_one_not_recorded_group():
    records = [rec("A", age=80), rec("B", age=90), rec("C", status=None, age=70), rec("D", status="unknown", age=60),
               rec("E", status="", age=None)]  # fmt: skip
    del records[2]["status"]  # absent field, not just null
    out = await agg(records, ["status"])
    assert out[("active",)] == (2, 85.0, 2)
    assert out[(MISSING_GROUP,)] == (3, 65.0, 2)  # absent, "unknown" placeholder, "" → one group; E has no age
    assert len(out) == 2


async def test_only_real_numbers_are_aggregated():
    records = [rec("A", age=80), rec("B", age="85"), rec("C", age=True), rec("D", age=None), rec("E", age=90)]
    assert await agg(records, ["status"]) == {("active",): (5, 85.0, 2)}  # "85" and True are not ages
    assert await agg(records, ["status"], "min") == {("active",): (5, 80, 2)}
    assert await agg(records, ["status"], "max") == {("active",): (5, 90, 2)}


async def test_group_without_any_recorded_age_has_no_value():
    out = await agg([rec("A", age=None), rec("B", status="discharged", age=88)], ["status"])
    assert out[("active",)] == (1, None, 0) and out[("discharged",)] == (1, 88.0, 1)


async def test_list_field_counts_each_value_and_missing_once():
    records = [rec("A", cg=["Nurse A", "Aide B"], age=80), rec("B", cg=[], age=90), rec("C", cg=None, age=70)]
    out = await agg(records, ["primary_caregiver"])
    assert out[("Nurse A",)] == (1, 80.0, 1) and out[("Aide B",)] == (1, 80.0, 1)
    assert out[(MISSING_GROUP,)] == (2, 80.0, 2)


async def test_breakdown_uses_the_same_missing_rule():
    records = [rec("A"), rec("B", wing="Unknown"), rec("C", wing=None), rec("D", wing="")]
    groups = await MemoryResidentRepository(records).group_count(Filters(), ["facility"])
    assert {g.key: g.count for g in groups} == {("North Wing",): 1, (MISSING_GROUP,): 3}


# ---------------------------------------------------------------- the reported crash, end to end
async def test_average_age_by_status_with_a_missing_status_does_not_crash():
    """Was: pydantic ValidationError (HTTP 500) — the "unknown" group key was turned back into a status filter."""
    records = [rec("A", age=80), rec("B", status="unknown", age=60), rec("C", status=None, age=None)]
    plan = QueryPlan(action="aggregate", aggregate={"op": "avg", "field": "age"}, group_by=["status"])
    resp = build_response(plan, await execute(plan, MemoryResidentRepository(records)))
    labels = {b["label"] for b in resp["breakdown"]}
    assert "Active — average 80.0" in labels
    assert "Not recorded — average 60.0 (age recorded for 1 of 2)" in labels
    assert resp["count"] == 3


async def test_average_when_no_matching_resident_has_an_age_says_so():
    plan = QueryPlan(action="aggregate", aggregate={"op": "avg", "field": "age"})
    resp = build_response(plan, await execute(plan, MemoryResidentRepository([rec("A", age=None), rec("B", age=None)])))
    assert resp["headline"] == "Average age: not recorded" and resp["count"] == 2
    assert "No matching residents" not in resp["headline"]


# ---------------------------------------------------------------- incomplete data never crashes a valid query
def corrupted_fixture() -> list[dict]:
    """The test residents with every kind of gap or malformed value the real data can contain."""
    records = json.loads(SEED_FILE.read_text(encoding="utf-8"))
    damage = [
        {"status": None}, {"status": "unknown"}, {"status": ""}, {"careType": None}, {"careType": "unknown"},
        {"age": None}, {"age": "eighty"}, {"age": True}, {"wing": None}, {"wing": "Unknown"}, {"wing": ""},
        {"primaryCaregiver": None}, {"primaryCaregiver": []}, {"primaryCaregiver": ["Nurse X", "Aide Y"]},
        {"roomNumber": None}, {"admissionDate": None}, {"createdDate": None}, {"gender": None},
        {"emergencyContact": None}, {"firstName": None}, {"lastName": ""},
    ]  # fmt: skip
    for i, d in enumerate(damage):
        records[i].update(d)
    for k in ("status", "careType", "age", "wing", "primaryCaregiver", "roomNumber", "lastName"):
        records[len(damage) + 1].pop(k, None)  # one record with whole fields absent
    return records


GROUPS = ["status", "care_type", "facility", "gender", "primary_caregiver"]
SORTS = ["admission_date", "created_date", "age", "last_name"]
PLANS = (
    [{"action": "count"}, {"action": "list"}, {"action": "list", "fields": ["room_number", "facility", "age"]}]
    + [{"action": "breakdown", "group_by": [g]} for g in GROUPS]
    + [{"action": "breakdown", "group_by": ["status", "facility"]}]
    + [{"action": "aggregate", "aggregate": {"op": op, "field": "age"}} for op in ("avg", "min", "max")]
    + [{"action": "aggregate", "aggregate": {"op": op, "field": "age"}, "group_by": [g]} for op in ("avg", "max") for g in GROUPS]
    + [{"action": "list", "sort": {"field": s, "direction": d}, "limit": 1} for s in SORTS for d in ("asc", "desc")]
    + [{"action": "list", "sort": {"field": s, "direction": "desc"}, "limit": 1, "fields": ["room_number"]} for s in SORTS]
    + [{"action": "count", "filters": {"status": {"missing": True}}}, {"action": "count", "filters": {"age": {"min": 80}}}]
)  # fmt: skip


@pytest.mark.parametrize("raw", PLANS, ids=lambda p: json.dumps(p, sort_keys=True)[:90])
async def test_valid_plans_never_crash_on_incomplete_data(raw):
    repo = MemoryResidentRepository(corrupted_fixture())
    plan = QueryPlan.model_validate(copy.deepcopy(raw))
    response = build_response(plan, await execute(plan, repo))
    assert isinstance(response, dict) and "headline" in response


async def test_lookup_of_a_resident_with_gaps_does_not_crash():
    records = corrupted_fixture()
    plan = QueryPlan.model_validate({"action": "lookup", "lookup": {"residents": [records[0]["residentId"]]}})
    assert build_response(plan, await execute(plan, MemoryResidentRepository(records)))["count"] == 1
