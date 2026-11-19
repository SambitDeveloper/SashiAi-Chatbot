"""Projection (`fields`, what to SHOW) — independent of conditions (what to FILTER BY).

Covers: the query model (and backward compatibility with lookup.attributes), the role of each phrase in the sentence,
the deterministic fast path, independent validation (unsupported / invented / omitted / single-record-only fields),
execution in the in-memory repository and — when PROJECTION_TEST_MONGODB_URI is set — identical projected rows from
MongoDB, the response columns and the end-to-end flow without a model call.
"""

import os
from datetime import date

import pytest
import yaml
from pydantic import ValidationError

from app.data.memory_repository import MemoryResidentRepository
from app.data.repository import project
from app.data.schema_config import identity_fields, output_fields
from app.engine.executor import ExecutionData, execute
from app.engine.grounding import check_plan, scan_question
from app.engine.responder import build_response
from app.models.plan import Filters, QueryPlan
from app.nlu.direct_query import DirectQueryBuilder
from app.nlu.intent_router import DEFAULT_ROUTES_FILE, IntentScore, SemanticIntentRouter
from app.nlu.missing_values import MissingValueDetector
from app.nlu.projection import ProjectionDetector, value_starts
from app.nlu.references import ReferenceResolver
from app.nlu.vocabulary import EntityIndex
from tests.conftest import FakeTranslator
from tests.test_direct_query import make_service

ROUTES = yaml.safe_load(DEFAULT_ROUTES_FILE.read_text())
DETECTOR = ProjectionDetector(ROUTES)
TODAY = date(2026, 11, 9)


@pytest.fixture
async def index():
    return EntityIndex(await MemoryResidentRepository().vocabulary())


@pytest.fixture
def builder():
    return DirectQueryBuilder(SemanticIntentRouter().config)


def plan_for(builder, index, q, route="list"):
    r = builder.build(q, index.mask(q), IntentScore(route, 0.95, 0.2, {route: 0.95}), index, TODAY)
    return r.plan if r.plan is not None else r.reason


# ---------------------------------------------------------------- query model + backward compatibility
def test_fields_are_validated_by_the_model():
    QueryPlan.model_validate({"action": "list", "fields": ["room_number", "primary_caregiver"]})
    with pytest.raises(ValidationError):
        QueryPlan.model_validate({"action": "count", "fields": ["room_number"]})  # a count shows no columns
    with pytest.raises(ValidationError):
        QueryPlan.model_validate({"action": "list", "fields": ["room_number", "room_number"]})
    with pytest.raises(ValidationError):
        QueryPlan.model_validate({"action": "list", "fields": ["diagnosis"]})  # not an attribute at all


def test_lookup_attributes_and_fields_are_one_projection():
    legacy = QueryPlan.model_validate({"action": "lookup", "lookup": {"residents": ["RES-1"], "attributes": ["age"]}})
    explicit = QueryPlan.model_validate(
        {"action": "lookup", "lookup": {"residents": ["RES-1"]}, "fields": ["room_number", "age"]}
    )
    assert legacy.output_fields == ["age"]
    assert explicit.output_fields == ["room_number", "age"]
    assert QueryPlan.model_validate({"action": "list"}).output_fields == []  # no projection: default shape


def test_supported_fields_come_from_the_schema():
    fields = output_fields()
    assert fields["room_number"] == "roomNumber" and fields["primary_caregiver"] == "primaryCaregiver"
    assert "residentId" in identity_fields() and "firstName" in identity_fields()


# ---------------------------------------------------------------- role of each phrase in the sentence
@pytest.mark.parametrize(
    ("question", "shown"),
    [
        ("List all Active residents under Skilled Nursing and show their room numbers.", ["room_number"]),
        ("list active residents with their caregivers and admission dates", ["primary_caregiver", "admission_date"]),
        ("show the room number, caregiver and age of residents in memory care", ["room_number", "primary_caregiver", "age"]),
        ("residents in the North Wing and their rooms", ["room_number"]),
        # conditions only: the attribute phrase is bound to a value
        ("list residents with caregiver Sarah Jenkins", []),
        ("residents with age over 80", []),
        ("list residents admitted last month", []),
        ("list active residents", []),
    ],
)
async def test_projection_role(index, question, shown):
    found = DETECTOR.scan(question, None, value_starts(question, index, TODAY))
    assert [m.attribute for m in found] == shown


# ---------------------------------------------------------------- fast path
@pytest.mark.parametrize(
    ("question", "plan"),
    [
        # the reported question
        ("List all Active residents under Skilled Nursing and show their room numbers.",
         {"action": "list", "filters": {"status": {"include": ["active"]}, "care_type": {"include": ["skilled_nursing"]}},
          "fields": ["room_number"]}),
        # the same attribute as a condition AND a projection
        ("List residents in room MC-101 and show their room numbers.",
         {"action": "list", "filters": {"room_number": {"include": ["MC-101"]}}, "fields": ["room_number"]}),
        # multiple fields, incl. a list-valued one (several caregivers)
        ("list active residents with their caregivers and admission dates",
         {"action": "list", "filters": {"status": {"include": ["active"]}},
          "fields": ["primary_caregiver", "admission_date"]}),
        # projection only
        ("show the room number, caregiver and age of residents in memory care",
         {"action": "list", "filters": {"care_type": {"include": ["memory_care"]}},
          "fields": ["room_number", "primary_caregiver", "age"]}),
        # filter only: unchanged, no projection
        ("list active residents", {"action": "list", "filters": {"status": {"include": ["active"]}}}),
        # a missing-value condition is never a column
        ("list residents with no caregiver", {"action": "list", "filters": {"primary_caregiver": {"missing": True}}}),
    ],
)  # fmt: skip
async def test_fast_path(builder, index, question, plan):
    assert plan_for(builder, index, question) == plan


# ---------------------------------------------------------------- independent validation
def ground(index, question, plan_dict, monkeypatch=None):
    mask = index.mask(question)
    plan = QueryPlan.model_validate(plan_dict)
    return check_plan(
        plan, question, mask, index, scan_question(mask.text, TODAY), ROUTES["direct"]["date_fields"], None,
        missing_detector=MissingValueDetector(ROUTES), reference_resolver=ReferenceResolver(ROUTES),
        projection_detector=DETECTOR,
    )  # fmt: skip


Q = "List all Active residents under Skilled Nursing and show their room numbers."
FILTERS = {"status": {"include": ["active"]}, "care_type": {"include": ["skilled_nursing"]}}


async def test_validation_of_projection(index):
    assert ground(index, Q, {"action": "list", "filters": FILTERS, "fields": ["room_number"]}).ok
    omitted = ground(index, Q, {"action": "list", "filters": FILTERS})  # Qwen3's plan in the report
    assert not omitted.ok and omitted.correctable and "show room number" in omitted.missing
    invented = ground(index, Q, {"action": "list", "filters": FILTERS, "fields": ["room_number", "age"]})
    assert not invented.ok and "show age" in invented.invented
    # conditions are validated on their own: a correct projection does not excuse a dropped condition
    dropped = ground(index, Q, {"action": "list", "filters": {"status": {"include": ["active"]}}, "fields": ["room_number"]})
    assert not dropped.ok and any("skilled nursing" in m for m in dropped.missing)


async def test_repeated_counts_require_grouping_and_preserve_projected_fields(index):
    question = (
        "Among residents admitted after August 1, 2026, how many are Active and how many are Discharged? "
        "Show the names and room numbers in each group."
    )
    filters = {
        "admission_date": {"start": "2026-08-02"},
        "status": {"include": ["active", "discharged"]},
    }
    valid = ground(
        index,
        question,
        {"action": "breakdown", "filters": filters, "group_by": ["status"], "fields": ["room_number"]},
    )
    assert valid.ok, valid.problems()
    assert valid.plan.filters.admission_date.start.isoformat() == "2026-08-02"

    combined_count = ground(index, question, {"action": "count", "filters": filters})
    assert not combined_count.ok
    assert any("breakdown" in problem for problem in combined_count.missing)
    assert any("grouping by status" in problem for problem in combined_count.missing)

    omitted = ground(index, question, {"action": "breakdown", "filters": filters, "group_by": ["status"]})
    assert not omitted.ok and "show room number" in omitted.missing

    invented = ground(
        index,
        question,
        {
            "action": "breakdown",
            "filters": filters,
            "group_by": ["status"],
            "fields": ["room_number", "primary_caregiver"],
        },
    )
    assert not invented.ok and "show primary caregiver" in invented.invented


async def test_single_record_only_fields_are_never_listed(index):
    q = "list active residents and show their emergency contacts"
    g = ground(index, q, {"action": "list", "filters": {"status": {"include": ["active"]}}, "fields": ["emergency_contact"]})
    assert not g.ok and not g.correctable and any("one resident at a time" in b for b in g.blocking)


async def test_unsupported_field_is_blocking(index, monkeypatch):
    import app.engine.grounding as grounding

    monkeypatch.setattr(grounding, "output_fields", lambda: {"room_number": "roomNumber"})  # a schema without age
    g = ground(index, "list residents and show their age", {"action": "list", "fields": ["age"]})
    assert not g.ok and any("not a field the resident records can show" in b for b in g.blocking)


# ---------------------------------------------------------------- execution + response
async def test_projected_rows_and_columns():
    repo = MemoryResidentRepository()
    plan = QueryPlan.model_validate({"action": "list", "filters": FILTERS, "fields": ["room_number", "primary_caregiver"]})
    data = await execute(plan, repo)
    assert data.rows and all(set(r) <= {*identity_fields(), "roomNumber", "primaryCaregiver"} for r in data.rows)
    assert all("emergencyContact" not in r and "age" not in r for r in data.rows)
    response = build_response(plan, data)
    assert [c["label"] for c in response["columns"]] == ["Room", "Primary Caregiver"]
    assert [c["key"] for c in response["columns"]] == ["roomNumber", "primaryCaregiver"]


async def test_breakdown_returns_projected_residents_within_each_group():
    records = [
        {
            "residentId": "RES-1", "id": "RES-1", "firstName": "Alex", "lastName": "Active",
            "status": "active", "admissionDate": "2026-08-02", "roomNumber": "A-1",
        },
        {
            "residentId": "RES-2", "id": "RES-2", "firstName": "Blair", "lastName": "Active",
            "status": "active", "admissionDate": "2026-08-10", "roomNumber": "A-2",
        },
        {
            "residentId": "RES-3", "id": "RES-3", "firstName": "Casey", "lastName": "Discharged",
            "status": "discharged", "admissionDate": "2026-08-03", "roomNumber": "D-1",
        },
        {
            "residentId": "RES-4", "id": "RES-4", "firstName": "Drew", "lastName": "Away",
            "status": "away", "admissionDate": "2026-08-04", "roomNumber": "X-1",
        },
    ]
    plan = QueryPlan.model_validate(
        {
            "action": "breakdown",
            "filters": {
                "admission_date": {"start": "2026-08-02"},
                "status": {"include": ["active", "discharged"]},
            },
            "group_by": ["status"],
            "fields": ["room_number"],
        }
    )
    data = await execute(plan, MemoryResidentRepository(records))
    response = build_response(plan, data)
    groups = {item["label"]: item for item in response["breakdown"]}
    assert groups["Active"]["count"] == 2
    assert groups["Discharged"]["count"] == 1
    assert [r["firstName"] for r in groups["Active"]["residents"]] == ["Alex", "Blair"]
    assert groups["Discharged"]["residents"][0]["roomNumber"] == "D-1"
    assert all(set(r) <= {*identity_fields(), "roomNumber"} for item in groups.values() for r in item["residents"])
    assert [column["label"] for column in response["columns"]] == ["Room"]


async def test_breakdown_includes_explicit_zero_count_group():
    records = [
        {
            "residentId": "RES-1", "id": "RES-1", "firstName": "Alex", "lastName": "Active",
            "status": "active", "admissionDate": "2026-08-02", "roomNumber": "A-1",
        }
    ]
    plan = QueryPlan.model_validate(
        {
            "action": "breakdown",
            "filters": {"status": {"include": ["active", "discharged"]}},
            "group_by": ["status"],
            "fields": ["room_number"],
        }
    )
    response = build_response(plan, await execute(plan, MemoryResidentRepository(records)))
    assert {item["label"]: item["count"] for item in response["breakdown"]} == {"Active": 1, "Discharged": 0}
    assert next(item for item in response["breakdown"] if item["label"] == "Discharged")["residents"] == []


async def test_default_shape_without_projection_is_unchanged():
    repo = MemoryResidentRepository()
    plan = QueryPlan.model_validate({"action": "list", "filters": FILTERS})
    data = await execute(plan, repo)
    assert "columns" not in build_response(plan, data)
    assert all({"age", "careType", "emergencyContact"} <= set(r) for r in data.rows)  # full default record shape
    assert all(r["emergencyContact"] == {"name": "", "relationship": "", "phone": ""} for r in data.rows)  # privacy rule kept


def test_projection_keeps_lists_and_nested_objects_whole():
    record = {
        "residentId": "R1", "id": "R1", "firstName": "A", "lastName": "B", "age": 80,
        "primaryCaregiver": ["Nurse X", "Aide Y"],
        "emergencyContact": {"name": "C", "relationship": "Son", "phone": "555"},
    }  # fmt: skip
    out = project(record, ["primaryCaregiver", "emergencyContact"])
    assert out["primaryCaregiver"] == ["Nurse X", "Aide Y"] and out["emergencyContact"]["phone"] == "555"
    assert "age" not in out and out["firstName"] == "A"


def test_lookup_with_explicit_fields_answers_like_legacy_attributes():
    resident = {
        "residentId": "RES-1001", "firstName": "Eleanor", "lastName": "Vance", "status": "active",
        "careType": "memory_care", "gender": "Female", "age": 82, "roomNumber": "MC-101", "wing": "Garden Terrace",
        "primaryCaregiver": "Nurse Sarah Jenkins, RN", "admissionDate": "2026-01-15", "createdDate": "2026-01-10",
        "dischargeDate": None, "emergencyContact": {"name": "Thomas Vance", "relationship": "Son", "phone": "555-0192"},
    }  # fmt: skip
    data = ExecutionData(total=88, matched=1, rows=[resident], looked_up=[resident])
    legacy = QueryPlan.model_validate({"action": "lookup", "lookup": {"residents": ["RES-1001"], "attributes": ["room_number"]}})
    explicit = QueryPlan.model_validate({"action": "lookup", "lookup": {"residents": ["RES-1001"]}, "fields": ["room_number"]})
    assert build_response(legacy, data) == build_response(explicit, data)


async def test_reported_question_end_to_end_without_a_model():
    fake = FakeTranslator({})
    service = await make_service({"count": 0.80, "list": 0.93}, fake)
    # a previous question with a caregiver condition must not leak into this one ("their" is local)
    await service.handle("How many residents does Care Aide Elena Rostova look after?", session_id="s1")
    r = await service.handle(Q, session_id="s1")
    assert r["tier"] == "direct" and r["modelCalls"] == 0 and fake.calls == []
    assert r["intent"]["parameters"]["fields"] == ["room_number"]
    assert set(r["intent"]["parameters"]["filters"]) == {"status", "care_type"}
    assert [c["label"] for c in r["response"]["columns"]] == ["Room"]
    assert all(set(row) <= {*identity_fields(), "roomNumber"} for row in r["response"]["residentsSample"])
    assert "Show: Room" in r["interpretation"]


# ---------------------------------------------------------------- MongoDB ≡ in-memory projection
MONGO_URI = os.environ.get("PROJECTION_TEST_MONGODB_URI")


@pytest.mark.skipif(not MONGO_URI, reason="set PROJECTION_TEST_MONGODB_URI to a throwaway local MongoDB")
async def test_mongodb_and_memory_projection_are_identical():
    import json

    from pymongo import MongoClient

    from app.data.mongo_repository import MongoResidentRepository
    from scripts.seed_mongo import SEED, to_care_app_shape

    assert MONGO_URI.startswith(("mongodb://localhost", "mongodb://127.0.0.1")), "local throwaway MongoDB only"
    docs, staffs, facilities = to_care_app_shape(json.loads(SEED.read_text(encoding="utf-8")))
    db = MongoClient(MONGO_URI)["care_app_projection_test"]
    for name, rows in (("residents", docs), ("staffs", staffs), ("facilities", facilities)):
        db[name].drop()
        db[name].insert_many(rows)
    mongo = MongoResidentRepository(MONGO_URI, "care_app_projection_test", "residents")
    await mongo.init()
    memory = MemoryResidentRepository(await mongo.find(Filters(), limit=5000))
    cases = [
        ["roomNumber"],
        ["primaryCaregiver"],  # list field (joined for display, only when projected)
        ["roomNumber", "primaryCaregiver", "age", "admissionDate"],
        ["emergencyContact"],  # nested object (repository level; lists are blocked by the validator)
        ["residentId"],
    ]
    f = Filters.model_validate({"status": {"include": ["active"]}})
    for fields in cases:
        a = sorted(await mongo.find(f, limit=5000, fields=fields), key=lambda r: r["residentId"])
        b = sorted(await memory.find(f, limit=5000, fields=fields), key=lambda r: r["residentId"])
        assert a == b, fields
        assert all(set(r) <= {*identity_fields(), *fields} for r in a), fields
    await mongo.close()
