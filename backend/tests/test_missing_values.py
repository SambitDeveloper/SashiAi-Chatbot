"""`missing: true/false` — "no value recorded" as a generic field-level query capability.

Covers: validation of include/exclude/missing combinations, what counts as missing per field type and schema mapping,
the MongoDB translation (old include/exclude queries unchanged), natural-language variations, the deterministic fast
path (no model call), the grounding check on model plans, and — against a real MongoDB when MISSING_TEST_MONGODB_URI
is set — identical results from the MongoDB aggregation and the in-memory repository.
"""

import os
from datetime import date

import pytest
import yaml
from pydantic import ValidationError

from app.data.memory_repository import MemoryResidentRepository
from app.data.repository import matches
from app.data.resident_view import to_mongo_filter
from app.data.schema_config import is_missing, missing_markers
from app.engine.grounding import check_plan, scan_question
from app.models.plan import Filters, QueryPlan
from app.nlu.direct_query import DirectQueryBuilder
from app.nlu.intent_router import DEFAULT_ROUTES_FILE, IntentScore, SemanticIntentRouter
from app.nlu.missing_values import MissingValueDetector
from app.nlu.vocabulary import EntityIndex
from tests.conftest import FakeTranslator
from tests.test_direct_query import make_service

ROUTES = yaml.safe_load(DEFAULT_ROUTES_FILE.read_text())
DETECTOR = MissingValueDetector(ROUTES)
TODAY = date(2026, 11, 9)


def F(**kw) -> Filters:
    return Filters.model_validate(kw)


# ---------------------------------------------------------------- 4. validation of combinations
@pytest.mark.parametrize(
    "bad",
    [
        {"primary_caregiver": {"exclude": ["X"], "missing": True}},  # a missing value never equals an excluded one
        {"primary_caregiver": {"include": ["X"], "missing": False}},  # include already implies a value
        {"status": {"include": ["active"], "exclude": ["active"]}},  # contradictory
        {"status": {}},  # nothing at all
        {"age": {"min": 80, "missing": False}},  # a range already implies a value
        {"admission_date": {"start": "2026-01-01", "missing": False}},
        {"admission_date": {}},
    ],
)
def test_conflicting_combinations_are_rejected(bad):
    with pytest.raises(ValidationError):
        Filters.model_validate(bad)


@pytest.mark.parametrize(
    "good",
    [
        {"primary_caregiver": {"missing": True}},
        {"primary_caregiver": {"missing": False}},
        {"primary_caregiver": {"include": ["X"], "missing": True}},  # X OR none (within-field OR)
        {"primary_caregiver": {"exclude": ["X"], "missing": False}},  # recorded AND not X
        {"age": {"missing": True}},
        {"age": {"min": 80, "missing": True}},  # over 80 OR no age recorded
        {"admission_date": {"missing": True}},
        {"admission_date": {"start": "2026-01-01", "missing": True}},
    ],
)
def test_valid_combinations_are_accepted(good):
    Filters.model_validate(good)


# ---------------------------------------------------------------- 2/5. what counts as missing (schema-driven)
@pytest.mark.parametrize(
    ("value", "field", "missing"),
    [
        (None, "roomNumber", True),
        ("", "roomNumber", True),
        ("MC-101", "roomNumber", False),
        ([], "primaryCaregiver", True),  # list field, nobody assigned
        (["", None], "primaryCaregiver", True),  # list field, no real value
        (["Nurse A"], "primaryCaregiver", False),
        ("Unknown", "wing", True),  # placeholder declared in resident_schema.yaml missing_values
        ("unknown", "status", True),
        ("unknown", "careType", True),
        ("active", "status", False),
        ("Unknown", "roomNumber", False),  # placeholders are per field, never global
        (None, "age", True),
        (0, "age", False),
    ],
)
def test_what_counts_as_missing(value, field, missing):
    assert is_missing(value, field) is missing


def test_missing_markers_come_from_the_schema_file():
    schema = yaml.safe_load(DEFAULT_ROUTES_FILE.with_name("resident_schema.yaml").read_text())
    for field, values in schema["missing_values"].items():
        assert missing_markers(field) == [None, "", *values]
    assert missing_markers("roomNumber") == [None, ""]


# ---------------------------------------------------------------- 10. old queries unchanged · new query shapes
def test_include_exclude_queries_are_unchanged():
    f = F(status={"include": ["active"]}, primary_caregiver={"exclude": ["X"]}, age={"min": 80},
          admission_date={"start": "2026-01-01"})  # fmt: skip
    assert to_mongo_filter(f) == {
        "status": {"$in": ["active"]},
        "primaryCaregiver": {"$nin": ["X"]},
        "age": {"$exists": True, "$ne": None, "$gte": 80},
        "admissionDate": {"$exists": True, "$nin": [None, ""], "$gte": "2026-01-01"},
    }


def test_missing_queries_follow_the_field_type():
    assert to_mongo_filter(F(room_number={"missing": True})) == {"$and": [{"roomNumber": {"$in": [None, ""]}}]}
    assert to_mongo_filter(F(wing={"missing": False})) == {"$and": [{"wing": {"$nin": [None, "", "Unknown"]}}]}
    # list field: "no element with a real value" (covers absent, null, [] and lists of blanks)
    assert to_mongo_filter(F(primary_caregiver={"missing": True})) == {
        "$and": [{"primaryCaregiver": {"$not": {"$elemMatch": {"$nin": [None, ""]}}}}]
    }
    within_or = to_mongo_filter(F(primary_caregiver={"include": ["X"], "missing": True}, status={"include": ["active"]}))
    assert within_or == {
        "$and": [
            {"status": {"$in": ["active"]}},
            {"$or": [{"primaryCaregiver": {"$in": ["X"]}}, {"primaryCaregiver": {"$not": {"$elemMatch": {"$nin": [None, ""]}}}}]},
        ]
    }


# ---------------------------------------------------------------- 3. in-memory semantics on crafted records
RECORDS = [
    {"residentId": "A", "status": "active", "primaryCaregiver": "Nurse X", "roomNumber": "1", "wing": "North Wing", "age": 80, "admissionDate": "2026-01-05"},
    {"residentId": "B", "status": "active", "primaryCaregiver": "", "roomNumber": "", "wing": "Unknown", "age": None, "admissionDate": None},
    {"residentId": "C", "status": "unknown", "primaryCaregiver": [], "wing": "North Wing", "age": 90},  # roomNumber/admissionDate absent
    {"residentId": "D", "status": "discharged", "primaryCaregiver": ["Nurse X"], "roomNumber": "2", "wing": "South Wing", "age": 70, "admissionDate": "2025-06-01"},
]  # fmt: skip


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"primary_caregiver": {"missing": True}}, "BC"),
        ({"primary_caregiver": {"missing": False}}, "AD"),
        ({"room_number": {"missing": True}}, "BC"),
        ({"wing": {"missing": True}}, "B"),
        ({"status": {"missing": True}}, "C"),
        ({"age": {"missing": True}}, "B"),
        ({"admission_date": {"missing": True}}, "BC"),
        ({"admission_date": {"missing": False}}, "AD"),
        # multiple missing fields (AND)
        ({"primary_caregiver": {"missing": True}, "room_number": {"missing": True}, "wing": {"missing": True}}, "B"),
        # within-field OR: Nurse X OR nobody
        ({"primary_caregiver": {"include": ["Nurse X"], "missing": True}}, "ABCD"),
        # recorded AND not X
        ({"primary_caregiver": {"exclude": ["Nurse X"], "missing": False}}, ""),
        # combined with other filters
        ({"status": {"include": ["active"]}, "room_number": {"missing": True}}, "B"),
        ({"age": {"min": 85, "missing": True}}, "BC"),  # over 85 OR no age
        ({"admission_date": {"start": "2026-01-01", "missing": True}}, "ABC"),  # 2026 admissions OR no date
    ],
)
def test_in_memory_semantics(filters, expected):
    assert "".join(r["residentId"] for r in RECORDS if matches(r, F(**filters))) == expected


# ---------------------------------------------------------------- 6. natural-language variations
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How many residents have no primary caregiver assigned?", {"primary_caregiver": True}),
        ("residents not assigned to any caregiver", {"primary_caregiver": True}),
        ("residents whose caregiver is not assigned", {"primary_caregiver": True}),
        ("residents who don't have a caregiver", {"primary_caregiver": True}),
        ("residents without a room", {"room_number": True}),
        ("list residents with room not recorded", {"room_number": True}),
        ("residents with an empty room number", {"room_number": True}),
        ("residents whose admission date is missing", {"admission_date": True}),
        ("residents with no date of birth", {"age": True}),
        ("residents with unknown status", {"status": True}),
        ("residents with a caregiver assigned", {"primary_caregiver": False}),
        ("residents with no caregiver and no room", {"primary_caregiver": True, "room_number": True}),
        ("who is the staff assigned to resident_1", {}),  # a lookup, not a condition
        ("no. of residents in memory care", {}),
        ("how many residents are active", {}),
    ],
)
def test_phrase_variations(question, expected):
    assert DETECTOR.scan(question).by_field == expected


def test_or_across_fields_is_flagged():
    assert DETECTOR.scan("residents with no caregiver or no room").or_across_fields
    assert not DETECTOR.scan("residents with no caregiver and no room").or_across_fields


# ---------------------------------------------------------------- 7/8. deterministic fast path
@pytest.fixture
async def index():
    return EntityIndex(await MemoryResidentRepository().vocabulary())


@pytest.fixture
def builder():
    return DirectQueryBuilder(SemanticIntentRouter().config)


def plan_for(builder, index, q, route):
    r = builder.build(q, index.mask(q), IntentScore(route, 0.95, 0.2, {route: 0.95}), index, TODAY)
    return r.plan if r.plan is not None else r.reason


@pytest.mark.parametrize(
    ("question", "route", "plan"),
    [
        ("How many residents have no primary caregiver assigned?", "count",
         {"action": "count", "filters": {"primary_caregiver": {"missing": True}}}),
        ("list residents without a room", "list", {"action": "list", "filters": {"room_number": {"missing": True}}}),
        ("residents with no caregiver and no room", "list",
         {"action": "list", "filters": {"primary_caregiver": {"missing": True}, "room_number": {"missing": True}}}),
        ("how many active residents in memory care have no room assigned", "count",
         {"action": "count", "filters": {"status": {"include": ["active"]}, "care_type": {"include": ["memory_care"]},
                                          "room_number": {"missing": True}}}),
        ("residents of Sarah Jenkins or without a caregiver", "list",
         {"action": "list", "filters": {"primary_caregiver": {"include": ["Nurse Sarah Jenkins, RN"], "missing": True}}}),
        ("how many residents have a caregiver assigned", "count",
         {"action": "count", "filters": {"primary_caregiver": {"missing": False}}}),
        ("residents whose admission date is missing", "list",
         {"action": "list", "filters": {"admission_date": {"missing": True}}}),
        ("residents with no caregiver or no room", "list", "missing_or_across_fields"),  # never silently an AND
    ],
)  # fmt: skip
async def test_fast_path_plans(builder, index, question, route, plan):
    assert plan_for(builder, index, question, route) == plan


async def test_missing_value_question_needs_no_model_call():
    fake = FakeTranslator({})
    service = await make_service({"count": 0.93, "list": 0.80}, fake)
    result = await service.handle("How many residents have no primary caregiver assigned?")
    assert result["tier"] == "direct" and result["modelCalls"] == 0 and fake.calls == []
    assert result["response"]["count"] == 0  # every test resident has a caregiver
    assert "Caregiver: not recorded" in result["interpretation"]
    with_one = await service.handle("how many residents have a caregiver assigned")
    assert with_one["tier"] == "direct" and with_one["response"]["count"] == 88


# ---------------------------------------------------------------- grounding of model plans
def ground(index, question, filters):
    mask = index.mask(question)
    plan = QueryPlan.model_validate({"action": "count", "filters": filters})
    anchors = ROUTES["direct"]["date_fields"]
    return check_plan(plan, question, mask, index, scan_question(mask.text, TODAY), anchors, None,
                      missing_detector=DETECTOR)  # fmt: skip


async def test_grounding_of_missing_values(index):
    q = "How many residents have no primary caregiver assigned?"
    assert ground(index, q, {"primary_caregiver": {"missing": True}}).ok
    dropped = ground(index, q, {})
    assert not dropped.ok and any("primary caregiver not recorded" in m for m in dropped.missing)
    named_none = ground(index, q, {"primary_caregiver": {"include": ["Nurse Sarah Jenkins, RN"]}})
    assert not named_none.ok  # a made-up caregiver instead of "none recorded"
    invented = ground(index, "How many residents are active?",
                      {"status": {"include": ["active"]}, "room_number": {"missing": True}})  # fmt: skip
    assert not invented.ok and any("room number not recorded" in x for x in invented.invented)
    either = ground(index, "residents with no caregiver or no room",
                    {"primary_caregiver": {"missing": True}, "room_number": {"missing": True}})  # fmt: skip
    # not something a model retry can fix: blocking, answered with a clarifying question straight away
    assert not either.ok and not either.correctable and any("OR between different fields" in m for m in either.blocking)
    # "without" is honoured by missing:true (not reported as a dropped negation)
    assert ground(index, "how many residents without a room", {"room_number": {"missing": True}}).ok


# ---------------------------------------------------------------- 3/9. MongoDB ≡ in-memory, against a real MongoDB
MONGO_URI = os.environ.get("MISSING_TEST_MONGODB_URI")


@pytest.mark.skipif(not MONGO_URI, reason="set MISSING_TEST_MONGODB_URI to a throwaway local MongoDB")
async def test_mongodb_and_memory_give_identical_results():
    import json
    from datetime import UTC, datetime

    from pymongo import MongoClient

    from app.data.mongo_repository import MongoResidentRepository
    from scripts.seed_mongo import SEED, to_care_app_shape

    assert MONGO_URI.startswith(("mongodb://localhost", "mongodb://127.0.0.1")), "local throwaway MongoDB only"
    docs, staffs, facilities = to_care_app_shape(json.loads(SEED.read_text(encoding="utf-8")))
    base = {k: v for k, v in docs[0].items() if k not in ("_id",)}
    gappy = [  # every kind of "not recorded" the real Care App data has
        {**base, "_id": "GAP-NO-STAFF-FIELD", "assignedStaff": None},
        {**base, "_id": "GAP-EMPTY-STAFF", "assignedStaff": []},
        {**base, "_id": "GAP-UNKNOWN-STAFF-ID", "assignedStaff": ["not-a-staff-id"]},
        {**base, "_id": "GAP-NO-ROOM", "unitNo": None},
        {**base, "_id": "GAP-BLANK-ROOM", "unitNo": ""},
        {**base, "_id": "GAP-NO-FACILITY", "facilityId": None},
        {**base, "_id": "GAP-NO-STATUS", "status": None},
        {**base, "_id": "GAP-NO-CARETYPE", "careType": None},
        {**base, "_id": "GAP-BAD-ADMISSION", "admissionDate": "16/07/2015"},
        {**base, "_id": "GAP-NO-ADMISSION", "admissionDate": ""},
        {**base, "_id": "GAP-NO-AGE", "age": None, "birthDate": None},
        # soft-deleted and gappy: must never be counted, whatever the filter
        {**base, "_id": "GAP-DELETED", "assignedStaff": [], "unitNo": "", "deletedAt": datetime.now(UTC)},
    ]
    for d in gappy:
        for k in [k for k, v in d.items() if v is None and k != "deletedAt"]:
            del d[k]  # an absent field, not a stored null (both cases appear above)
    db = MongoClient(MONGO_URI)["care_app_missing_test"]
    for name, rows in (("residents", docs + gappy), ("staffs", staffs), ("facilities", facilities)):
        db[name].drop()
        db[name].insert_many(rows)

    mongo = MongoResidentRepository(MONGO_URI, "care_app_missing_test", "residents")
    await mongo.init()
    mapped = await mongo.find(Filters(), limit=5000)
    assert all(r["residentId"] != "GAP-DELETED" for r in mapped)  # soft-delete filter preserved
    memory = MemoryResidentRepository(mapped)

    cases = [
        {"primary_caregiver": {"missing": True}},
        {"primary_caregiver": {"missing": False}},
        {"room_number": {"missing": True}},
        {"wing": {"missing": True}},
        {"status": {"missing": True}},
        {"care_type": {"missing": True}},
        {"age": {"missing": True}},
        {"age": {"missing": False}},
        {"admission_date": {"missing": True}},
        {"admission_date": {"missing": False}},
        {"discharge_date": {"missing": True}},
        {"primary_caregiver": {"missing": True}, "room_number": {"missing": True}},
        {"primary_caregiver": {"include": ["Nurse Sarah Jenkins, RN"], "missing": True}},
        {"primary_caregiver": {"exclude": ["Nurse Sarah Jenkins, RN"], "missing": False}},
        {"status": {"include": ["active"]}, "room_number": {"missing": True}},
        {"age": {"min": 85, "missing": True}},
        {"admission_date": {"start": "2026-01-01", "missing": True}},
        {"status": {"include": ["active"]}, "care_type": {"exclude": ["memory_care"]}},  # old behaviour, unchanged
    ]
    for case in cases:
        f = F(**case)
        in_mongo = sorted(r["residentId"] for r in await mongo.find(f, limit=5000))
        in_memory = sorted(r["residentId"] for r in await memory.find(f, limit=5000))
        assert in_mongo == in_memory, case
        assert await mongo.count(f) == await memory.count(f) == len(in_mongo), case
    # every gap kind was really exercised
    assert len(await mongo.find(F(primary_caregiver={"missing": True}), limit=5000)) == 3
    assert len(await mongo.find(F(room_number={"missing": True}), limit=5000)) == 2
    await mongo.close()


def test_placeholder_values_are_never_recognised_as_names():
    """Regression (sal_staging): the facility list contains the placeholder "Unknown"; "unknown status" was read as
    the facility "Unknown" and the question failed."""
    from app.data.repository import Vocabulary

    idx = EntityIndex(Vocabulary(wings=["North Wing", "Unknown"], caregivers=["", "Nurse A B"], rooms=["", "101"]))
    assert idx.wings == ["North Wing"] and idx.mentioned_wings("residents with unknown status") == {}
    assert idx.caregiver_values() == ["Nurse A B"]
    builder = DirectQueryBuilder(SemanticIntentRouter().config)
    assert plan_for(builder, idx, "How many residents have unknown status?", "count") == {
        "action": "count", "filters": {"status": {"missing": True}}
    }


async def test_or_across_fields_is_answered_without_model_calls():
    fake = FakeTranslator({})
    service = await make_service({"count": 0.93, "list": 0.80}, fake)
    result = await service.handle("How many residents have no caregiver or no room?")
    assert result["modelCalls"] == 0 and fake.calls == [] and result["mcpCall"] is None  # nothing executed
    assert "separately" in result["response"]["summaryText"]
