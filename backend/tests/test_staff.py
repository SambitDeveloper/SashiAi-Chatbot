"""Staff (staffs collection) at every layer: role mapping, resident-side derivation (care team, roles, primary physician),
MongoDB stages, the staff entity (count / list / breakdown / lookup), compiler, resolver, grounding, answers, privacy and
the template cache. Fixture: tests/data/staffs.json (fictional), linked to tests/data/residents.json.

Fixture facts used below (computed independently from the JSON files in `expected_*` helpers where it matters):
  active physicians: Priya Raman (Physician)            inactive: Owen Hart (Doctor)
  Priya Raman is assigned to RES-1001, RES-1002, RES-1004 and is RES-1002's primary physician
  Jane Okafor (Case Manager) is assigned to RES-1001 and RES-1005
  "Lobby Tablet" (designation TABLET) is a system account: never staff, never in a care team
  Ann Boyd (Manager) has no role: matched only by her designation
"""

import json

import pytest
from pydantic import ValidationError

from app.data.memory_repository import MemoryResidentRepository
from app.data.repository import list_view
from app.data.resident_view import ResidentView, to_mongo_filter
from app.data.staff import plan_uses_staff, staff_config, staff_matches, staff_mongo_filter
from app.engine.describe import describe_filters, describe_plan
from app.models.plan import Filters, QueryPlan
from app.nlu.prompt import build_system_prompt
from app.nlu.request_ir import compile_request_ir
from app.nlu.translator import SchemaContext
from tests.conftest import FakeTranslator
from tests.test_semantic_cache import ask, build

STAFF = json.load(open("tests/data/staffs.json", encoding="utf-8"))
RESIDENTS = json.load(open("tests/data/residents.json", encoding="utf-8"))
CFG = staff_config()


def visible():
    return [s for s in STAFF if s["designation"] not in ("TABLET",)]


# ======================================================================== 1. configuration: words → roles


@pytest.mark.parametrize(
    ("word", "roles"),
    [
        ("doctors", ["physician"]),
        ("Physician", ["physician"]),
        ("case managers", ["case_manager"]),
        ("nurse practitioner", ["nurse_practitioner"]),
        ("registered dietician", ["dietitian"]),
        ("case_manager", ["case_manager"]),
    ],
)
def test_role_words(word, roles):
    assert CFG.expand(word) == roles


def test_groups_and_exclusions():
    therapists = CFG.expand("therapists")
    assert "physical_therapist" in therapists and "director_of_rehab" in therapists
    assert "massage_therapist" not in therapists  # a massage therapist is not rehab therapy
    assert set(CFG.expand("nursing")) == {"nurse", "director_of_nursing"}
    assert CFG.expand("caregiver") == ["caregiver"]  # the canonical role name still resolves…
    assert CFG.roles_in("who is the caregiver") == []  # …but the word in a question means ANY assigned staff
    assert CFG.role_for("  case manager ") == "case_manager" and CFG.role_for("Manager") is None
    assert CFG.is_hidden("TABLET") and CFG.is_hidden("time pass") and not CFG.is_hidden("Manager")


def test_specific_roles_skip_generic_words_and_titles():
    assert [w for w, _ in CFG.specific_roles_in("residents without a case manager")] == ["case manager"]
    assert CFG.specific_roles_in("how many nurses") == []  # generic: may simply be the caregiver
    assert CFG.specific_roles_in("residents of Dr. Raman") == []  # a title before a name
    assert CFG.specific_roles_in("residents of dr raman", {"raman"}) == []
    assert CFG.specific_roles_in("how many nurses", include_generic=True)[0][1] == ["nurse"]


# ======================================================================== 2. resident-side derivation (in memory)


async def test_care_team_roles_and_primary_physician():
    repo = MemoryResidentRepository()
    by_id = {r["residentId"]: r for r in await repo.find_by_ids(["RES-1001", "RES-1002", "RES-1003"])}
    eleanor, arthur, martha = by_id["RES-1001"], by_id["RES-1002"], by_id["RES-1003"]
    assert eleanor["careTeam"] == ["Jane Okafor (Case Manager)", "Nurse Sarah Jenkins, RN (Nurse)", "Priya Raman (Physician)"]
    assert "Lobby Tablet" not in json.dumps(eleanor)  # a system account is never part of a care team
    assert eleanor["staffRoles"] == ["case_manager", "nurse", "physician"] and eleanor["staffCount"] == 3
    assert eleanor["primaryPhysician"] == ""  # assigned physician ≠ recorded primary physician
    assert arthur["primaryPhysician"] == "Priya Raman (Physician)"
    # inactive staff are shown (marked) but never match a role or count as assigned staff
    assert "Owen Hart (Doctor, inactive)" in martha["careTeam"] and "physician" not in martha["staffRoles"]
    assert "Care Aide Marcus Chen (Caregiver PM, inactive)" in martha["careTeam"] and martha["staffCount"] == 0


def test_bulk_rows_never_carry_care_team_details():
    row = list_view({"residentId": "R", "careTeam": ["A (Nurse)"], "careTeamMembers": [{}], "staffRoles": ["nurse"], "staffCount": 1})
    assert "careTeamMembers" not in row and "careTeam" not in row and row["staffCount"] == 1


def expected_without(role_designations: set[str]) -> int:
    """Residents with NO active, visible assigned staff member of these designations, from the JSON files directly."""
    n = 0
    for r in RESIDENTS:
        team = [s for s in STAFF if s["name"] == r["primaryCaregiver"] or r["residentId"] in s.get("fixtureResidents", [])]
        if not any(s["active"] and s["designation"] in role_designations for s in team):
            n += 1
    return n


async def test_resident_staff_filters_match_an_independent_count():
    repo = MemoryResidentRepository()
    assert await repo.count(Filters(staff_role={"exclude": ["case_manager"]})) == expected_without({"Case Manager"})
    assert await repo.count(Filters(staff_role={"include": ["physician"]})) == len(RESIDENTS) - expected_without({"Physician", "Doctor"})
    assert await repo.count(Filters(staff_count={"min": 3})) == 1  # only Eleanor Vance has 3 active assigned staff


# ======================================================================== 3. MongoDB stages


def test_staff_join_is_added_only_when_used():
    view = ResidentView()
    plain = view.pipeline(Filters(status={"include": ["active"]}))
    staffed = view.pipeline(Filters(staff_role={"exclude": ["case_manager"]}))
    froms = lambda p: [s["$lookup"]["from"] for s in p if "$lookup" in s]  # noqa: E731
    assert "staffs" in froms(plain) and froms(plain).count("staffs") == 1  # the existing names-only caregiver join
    assert froms(staffed).count("staffs") == 3  # + assigned staff with designations + primary physician
    assert to_mongo_filter(Filters(staff_role={"exclude": ["case_manager"]})) == {"staffRoles": {"$nin": ["case_manager"]}}
    assert plan_uses_staff(None, ["careTeam"]) and not plan_uses_staff(Filters(), ["roomNumber"])


def test_staff_view_hides_system_accounts_and_contacts():
    stages = CFG.entity_stages()
    assert stages[0] == CFG.visible_match and "tablet" in json.dumps(stages[0])
    project = stages[-1]["$project"]
    assert "email" not in project and "phone" not in project and "accessPermissions" not in json.dumps(stages)
    assert {"email", "phone"} <= set(CFG.entity_stages(contacts=True)[-1]["$project"])


@pytest.mark.parametrize(
    "filters",
    [
        {"staff_role": {"include": ["physician"]}},
        {"staff_role": {"exclude": ["physician"]}, "staff_status": {"include": ["active"]}},
        {"staff_designation": {"contains": ["coord"]}},
        {"staff_type": {"missing": True}},
        {"staff_role": {"missing": True}},
        {"staff_resident_count": {"min": 2}},
    ],
)
def test_staff_filter_engines_agree_on_the_fixture(filters):
    """The MongoDB query and the in-memory rule select the same staff (tiny evaluator for the operators used)."""
    f = Filters.model_validate(filters)
    rows = [{**CFG.entity_record(s, {}), "residentCount": len(s.get("fixtureResidents", []))} for s in visible()]
    mongo = staff_mongo_filter(f)

    def ok(row, q):
        if "$and" in q:
            return all(ok(row, c) for c in q["$and"])
        if "$or" in q:
            return any(ok(row, c) for c in q["$or"])
        (field, cond), = q.items()
        v = row.get(field)
        checks = {"$in": lambda a: v in a, "$nin": lambda a: v not in a, "$gte": lambda a: v >= a, "$lte": lambda a: v <= a,
                  "$regex": lambda a: __import__("re").search(a, v or "", __import__("re").I) is not None, "$options": lambda a: True}
        return all(checks[op](arg) for op, arg in cond.items())

    assert [r["name"] for r in rows if ok(r, mongo or {"name": {"$nin": []}})] == [r["name"] for r in rows if staff_matches(r, f)]


# ======================================================================== 4. plan validation and compiler


@pytest.mark.parametrize(
    "plan",
    [
        {"action": "count", "filters": {"staff_status": {"include": ["active"]}}},  # staff-only field, resident plan
        {"action": "list", "fields": ["staff_email"], "entity": "staff"},  # contact details for many staff
        {"action": "count", "entity": "staff", "filters": {"staff_count": {"min": 1}}},  # resident-side count
        {"action": "aggregate", "entity": "staff", "aggregate": {"op": "avg"}},
        {"action": "lookup", "lookup": {"residents": ["RES-1"], "staff": ["A"]}},
        {"action": "breakdown", "group_by": ["staff_role"]},  # staff grouping needs entity staff
    ],
)
def test_invalid_staff_plans_are_rejected(plan):
    with pytest.raises(ValidationError):
        QueryPlan.model_validate(plan)


def test_resident_plans_serialise_exactly_as_before():
    dumped = QueryPlan.model_validate({"action": "lookup", "lookup": {"residents": ["RES-1"]}}).model_dump(exclude_none=True)
    assert "entity" not in dumped and "staff" not in dumped["lookup"]
    assert QueryPlan.model_validate({"action": "count", "entity": "resident"}).entity is None


def P(evidence, field, op, value=None):
    return {"kind": "predicate", "status": "supported", "evidence": evidence, "field": field, "operator": op, "value": value}


def test_compiles_a_staff_question_with_resident_conditions():
    q = "how many physicians care for memory care residents"
    ir = {"action": "count", "entity": "staff", "action_status": "supported", "action_evidence": "how many",
          "where": {"kind": "and", "status": "supported", "evidence": "physicians care for memory care",
                    "children": [P("physicians", "staff_role", "eq", "physicians"), P("memory care", "care_type", "eq", "memory_care")]},
          "unsupported": [], "unresolved": [], "ambiguous": []}  # fmt: skip
    _, plan = compile_request_ir(ir, q)
    assert plan["entity"] == "staff" and plan["filters"]["care_type"] == {"include": ["memory_care"]}


def test_compiles_a_staff_lookup():
    q = "what is Priya Raman's phone number"
    ir = {"action": "lookup", "entity": "staff", "action_status": "supported", "action_evidence": "what is", "where": None,
          "lookup": {"staff": ["Priya Raman"], "attributes": ["staff_phone"], "status": "supported", "evidence": "Priya Raman's"},
          "unsupported": [], "unresolved": [], "ambiguous": []}  # fmt: skip
    _, plan = compile_request_ir(ir, q)
    assert plan["lookup"] == {"residents": [], "staff": ["Priya Raman"], "attributes": ["staff_phone"]}


def test_staff_section_only_in_the_detailed_prompt():
    ctx = SchemaContext(["active"], ["memory_care"], [])
    assert '"entity": "staff"' in build_system_prompt(ctx, operator_table=True)
    assert "Staff members" not in build_system_prompt(ctx)


# ======================================================================== 5. end to end (service, fake model)


async def test_count_physicians_is_active_staff_by_default(make_service):
    q = "How many physicians are there?"
    plan = {"action": "count", "entity": "staff", "filters": {"staff_role": {"include": ["physicians"]}}}
    r = await (await make_service(FakeTranslator({q: plan}))).handle(q)
    assert r["response"]["count"] == 1 and "Only active staff" in r["response"]["summaryText"]
    assert "Staff members" in r["interpretation"] and "Staff role: Physician" in r["interpretation"]


async def test_inactive_staff_when_asked(make_service):
    q = "How many physicians are there, including inactive ones?"
    plan = {"action": "count", "entity": "staff",
            "filters": {"staff_role": {"include": ["physician"]}, "staff_status": {"include": ["active", "inactive"]}}}  # fmt: skip
    r = await (await make_service(FakeTranslator({q: plan}))).handle(q)
    assert r["response"]["count"] == 2


async def test_list_case_managers(make_service):
    q = "List the case managers"
    plan = {"action": "list", "entity": "staff", "filters": {"staff_role": {"include": ["case manager"]}},
            "fields": ["staff_designation", "staff_resident_count"]}  # fmt: skip
    r = await (await make_service(FakeTranslator({q: plan}))).handle(q)
    row = r["response"]["residentsSample"][0]
    assert r["response"]["count"] == 1 and row["firstName"] == "Jane Okafor" and row["residentCount"] == 2
    assert "staffId" not in row and "c-okafor" not in json.dumps(r["response"])  # the record key never leaves


async def test_staff_by_designation_breakdown(make_service):
    q = "Show staff grouped by designation"
    plan = {"action": "breakdown", "entity": "staff", "group_by": ["staff_designation"]}
    r = await (await make_service(FakeTranslator({q: plan}))).handle(q)
    labels = {b["label"]: b["count"] for b in r["response"]["breakdown"]}
    assert labels["Physician"] == 1 and "TABLET" not in labels and "Doctor" not in labels  # active only by default
    assert sum(labels.values()) == sum(1 for s in visible() if s["active"])


async def test_residents_per_physician(make_service):
    q = "How many residents does each physician have?"
    plan = {"action": "list", "entity": "staff", "filters": {"staff_role": {"include": ["physician"]}},
            "fields": ["staff_resident_count"], "sort": {"field": "staff_resident_count", "direction": "desc"}}  # fmt: skip
    r = await (await make_service(FakeTranslator({q: plan}))).handle(q)
    assert [(x["firstName"], x["residentCount"]) for x in r["response"]["residentsSample"]] == [("Priya Raman", 3)]


async def test_staff_caring_for_memory_care_residents(make_service):
    q = "Which staff members care for memory care residents?"
    plan = {"action": "list", "entity": "staff", "filters": {"care_type": {"include": ["memory_care"]}}, "fields": ["staff_resident_count"]}
    r = await (await make_service(FakeTranslator({q: plan}))).handle(q)
    memory = {x["residentId"] for x in RESIDENTS if x["careType"] == "memory_care"}
    expected = sorted(s["name"] for s in visible() if s["active"] and any(
        x["residentId"] in memory and (x["primaryCaregiver"] == s["name"] or x["residentId"] in s.get("fixtureResidents", []))
        for x in RESIDENTS))  # fmt: skip
    assert sorted(x["firstName"] for x in r["response"]["residentsSample"]) == expected


async def test_one_residents_physician_from_the_care_team(make_service):
    q = "Who is Eleanor Vance's physician?"
    plan = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["care_team"]},
            "filters": {"staff_role": {"include": ["physician"]}}}  # fmt: skip
    r = await (await make_service(FakeTranslator({"Who is resident_1's physician?": plan}))).handle(q)
    assert r["response"]["attributeSpotlight"]["value"] == "Physician: Priya Raman"


async def test_one_residents_care_team_grouped_by_designation(make_service):
    q = "Who is on Martha Stewart-Wells's care team?"
    plan = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["care_team"]}}
    r = await (await make_service(FakeTranslator({"Who is on resident_1's care team?": plan}))).handle(q)
    spot = r["response"]["attributeSpotlight"]
    assert "Doctor: Owen Hart (inactive)" in spot["value"] and "(2 inactive)" in spot["supportingDetail"]


async def test_primary_physician_is_its_own_fact(make_service):
    q = "Who is Arthur Pendleton's primary physician?"
    plan = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["primary_physician"]}}
    r = await (await make_service(FakeTranslator({"Who is resident_1's primary physician?": plan}))).handle(q)
    assert r["response"]["attributeSpotlight"]["value"] == "Priya Raman (Physician)"


async def test_residents_without_a_case_manager(make_service):
    q = "How many residents do not have a case manager?"
    plan = {"action": "count", "filters": {"staff_role": {"exclude": ["case manager"]}}}
    r = await (await make_service(FakeTranslator({q: plan}))).handle(q)
    assert r["response"]["count"] == expected_without({"Case Manager"})


async def test_staff_phone_for_one_person_only(make_service):
    q = "What is Priya Raman's phone number?"
    plan = {"action": "lookup", "entity": "staff", "lookup": {"staff": ["Priya Raman"], "attributes": ["staff_phone"]}}
    r = await (await make_service(FakeTranslator({q: plan}))).handle(q)
    assert r["response"]["attributeSpotlight"]["value"] == "555-0108"
    q2 = "What are Maya Patel's email addresses?"  # two staff records named Maya Patel: never several people's contacts
    plan2 = {"action": "lookup", "entity": "staff", "lookup": {"staff": ["Care Coordinator Maya Patel", "Wellness Coordinator Maya Patel"],
             "attributes": ["staff_email"]}}  # fmt: skip
    r2 = await (await make_service(FakeTranslator({q2: plan2}))).handle(q2)
    assert "@" not in json.dumps(r2["response"])


async def test_a_role_without_mapping_is_matched_by_its_designation(make_service):
    q = "How many managers are there?"
    plan = {"action": "count", "entity": "staff", "filters": {"staff_role": {"include": ["manager"]}}}
    r = await (await make_service(FakeTranslator({q: plan}))).handle(q)
    assert r["response"]["count"] == 1 and "Staff designation: Manager" in r["interpretation"]


async def test_unknown_role_asks(make_service):
    q = "How many astronauts are on staff?"
    plan = {"action": "count", "entity": "staff", "filters": {"staff_role": {"include": ["astronaut"]}}}
    r = await (await make_service(FakeTranslator({q: plan}))).handle(q)
    assert r["response"].get("needsClarification") and "astronaut" in r["response"]["summaryText"]


# ======================================================================== 6. grounding


async def test_a_staff_question_answered_as_residents_is_caught(make_service):
    q = "How many physicians are there?"
    residents = {"action": "count"}
    fake = FakeTranslator({q: [residents, residents]})
    r = await (await make_service(fake)).handle(q)
    assert len(fake.calls) == 2 and r["response"].get("needsClarification")  # never "88 residents"


async def test_generic_staff_rows_are_not_answered_as_residents(make_service):
    q = "how many nurses are there"
    residents = {"action": "count"}
    fake = FakeTranslator({q: [residents, residents]})
    r = await (await make_service(fake)).handle(q)
    assert r["response"].get("needsClarification")


async def test_invented_inactive_staff_and_resident_status_are_caught(make_service):
    q = "How many active physicians are there?"
    wrong = {"action": "count", "entity": "staff",
             "filters": {"staff_role": {"include": ["physician"]}, "status": {"include": ["active"]}}}  # fmt: skip
    right = {"action": "count", "entity": "staff",
             "filters": {"staff_role": {"include": ["physician"]}, "staff_status": {"include": ["active"]}}}  # fmt: skip
    fake = FakeTranslator({q: [wrong, right]})
    r = await (await make_service(fake)).handle(q)
    assert len(fake.calls) == 2 and "staff_status" in (fake.calls[1]["hint"] or "") and r["response"]["count"] == 1


async def test_caregiver_questions_are_unchanged(make_service):
    q = "Which residents does Nurse Sarah Jenkins look after?"
    plan = {"action": "list", "filters": {"primary_caregiver": {"include": ["person_1"]}}}  # the model reads person_N
    fake = FakeTranslator({q.replace("Sarah Jenkins", "person_1"): plan})
    r = await (await make_service(fake)).handle(q)
    assert not r["response"].get("needsClarification") and r["response"]["count"] == sum(
        1 for x in RESIDENTS if x["primaryCaregiver"] == "Nurse Sarah Jenkins, RN")
    assert len(fake.calls) <= 1  # no retry


def test_describe_chips():
    chips = describe_filters(Filters.model_validate({"staff_role": {"exclude": ["case_manager"]}, "staff_count": {"max": 0}}))
    assert "Staff role: not Case Manager" in chips and "Active assigned staff: none" in chips
    plan = QueryPlan.model_validate({"action": "lookup", "entity": "staff", "lookup": {"staff": ["Priya Raman"]}})
    assert describe_plan(plan, str)[:2] == ["Staff members", "Staff: Priya Raman"]


# ======================================================================== 7. template cache


async def test_staff_plans_are_never_stored_as_templates(repo, settings):
    q = "How many physicians are there?"
    service, fake, store = await build(repo, settings, {q: {"action": "count", "entity": "staff",
                                                            "filters": {"staff_role": {"include": ["physician"]}}}})  # fmt: skip
    r = await ask(service, q)
    assert r["response"]["count"] == 1 and store.docs == {}


async def test_staff_records_named_after_a_role_are_not_read_as_names(settings):
    """Staging has staff records named like their role ("Test Case Manager", "Dining Staff"): the role words in a
    question are its wording, never a demand for that person."""
    from app.engine.service import QueryService

    extra = [*STAFF, {"_id": "st-x1", "cName": "c-x1", "name": "Test Case Manager", "designation": "Case Manager",
                      "active": True, "facilityId": "East Wing"},
             {"_id": "st-x2", "cName": "c-x2", "name": "Physician Staff", "designation": "Physician", "active": True,
              "facilityId": "East Wing", "fixtureResidents": ["RES-1010"]}]  # fmt: skip
    plans = {
        "How many residents do not have a case manager?": {"action": "count", "filters": {"staff_role": {"exclude": ["case manager"]}}},
        "Show active staff grouped by designation": {"action": "breakdown", "entity": "staff", "group_by": ["staff_designation"],
                                                     "filters": {"staff_status": {"include": ["active"]}}},
    }  # fmt: skip
    service = QueryService(_repo_with(extra), FakeTranslator(plans), settings)
    await service.refresh_vocabulary()
    for q in plans:
        r = await service.handle(q)
        assert not r["response"].get("needsClarification"), r["response"]["summaryText"]


def _repo_with(staff):
    repo = MemoryResidentRepository()
    repo._staff_raw = list(staff)
    records = json.load(open("tests/data/residents.json", encoding="utf-8"))
    records[9]["primaryCaregiver"], records[10]["primaryCaregiver"] = "Physician Staff", "Test Case Manager"  # as in staging
    repo._records = records
    repo._attach_staff()
    return repo
