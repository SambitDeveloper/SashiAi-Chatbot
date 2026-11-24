"""Family members (familymembers collection) at every layer: derivation rules, MongoDB stages, filters, compiler,
resolver, grounding, answers, privacy and the template cache. Fixture: tests/data/familymembers.json (fictional).

Fixture facts used below (after the rules: "Self" excluded, aliases applied, orphans attached to no one):
  RES-1001 Eleanor Vance   daughter, son, spouse(emergency, guarantor)   portal_access (Margaret)   3 members
  RES-1002 Arthur Pendleton daughter-in-law, granddaughter               app_access                  2 members
  RES-1003 Martha S-W      brother, sister, (no relation)                health_access               3 members
  RES-1004 Harold Finch    wife (emergency)                              portal_access               1 member
  everyone else (84 of 88) 0 members; fm11 belongs to RES-9999, who does not exist
"""

import json

import pytest
from pydantic import ValidationError

from app.data.memory_repository import MemoryResidentRepository
from app.data.relations import FAMILY_VIEW_FIELDS, family_relation, plan_uses_family
from app.data.repository import list_view, matches
from app.data.resident_view import ResidentView, to_mongo_filter
from app.engine.describe import describe_filters
from app.models.plan import Filters, QueryPlan
from app.nlu.prompt import build_system_prompt
from app.nlu.request_ir import IRCompilationError, compile_request_ir
from app.nlu.translator import SchemaContext
from tests.conftest import FakeTranslator
from tests.test_semantic_cache import ask, build

FAMILY = json.load(open("tests/data/familymembers.json", encoding="utf-8"))
REL = family_relation()


def members(resident_id):
    return [m for m in FAMILY if m["residentId"] == resident_id]


# ======================================================================== 1. derivation rules


def test_relation_normalisation_and_self_exclusion():
    d = REL.derive(members("RES-1001"))
    assert d["familyCount"] == 3  # "Self" is the resident, not a family member
    assert d["familyRelations"] == ["daughter", "son", "spouse"]  # " spouse " trimmed + lower-cased
    assert d["familyTypes"] == ["emergency", "family"]
    assert d["familyFlags"] == ["portal_access", "guarantor"]  # Eleanor's own portal access (Self) does not count
    assert d["familyMembers"] == ["Margaret Vance (Daughter)", "Thomas Vance (Son)", "Peter Vance (spouse)"]


def test_aliases_and_empty_relation():
    assert REL.derive(members("RES-1002"))["familyRelations"] == ["daughter-in-law", "granddaughter"]  # typo fixed
    d = REL.derive(members("RES-1003"))
    assert d["familyCount"] == 3 and d["familyRelations"] == ["brother", "sister"]  # empty relation: member, no value
    assert d["familyMembers"][-1] == "Kim Wells"


def test_no_members_and_cap():
    assert REL.derive([]) == {"familyCount": 0, "familyRelations": [], "familyTypes": [], "familyFlags": [],
                              "familyMembers": [], "familyContacts": [], "familyIds": []}  # fmt: skip
    many = [{"_id": f"x{i:02d}", "name": f"P{i}", "relation": "Son"} for i in range(30)]
    assert REL.derive(many)["familyCount"] == REL.cap == 20


@pytest.mark.parametrize(
    ("word", "values"),
    [("spouse", ["spouse", "wife", "husband"]), ("children", ["son", "daughter", "child"]), ("daughters", ["daughter"]),
     ("Mum", ["mother"]), ("in-laws", ["daughter-in-law", "son-in-law", "sister-in-law", "brother-in-law"]),
     ("Granddaugter", ["granddaughter"]), ("brother", ["brother"])],
)  # fmt: skip
def test_question_words_expand_to_relation_values(word, values):
    assert REL.expand(word) == values


# ======================================================================== 2. MongoDB stages (same rules)


def test_family_join_is_added_only_when_used():
    view = ResidentView()
    plain = view.pipeline(Filters.model_validate({"status": {"include": ["active"]}}))
    assert not any("$lookup" in st and st["$lookup"]["from"] == "familymembers" for st in plain)
    fam = view.pipeline(Filters.model_validate({"family_count": {"min": 1}}))
    lookup = next(st["$lookup"] for st in fam if "$lookup" in st and st["$lookup"]["from"] == "familymembers")
    assert lookup["localField"] == "_id" and lookup["foreignField"] == "residentId"  # ObjectId to ObjectId
    sub = json.dumps(lookup["pipeline"])
    assert '"$nin": ["self"]' in sub and '"$limit": 20' in sub and "granddaugter" in sub
    project = next(st["$project"] for st in fam if "$project" in st and "familyCount" in st["$project"])
    assert set(FAMILY_VIEW_FIELDS) <= set(project)


def test_one_resident_lookup_joins_family_and_plain_lists_do_not():
    view = ResidentView()
    assert any("$lookup" in st and st["$lookup"]["from"] == "familymembers" for st in view.id_pipeline(["X"]))
    assert view.display_stage() is not None and "familyMembers" not in json.dumps(view.display_stage())


def test_mongo_filters_for_family_fields():
    q = to_mongo_filter(Filters.model_validate({"family_relation": {"include": ["daughter"]}, "family_count": {"min": 2}}))
    assert q["familyRelations"] == {"$in": ["daughter"]}
    assert q["familyCount"]["$gte"] == 2
    none = to_mongo_filter(Filters.model_validate({"family_count": {"max": 0}}))
    assert none["familyCount"]["$lte"] == 0


def test_plan_uses_family():
    assert not plan_uses_family(Filters.model_validate({"status": {"include": ["active"]}}))
    assert plan_uses_family(Filters.model_validate({"family_flags": {"include": ["portal_access"]}}))
    assert plan_uses_family(None, ["familyMembers"]) and plan_uses_family(None, None, "family_count")


# ======================================================================== 3. filters in memory


@pytest.mark.parametrize(
    ("filters", "ids"),
    [
        ({"family_relation": {"include": ["daughter"]}}, {"RES-1001"}),
        ({"family_relation": {"include": ["spouse", "wife", "husband"]}}, {"RES-1001", "RES-1004"}),
        ({"family_flags": {"include": ["portal_access"]}}, {"RES-1001", "RES-1004"}),
        ({"family_type": {"include": ["emergency"]}}, {"RES-1001", "RES-1004"}),
        ({"family_count": {"min": 3}}, {"RES-1001", "RES-1003"}),
        ({"family_count": {"min": 1}, "family_relation": {"exclude": ["daughter"]}}, {"RES-1002", "RES-1003", "RES-1004"}),
    ],
)  # fmt: skip
async def test_family_filters(filters, ids):
    rows = await MemoryResidentRepository().find(Filters.model_validate(filters))
    assert {r["residentId"] for r in rows} == ids


@pytest.mark.parametrize(
    ("flag", "ids"),
    [
        ("has_email", {"RES-1001"}),  # only Margaret (Eleanor's daughter) has an email recorded
        ("has_phone", {"RES-1001", "RES-1002", "RES-1003", "RES-1004"}),  # Eleanor's own "Self" contact does not count
    ],
)
async def test_contact_flags_filter_on_a_recorded_contact(flag, ids):
    """"residents with a family member whose email is registered": the yes/no only, from a non-blank contact."""
    rows = await MemoryResidentRepository().find(Filters.model_validate({"family_flags": {"include": [flag]}}))
    assert {r["residentId"] for r in rows} == ids


def test_contact_flags_ignore_blank_contacts():
    d = REL.derive([{"_id": "a", "relation": "Son", "email": "  ", "phone": None}])
    assert d["familyFlags"] == []


def test_contact_flags_in_mongo_stages_and_prompt():
    stage = json.dumps(REL.lookup_stage)
    assert '"f_has_email": {"$gt": [{"$strLenCP"' in stage
    assert to_mongo_filter(Filters.model_validate({"family_flags": {"include": ["has_email"]}})) == {
        "familyFlags": {"$in": ["has_email"]}
    }
    ctx = SchemaContext(["active"], ["memory_care"], [], family_relations=["daughter"])
    assert 'family_flags eq "has_email"' in build_system_prompt(ctx, operator_table=True)  # Bedrock prompt


async def test_no_family_members_and_orphans():
    repo = MemoryResidentRepository()
    assert await repo.count(Filters.model_validate({"family_count": {"max": 0}})) == 84
    assert await repo.count(Filters.model_validate({"family_count": {"min": 1}})) == 4  # the orphan fm11 is nobody's


def test_bulk_rows_never_carry_family_contacts():
    r = {**{"residentId": "R"}, **REL.derive(members("RES-1001")), "emergencyContact": {}}
    shown = list_view(r)
    assert shown["familyCount"] == 3  # a number, not personal data
    assert not any(k in shown for k in FAMILY_VIEW_FIELDS if k != "familyCount")


async def test_projected_family_members_are_names_and_relations_only():
    rows = await MemoryResidentRepository().find(
        Filters.model_validate({"family_count": {"min": 1}}), fields=["familyMembers", "familyCount"]
    )
    eleanor = next(r for r in rows if r["residentId"] == "RES-1001")
    assert eleanor["familyMembers"] == "Margaret Vance (Daughter), Thomas Vance (Son), Peter Vance (spouse)"
    assert "familyContacts" not in eleanor and "555" not in json.dumps(rows)


# ======================================================================== 4. plan model and compiler (negatives)


@pytest.mark.parametrize(
    "filters",
    [
        {"family_type": {"include": ["neighbour"]}},  # not a contact type
        {"family_flags": {"include": ["admin"]}},  # not a flag
        {"family_count": {"min": 3, "max": 1}},
        {"family_count": {"min": -1}},
        {"family_count": {"missing": True}},  # a count is always recorded
        {"family_relation": {"starts_with": ["d"]}},  # text matching stays on room / facility / caregiver
    ],
)
def test_invalid_family_filters_are_rejected(filters):
    with pytest.raises(ValidationError):
        Filters.model_validate(filters)


def test_family_contacts_are_not_a_projectable_field():
    with pytest.raises(ValidationError):
        QueryPlan.model_validate({"action": "list", "fields": ["family_contacts"]})


def P(evidence, field, op, value):
    return {"kind": "predicate", "status": "supported", "evidence": evidence, "field": field, "operator": op, "value": value}


def IR(where, action="count"):
    return {"action": action, "action_status": "supported", "action_evidence": "How many", "where": where,
            "unsupported": [], "unresolved": [], "ambiguous": []}  # fmt: skip


def test_compiles_family_conditions():
    q = "How many residents have exactly 2 family members"
    assert compile_request_ir(IR(P("exactly 2 family members", "family_count", "eq", 2)), q)[1]["filters"] == {
        "family_count": {"min": 2, "max": 2}
    }
    q = "How many residents have more than 2 family members"
    assert compile_request_ir(IR(P("more than 2", "family_count", "gt", 2)), q)[1]["filters"] == {"family_count": {"min": 3}}


def test_family_operator_errors_are_precise():
    with pytest.raises(IRCompilationError) as exc:
        compile_request_ir(IR(P("How many", "family_count", "missing", None)), "How many")
    assert exc.value.code == "operator" and "family_count" in exc.value.details[0]


def test_detailed_prompt_lists_the_live_relation_labels_only_for_bedrock():
    ctx = SchemaContext(["active"], ["memory_care"], [], family_relations=["daughter", "son"])
    assert "family_relation: daughter | son" in build_system_prompt(ctx, operator_table=True)
    assert "family_relation:" not in build_system_prompt(ctx)  # Qwen's grammar carries the field names


# ======================================================================== 5. end to end (service, fake model)


async def test_count_residents_with_a_daughter(make_service):
    q = "How many residents have a daughter listed as family?"
    service = await make_service(FakeTranslator({q: {"action": "count", "filters": {"family_relation": {"include": ["daughter"]}}}}))
    r = await service.handle(q)
    assert r["response"]["count"] == 1 and "Family: daughter" in r["interpretation"]


async def test_spouse_matches_wife_and_spouse(make_service):
    q = "How many residents have a spouse in their family contacts?"
    service = await make_service(FakeTranslator({q: {"action": "count", "filters": {"family_relation": {"include": ["spouse"]}}}}))
    assert (await service.handle(q))["response"]["count"] == 2


async def test_residents_with_no_family_members(make_service):
    q = "How many residents have no family members?"
    service = await make_service(FakeTranslator({q: {"action": "count", "filters": {"family_count": {"max": 0}}}}))
    r = await service.handle(q)
    assert r["response"]["count"] == 84 and "Family: none recorded" in r["interpretation"]


async def test_one_residents_family_shows_contacts(make_service):
    q = "Who are Eleanor Vance's family members?"
    plan = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["family_members"]}}
    fake = FakeTranslator({"Who are resident_1's family members?": plan})
    r = await (await make_service(fake)).handle(q)
    text = json.dumps(r["response"])
    assert "Margaret Vance (Daughter)" in text and "555-0101" in text and "margaret.vance@example.com" in text
    assert "555-0100" not in text  # the resident's own "Self" contact is not a family member
    assert not fake.calls or "Margaret" not in json.dumps(fake.calls)  # family data is never sent to the model


async def test_son_phone_question_is_answered_from_family_details(make_service):
    q = "What is Eleanor Vance's son's phone number?"
    plan = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["family_members"]}}
    r = await (await make_service(FakeTranslator({"What is resident_1's son's phone number?": plan}))).handle(q)
    assert not r["response"].get("needsClarification") and "555-0102" in json.dumps(r["response"])


async def test_emergency_contact_shows_both_sources(make_service):
    q = "Who is Harold Finch's emergency contact?"
    plan = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["emergency_contact"]}}
    r = await (await make_service(FakeTranslator({"Who is resident_1's emergency contact?": plan}))).handle(q)
    text = json.dumps(r["response"])
    assert "Family emergency contacts: Grace Finch (Wife)" in text


async def test_list_with_family_column_has_no_contact_details(make_service):
    q = "List residents with their family members"
    plan = {"action": "list", "filters": {"family_count": {"min": 1}}, "fields": ["family_members"]}
    r = await (await make_service(FakeTranslator({q: plan}))).handle(q)
    text = json.dumps(r["response"])
    assert r["response"]["count"] == 4 and "Thomas Vance (Son)" in text and "555-01" not in text


async def test_resident_with_most_family_members(make_service):
    q = "Which resident has the most family members?"
    plan = {"action": "list", "sort": {"field": "family_count", "direction": "desc"}, "limit": 1, "fields": ["family_count"]}
    r = await (await make_service(FakeTranslator({q: plan}))).handle(q)
    assert not r["response"].get("needsClarification")
    assert r["response"]["residentsSample"][0]["residentId"] in ("RES-1001", "RES-1003")


# ======================================================================== 6. grounding


async def test_a_family_question_answered_without_family_is_caught(make_service):
    q = "How many residents have a daughter listed as family?"
    dropped = {"action": "count"}
    fake = FakeTranslator({q: [dropped, dropped]})
    r = await (await make_service(fake)).handle(q)
    assert len(fake.calls) == 2 and "family" in (fake.calls[1]["hint"] or "").lower()
    assert r["response"].get("needsClarification")  # never "all 88 residents"


async def test_an_invented_family_condition_is_caught(make_service):
    q = "How many active residents are there?"
    invented = {"action": "count", "filters": {"status": {"include": ["active"]}, "family_count": {"min": 1}}}
    fake = FakeTranslator({q: [invented, invented]})
    r = await (await make_service(fake)).handle(q)
    assert len(fake.calls) == 2 and r["response"].get("needsClarification")


def test_describe_chips():
    chips = describe_filters(Filters.model_validate(
        {"family_relation": {"include": ["daughter"]}, "family_flags": {"include": ["portal_access"]}, "family_count": {"min": 3}}
    ))  # fmt: skip
    assert "Family: daughter" in chips and "Family with: portal access" in chips and "Family members: at least 3" in chips


# ======================================================================== 7. template cache


async def test_family_plans_are_never_stored_as_templates(repo, settings):
    q = "How many residents have at least 3 family members?"
    service, fake, store = await build(repo, settings, {q: {"action": "count", "filters": {"family_count": {"min": 3}}}})
    r = await ask(service, q)
    assert r["response"]["count"] == 2 and store.docs == {}


def test_matches_uses_the_same_family_fields():
    r = {"residentId": "R", **REL.derive(members("RES-1004"))}
    assert matches(r, Filters.model_validate({"family_relation": {"include": ["wife"]}}))
    assert not matches(r, Filters.model_validate({"family_count": {"min": 2}}))
