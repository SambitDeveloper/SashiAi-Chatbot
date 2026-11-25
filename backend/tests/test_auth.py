"""Authorization guardrails (app/auth; design "SashiAI Chatbot — Authorization Guardrails", threat matrix).

Fixture: the 88 fictional residents, all moved to facility FAC-A; the staff member "s-nurse" is assigned to
RES-1001 (Eleanor Vance) and RES-1004 (Harold Finch) only. Model plans come from FakeTranslator: a question that
must be refused before the model has NO fake plan, so any model call fails the test.
"""

import json
import os

import pytest

from app.auth.gate import check_question
from app.auth.guard import AccessDenied, check_plan, required_capabilities, use_principal
from app.auth.identity import AuthError, IdentityResolver, principal_from_context
from app.auth.policy import access_policy, norm_module, resolve_ref
from app.auth.principal import SYSTEM, Principal, readable_modules
from app.config import Settings
from app.data import scope
from app.data.memory_repository import MemoryResidentRepository
from app.engine.service import QueryService
from app.models.plan import Filters, QueryPlan
from tests.conftest import FakeTranslator

ALL = ["Residents", "residents-list", "resident-details", "Contact Management", "Staff", "Access Management"]
NURSE = Principal(kind="staff", id="s-nurse", facility_id="FAC-A", scope_key="s-nurse",
                  modules=frozenset(norm_module(m) for m in ALL))  # fmt: skip
LIST_ONLY = Principal(kind="staff", id="s-list", facility_id="FAC-A", scope_key="s-nurse",
                      modules=frozenset(norm_module(m) for m in ["Residents", "residents-list"]))  # fmt: skip
NOTHING = Principal(kind="staff", id="s-none", facility_id="FAC-A", scope_key="s-nurse")
NO_RESIDENTS = Principal(kind="staff", id="s-new", facility_id="FAC-A", scope_key="s-new",
                         modules=frozenset(norm_module(m) for m in ALL))  # fmt: skip
ADMIN = Principal(kind="admin", id="a-1", facility_id="FAC-A")


def _repo() -> MemoryResidentRepository:
    repo = MemoryResidentRepository()
    for r in repo._records:
        r["facilityId"] = "FAC-A"
        r["assignedStaff"] = ["s-nurse"] if r["residentId"] in ("RES-1001", "RES-1004") else []
    return repo


async def _service(settings, plans=None) -> tuple[QueryService, FakeTranslator]:
    fake = FakeTranslator(plans or {})
    service = QueryService(_repo(), fake, settings)
    await service.refresh_vocabulary()
    return service, fake


def _ids(r) -> set[str]:
    return {x["residentId"] for x in r["response"].get("residentsSample") or []}


# ======================================================================== D. row scope (R3, R4)


async def test_a_staff_member_counts_and_lists_only_their_residents(settings):
    count = {"action": "count"}
    service, _ = await _service(settings, {"How many residents are there?": count,
                                          "List all residents": {"action": "list"}})  # fmt: skip
    assert (await service.handle("How many residents are there?", principal=ADMIN))["response"]["count"] == 88
    assert (await service.handle("How many residents are there?", principal=NURSE))["response"]["count"] == 2
    listed = await service.handle("List all residents", principal=NURSE)
    assert _ids(listed) == {"RES-1001", "RES-1004"}  # R4: "all" = their assigned residents


async def test_a_resident_outside_scope_is_not_found_before_any_model_call(settings):
    service, fake = await _service(settings)  # no fake plan: a model call fails the test
    r = await service.handle("What room is Arthur Pendleton in?", principal=NURSE)  # RES-1002: not assigned
    assert fake.calls == [] and r["tokensConsumed"] == 0
    assert r["response"]["summaryText"] == access_policy().message("resident_not_in_scope")
    assert "MC-104" not in str(r)  # never the resident's data


async def test_their_own_resident_is_answered(settings):
    room = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["room_number"]}}
    service, _ = await _service(settings, {"What room is resident_1 in?": room})
    r = await service.handle("What room is Eleanor Vance in?", principal=NURSE)
    assert r["response"]["headline"].startswith("MC-101")


async def test_no_assigned_residents(settings):
    service, fake = await _service(settings)
    r = await service.handle("How many residents are there?", principal=NO_RESIDENTS)
    assert fake.calls == [] and r["response"]["summaryText"] == access_policy().message("no_residents")


def test_mongo_row_scope_is_facility_and_assignment():
    with scope.facility_scope({"FAC-A"}), scope.assignment_scope(NURSE.assignment()):
        assert scope.mongo_match() == {"$match": {"facilityId": {"$in": ["FAC-A"]}, "assignedStaff": "s-nurse"}}
        assert scope.record_in_scope({"facilityId": "FAC-A", "assignedStaff": ["x", "s-nurse"]})
        assert not scope.record_in_scope({"facilityId": "FAC-A", "assignedStaff": ["x"]})
        assert not scope.record_in_scope({"facilityId": "FAC-B", "assignedStaff": ["s-nurse"]})
    assert scope.mongo_match() is None  # unrestricted again after the request


async def test_the_data_browser_is_scoped(settings):
    service, _ = await _service(settings)
    rows = await service.browse(Filters(), None, 500, NURSE)
    assert {r["residentId"] for r in rows} == {"RES-1001", "RES-1004"}
    with pytest.raises(AccessDenied):
        await service.browse(Filters(), None, 500, NOTHING)


# ======================================================================== E. pre-model gate (0 tokens)


async def test_family_without_resident_details_is_refused_before_the_model(settings):
    service, fake = await _service(settings)
    r = await service.handle("How many residents have a daughter listed?", principal=LIST_ONLY)
    assert fake.calls == [] and r["tokensConsumed"] == 0
    assert r["policy"]["policyCategory"] == "ACCESS_DENIED" and r["authorization"]["layer"] == "gate"
    assert r["response"]["summaryText"] == "You don't have access to family members in Shashi Care, so I can't answer that."


@pytest.mark.parametrize("question", ["What is the room number of Eleanor Vance?", "Who are Eleanor Vance's family members?",
                                      "Is Eleanor Vance active?"])  # fmt: skip
async def test_one_resident_needs_resident_details(settings, question):
    """Decision 2026-11-24: residents-list alone = counts, lists and groupings; anything about ONE resident (even the
    room, a list-page field) needs resident-details — refused before any model call."""
    service, fake = await _service(settings)
    r = await service.handle(question, principal=LIST_ONLY)
    assert fake.calls == [] and r["authorization"]["capability"] == "resident_details"
    assert "MC-101" not in str(r)


async def test_lists_and_counts_need_only_the_residents_list(settings):
    service, _ = await _service(settings, {"List residents in memory care": {"action": "list",
                                           "filters": {"care_type": {"include": ["memory_care"]}}}})  # fmt: skip
    r = await service.handle("List residents in memory care", principal=LIST_ONLY)
    assert "authorization" not in r and r["response"]["count"] == 2


def test_a_lookup_plan_needs_resident_details():
    plan = QueryPlan.model_validate({"action": "lookup", "lookup": {"residents": ["RES-1001"], "attributes": ["room_number"]}})
    assert required_capabilities(plan) == ["residents", "resident_details"]


async def test_nothing_readable_refuses_resident_questions_before_the_model(settings):
    service, fake = await _service(settings)
    r = await service.handle("How many residents are in memory care?", principal=NOTHING)
    assert fake.calls == [] and r["authorization"]["capability"] == "residents"


async def test_app_permission_questions_need_access_management(settings):
    service, fake = await _service(settings)
    r = await service.handle("Can Priya Raman access the transport calendar?", principal=LIST_ONLY)
    assert fake.calls == [] and r["authorization"]["capability"] == "staff_permissions"


def test_portal_access_is_a_family_word_not_an_app_permission():
    gate = check_question("Which residents have a family member with portal access?", LIST_ONLY)
    assert gate.denied is not None and gate.denied.capability == "family"  # not staff_permissions


# ======================================================================== C. plan policy (every path)


NO_STAFF = Principal(kind="staff", id="s-cm", facility_id="FAC-A", scope_key="s-nurse",
                     modules=frozenset(norm_module(m) for m in ["Residents", "residents-list", "resident-details"]))  # fmt: skip


async def test_a_staff_directory_question_is_refused_before_the_model(settings):
    service, fake = await _service(settings)
    r = await service.handle("How many physicians are there?", principal=NO_STAFF)  # speaks of no resident
    assert fake.calls == [] and r["authorization"]["capability"] == "staff_directory"


async def test_a_plan_the_gate_could_not_place_is_refused_before_execution(settings):
    """A gate miss: the words allow a care-team reading, the model wrote a staff plan → refused at the plan check."""
    q = "Which physicians look after residents in memory care?"
    plan = {"action": "list", "entity": "staff", "filters": {"staff_role": {"include": ["physician"]},
                                                              "care_type": {"include": ["memory_care"]}}}  # fmt: skip
    service, fake = await _service(settings, {q: plan})
    r = await service.handle(q, principal=NO_STAFF)
    assert len(fake.calls) == 1 and r["authorization"] == {"decision": "deny", "layer": "plan", "capability": "staff_directory"}


@pytest.mark.parametrize(
    ("plan", "needed"),
    [
        ({"action": "count"}, ["residents"]),
        ({"action": "list", "filters": {"age": {"min": 80}}}, ["residents", "resident_details"]),
        ({"action": "count", "filters": {"family_relation": {"include": ["daughter"]}}}, ["residents", "family"]),
        ({"action": "count", "filters": {"staff_role": {"include": ["physician"]}}}, ["residents", "care_team"]),
        ({"action": "count", "entity": "staff"}, ["staff_directory"]),
        ({"action": "list", "entity": "staff", "fields": ["staff_resident_count"]}, ["staff_directory", "residents"]),
        ({"action": "count", "entity": "staff", "filters": {"care_type": {"include": ["memory_care"]}}},
         ["staff_directory", "residents"]),
        ({"action": "clarify", "message": "?"}, []),
    ],
)  # fmt: skip
def test_required_capabilities(plan, needed):
    assert required_capabilities(QueryPlan.model_validate(plan)) == needed


def test_check_plan_uses_the_request_principal():
    plan = QueryPlan.model_validate({"action": "count", "entity": "staff"})
    with use_principal(LIST_ONLY), pytest.raises(AccessDenied) as exc:
        check_plan(plan)
    assert exc.value.capability == "staff_directory" and exc.value.layer == "plan"
    with use_principal(ADMIN):
        check_plan(plan)  # R1: an admin needs no module


# ======================================================================== caches, sessions, policy integrity


async def test_caches_and_sessions_are_bound_to_the_caller(settings):
    service, _ = await _service(settings)
    contexts = set()
    for p in (SYSTEM, ADMIN, NURSE, LIST_ONLY):
        with use_principal(p):
            contexts.add(service.cache_context)
    assert len(contexts) == 4  # failure cache and conversation memory never cross callers


def test_every_plan_field_belongs_to_a_capability():
    policy = access_policy()
    names = set(Filters.model_fields) | set(QueryPlan.model_fields["fields"].annotation.__args__[0].__args__)
    for name in names:
        assert policy.capability_for_field(name, "resident") in policy.capabilities


def test_policy_references_resolve_and_link_the_resident_scope():
    assert resolve_ref("@schema:relations.staff.resident_link") == "assignedStaff"
    assert access_policy().gate  # every "@..." word list resolved at load


def test_readable_modules_need_read_and_the_parent():
    perms = [{"name": "Transport", "allowed": True, "isRead": True, "isWrite": False, "children": [
                 {"name": "transport-calendar", "allowed": True, "isRead": True},
                 {"name": "transport-requests", "allowed": True, "isRead": False, "isWrite": True}]},
             {"name": "Staff", "allowed": False, "isRead": True, "children": [{"name": "x", "allowed": True, "isRead": True}]}]  # fmt: skip
    assert readable_modules(perms, ("allowed", "isRead")) == {"transport", "transport calendar"}  # write grants nothing


# ======================================================================== identity (the app's user data)

DUMMY = "app/auth/dummy_user_context.json"  # edited by hand while testing: only checked to load
PROFILE = "tests/data/user_context_staff.json"  # the app's user data as first shared (Jone, Transport Driver)


def test_the_shipped_dummy_user_loads():
    assert principal_from_context(json.loads(open(DUMMY, encoding="utf-8").read())).kind in ("admin", "staff")


def _context(**changes) -> dict:
    data = json.loads(open(PROFILE, encoding="utf-8").read())
    data.update(changes)
    return data


def test_the_dummy_user_is_read_like_the_apps_user_data():
    staff = principal_from_context(_context(scope="staff"))
    assert (staff.kind, staff.facility_id, staff.scope_key) == ("staff", "S101", "83fd1390-6d21-4484-bf23-b1aebdf07205")
    # Residents with its 3 children, Transport with 2 (transport-calendar is off), Settings; never Staff etc.
    assert staff.can("residents") and staff.can("resident_details") and staff.can("family")
    assert not staff.can("staff_directory") and not staff.can("family_contacts") and not staff.can("staff_permissions")
    assert "transport calendar" not in staff.modules and "transport requests" in staff.modules
    admin = principal_from_context(_context(scope="admin"))
    assert admin.kind == "admin" and admin.unrestricted and admin.facilities() is None  # R1 (admin_facility_scope all)


@pytest.mark.parametrize(
    ("changes", "status"),
    [
        ({"scope": "visitor"}, 401),  # not a role the policy knows
        ({"scope": None}, 401),
        ({"cName": ""}, 401),
        ({"scope": "staff", "facilityId": None}, 401),  # a staff member needs an active facility
        ({"scope": "staff", "active": False}, 403),
    ],
)
def test_incomplete_or_inactive_users_are_refused(changes, status):
    with pytest.raises(AuthError) as exc:
        principal_from_context(_context(**changes))
    assert exc.value.status == status


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, llm_enabled=False, plan_cache_enabled=False, **kw)


async def test_file_mode_rereads_the_file_when_it_changes(tmp_path):
    f = tmp_path / "user.json"
    f.write_text(json.dumps(_context(scope="admin")))
    resolver = IdentityResolver(_settings(auth_mode="file", auth_user_context_file=str(f)))
    assert (await resolver.resolve()).kind == "admin"
    f.write_text(json.dumps(_context(scope="staff")))
    os.utime(f, (f.stat().st_atime, f.stat().st_mtime + 5))
    assert (await resolver.resolve()).kind == "staff"  # no restart needed to switch user or role
    f.unlink()
    with pytest.raises(AuthError):
        await resolver.resolve()


def test_the_dummy_user_never_reaches_production():
    with pytest.raises(ValueError, match="not allowed in production"):
        _settings(environment="production", auth_mode="file")


async def test_identity_off_is_todays_unrestricted_behaviour():
    assert await IdentityResolver(_settings()).resolve() is SYSTEM


async def test_every_answer_says_who_asked_and_staff_answers_are_marked_as_scoped(settings):
    service, _ = await _service(settings, {"How many residents are there?": {"action": "count"}})
    staff = await service.handle("How many residents are there?", principal=NURSE)
    assert staff["caller"] == {"role": "staff", "name": None, "facilityId": "FAC-A", "residentScope": "assigned"}
    assert staff["response"]["summaryText"].endswith(access_policy().message("scoped_note"))
    assert "census" not in staff["response"]["summaryText"]
    admin = await service.handle("How many residents are there?", principal=ADMIN)
    assert admin["caller"]["residentScope"] == "all" and access_policy().message("scoped_note") not in str(admin)
    off = await service.handle("How many residents are there?")
    assert off["caller"] == {"role": "system", "name": None, "facilityId": None, "residentScope": "all"}


async def test_a_value_none_of_their_residents_has_is_still_understood(settings):
    """Staging: "How many residents are in memory care?" for a staff member whose 10 residents are in other care
    types answered 10 — their view of the names had lost "memory care", so the condition was dropped. Only names are
    narrowed; values (care types, statuses, ...) stay shared, and the answer is 0."""
    q = "How many residents are in skilled nursing?"  # NURSE's two residents are both in memory care
    plan = {"action": "count", "filters": {"care_type": {"include": ["skilled_nursing"]}}}
    service, _ = await _service(settings, {q: plan})
    r = await service.handle(q, principal=NURSE)
    assert r["response"]["count"] == 0
    assert "skilled_nursing" in service.index.care_types  # the shared index (outside a request)


async def test_a_callers_index_narrows_names_only(settings):
    service, _ = await _service(settings)
    caller = NURSE.caller(service_caller := __import__("app.engine.caller", fromlist=["CallerContext"]).CallerContext())
    with scope.facility_scope(caller.facilities), scope.assignment_scope(NURSE.assignment()):
        mine = await service._index_for(NURSE, caller)
    assert set(mine.display_by_id) == {"RES-1001", "RES-1004"}  # names: their residents only
    assert set(mine.care_types) == set(service._index.care_types)  # values: shared
    assert set(mine.statuses) == set(service._index.statuses) and mine.wings == service._index.wings
    assert service_caller.facilities is None


@pytest.mark.parametrize("question", ["How many residents are in skilled nursing?", "List residents in memory care",
                                      "How many residents are on leave?"])  # fmt: skip
def test_care_types_and_statuses_are_not_staff_words(question):
    """Staging: "skilled nursing" was refused as the staff role "nurse" (care teams) for a residents-list user."""
    assert check_question(question, LIST_ONLY).denied is None
