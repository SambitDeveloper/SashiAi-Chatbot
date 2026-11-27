"""SnfMcpRepository without a server: admins → run_query; staff → only the API tools in staff_tools.yaml."""

import logging

import pytest

from app.data.repository import DataUnavailable, NotSupported
from app.data.snf_mcp_repository import SnfMcpRepository, use_credentials
from app.engine.executor import execute
from app.models.plan import Filters, QueryPlan

ADMIN = {"c_name": "adm", "groups": ["ADMIN"], "facility_id": "fac-1", "profile": None}
STAFF = {"c_name": "c-n", "groups": ["STAFF"], "facility_id": "fac-1",
         "profile": {"name": "Sara Nurse", "designation": "Nurse", "active": True, "access_permissions": []}}  # fmt: skip
CASELOAD = [
    {"name": "Ann Lee", "unit": "101", "status": "Active", "admission_date": "2026-01-02",
     "primary_physician": "Dan Doc", "assigned_staff": [{"name": "Sara Nurse", "role": "Nurse"},
                                                        {"name": "Dan Doc", "role": "Physician"}],
     "family_members": [{"name": "Bo Lee", "role": "Son"}]},
    {"name": "Tom Ray", "unit": "102", "status": "Discharged", "admission_date": "2026-02-03",
     "primary_physician": None, "assigned_staff": [], "family_members": []},
]  # fmt: skip
API_TOOLS = {"list_my_residents", "find_staff", "get_caller"}


def _repo(caller, tools=API_TOOLS, log_responses=False, caseload=CASELOAD):
    repo = SnfMcpRepository("http://mcp.test/mcp", 5, log_responses=log_responses)
    use_credentials("token", "fac-1")  # what api/routes.py does for every request
    sent: list = []

    async def call(tool, args):
        sent.append((tool, args))
        return {
            "get_caller": caller,
            "run_query": {"caller": caller, "documents": [{"n": 3}]},
            "get_field_values": {"values": {"status": ["Active"], "careType": ["memory_care"]}},
            "list_my_residents": {"residents": caseload, "has_more": False},
            "find_staff": {"staff": [{"name": f"Dr {args.get('designation')}", "designation": args.get("designation")}],
                           "has_more": False},
        }[tool]  # fmt: skip

    async def list_tools():
        return set(tools)

    repo._call, repo._list_tools = call, list_tools
    return repo, sent


def _plan(**p) -> QueryPlan:
    return QueryPlan.model_validate(p)


async def test_an_admin_question_is_a_run_query_aggregation():
    repo, sent = _repo(ADMIN)
    data = await execute(_plan(action="count", filters={"status": {"include": ["active"]}}), repo)
    assert data.matched == 3
    queries = [args for tool, args in sent if tool == "run_query"]
    assert queries and all(q["collection"] == "residents" for q in queries)
    assert {"status": {"$in": ["active"]}} in [s.get("$match") for s in queries[-1]["pipeline"]]
    assert not [t for t, _ in sent if t == "list_my_residents"]


async def test_admins_learn_every_status_and_care_type_of_the_facility():
    repo, _ = _repo(ADMIN)
    vocab = await repo.vocabulary()
    assert "memory_care" in vocab.care_types and "active" in vocab.statuses


async def test_a_staff_question_runs_on_their_caseload_from_the_api_tool():
    repo, sent = _repo(STAFF)
    data = await execute(_plan(action="count", filters={"status": {"include": ["active"]}}), repo)
    assert (data.matched, data.total) == (1, 2)
    assert ("list_my_residents", {"status": "All", "page": 1}) in sent
    assert not [t for t, _ in sent if t in ("run_query", "get_field_values")]  # staff never reach the database
    [ann] = await repo.find_by_ids([r["residentId"] for r in await repo.find(Filters()) if r["firstName"] == "Ann"])
    assert ann["careTeam"] == ["Dan Doc (Physician)", "Sara Nurse (Nurse)"]
    assert ann["familyMembers"] == ["Bo Lee (Son)"] and ann["roomNumber"] == "101"


async def test_staff_still_understand_a_status_none_of_their_residents_has():
    vocab = await _repo(STAFF)[0].vocabulary()
    assert "away" in vocab.statuses  # understood, then refused by only_values (not a wrong 0)


@pytest.mark.parametrize(
    "plan",
    [
        {"action": "count", "filters": {"care_type": {"include": ["memory_care"]}}},
        {"action": "aggregate", "aggregate": {"op": "avg", "field": "age"}},
        {"action": "breakdown", "group_by": ["gender"]},
    ],
)
async def test_a_staff_plan_using_a_field_the_api_tool_lacks_is_not_supported(plan):
    with pytest.raises(DataUnavailable, match="not available to staff"):
        await execute(_plan(**plan), _repo(STAFF)[0])


async def test_a_staff_question_without_its_tool_in_snf_mcp_is_not_supported():
    repo, _ = _repo(STAFF, tools={"get_caller"})
    with pytest.raises(DataUnavailable, match="no snf_mcp tool"):
        await execute(_plan(action="count"), repo)


async def test_staff_directory_questions_call_find_staff_per_designation_of_the_role():
    repo, sent = _repo(STAFF)
    plan = _plan(action="count", entity="staff", filters={"staff_role": {"include": ["physician"]}})
    data = await execute(plan, repo)
    assert sorted(a["designation"] for t, a in sent if t == "find_staff") == ["Doctor", "Physician"]
    assert data.matched == 2
    with pytest.raises(DataUnavailable, match="need a role"):
        await execute(_plan(action="count", entity="staff"), repo)


@pytest.mark.parametrize("enabled", [False, True])
async def test_calls_are_logged_only_when_enabled(caplog, enabled):
    repo = SnfMcpRepository("http://mcp.test/mcp", 5, log_responses=enabled)
    use_credentials("token", "fac-1")

    async def session(action):
        class Result:
            is_error, structured_content, content = False, ADMIN, []

        return Result()

    repo._session = session
    with caplog.at_level(logging.INFO, logger="app.data.snf_mcp_repository"):
        await repo.caller()
    logged = [r for r in caplog.records if r.getMessage() == "snf_mcp response"]
    assert len(logged) == int(enabled)


def test_every_route_answers_503_not_500_when_snf_mcp_is_unavailable(settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    repo, _ = _repo(ADMIN)

    async def down(tool, args):  # e.g. SNF_MCP_URL pointing at /api: 404 Not Found
        raise DataUnavailable("snf_mcp call to http://127.0.0.1:7100/api failed: MCPError: Not Found")

    repo._call = down
    signed_in = {"Authorization": "Bearer token", "x-facility-id": "fac-1"}
    with TestClient(create_app(settings=settings, repo=repo, use_default_translator=False)) as c:  # startup survives
        for r in (c.get("/api/meta", headers=signed_in),
                  c.post("/api/query", json={"query": "How many residents are there?"}, headers=signed_in)):  # fmt: skip
            assert r.status_code == 503 and "isn't supported" in r.json()["error"]


@pytest.mark.parametrize("statuses", [["away"], ["active", "away"]])
async def test_a_status_the_staff_tool_never_returns_is_not_supported(statuses):
    plan = _plan(action="count", filters={"status": {"include": statuses}})
    with pytest.raises(DataUnavailable, match="status"):
        await execute(plan, _repo(STAFF)[0])
    assert (await execute(_plan(action="count", filters={"status": {"include": ["discharged"]}}), _repo(STAFF)[0])).matched == 1


async def test_staff_are_offered_follow_ups_their_tools_can_answer(settings):
    from app.auth.identity import principal_from_caller
    from app.engine.responder import HELP_OPTIONS
    from app.engine.service import QueryService
    from tests.conftest import FakeTranslator

    q = "How many residents are in skilled nursing?"
    repo, _ = _repo(STAFF)
    service = QueryService(repo, FakeTranslator({q: {"action": "count", "filters": {"care_type": {"include": ["skilled_nursing"]}}}}),
                           settings)  # fmt: skip
    await service.refresh_vocabulary()
    perms = [{"name": "Residents", "allowed": True, "isRead": True, "children": [
        {"name": "residents-list", "allowed": True, "isRead": True},
        {"name": "resident-details", "allowed": True, "isRead": True}]}]  # fmt: skip
    caller = {**STAFF, "profile": {**STAFF["profile"], "access_permissions": perms}}
    staff = await service.handle(q, principal=principal_from_caller(caller))
    assert staff["response"]["headline"] == "Not supported yet"
    # a data limitation, not an authorization denial (the caller has the Residents permissions)
    assert staff["policy"] == {"passed": True, "policyCategory": "NOT_SUPPORTED", "violationReason": None}
    assert staff["intent"]["category"] == "NOT_SUPPORTED"
    assert staff["response"]["isPolicyViolation"] is False and staff["response"]["isNotSupported"] is True
    assert staff["response"]["summaryText"].startswith("Care Type data isn't available to the assistant for your role")
    options = staff["response"]["clarificationOptions"]
    assert options == repo.staff_suggestions[: len(options)] and not any("care type" in o["query"] for o in options)
    admin = await service.handle(q, principal=principal_from_caller({**ADMIN, "facility_id": "fac-1"}))
    assert all(o in HELP_OPTIONS for o in admin["response"].get("clarificationOptions") or HELP_OPTIONS)


async def test_genuine_policy_refusals_keep_their_classification(settings):
    from app.auth.identity import principal_from_caller
    from app.engine.service import QueryService
    from tests.conftest import FakeTranslator

    repo, _ = _repo(STAFF)
    active = {"action": "count", "filters": {"status": {"include": ["active"]}}}
    service = QueryService(repo, FakeTranslator({"How many residents are active?": active}), settings)
    await service.refresh_vocabulary()
    perms = [{"name": "Residents", "allowed": True, "isRead": True,
              "children": [{"name": "residents-list", "allowed": True, "isRead": True}]}]  # fmt: skip
    staff = principal_from_caller({**STAFF, "profile": {**STAFF["profile"], "access_permissions": perms}})
    meds = await service.handle("What medications are residents taking?", principal=staff)
    assert meds["policy"]["policyCategory"] == "OUT_OF_SCOPE" and meds["response"]["isPolicyViolation"] is True

    async def refused(plan):  # snf_mcp refused the caller: not a data limitation, never relabelled NOT_SUPPORTED
        raise DataUnavailable("Error executing tool get_caller: Facility not permitted for this user.")

    repo.for_plan = refused
    r = await service.handle("How many residents are active?", principal=staff)
    assert r["policy"]["policyCategory"] != "NOT_SUPPORTED" and not r["response"].get("isNotSupported")


# ======================================================================== care type for staff: off until the API sends it

WITH_CARE = [{**CASELOAD[0], "care_type": "memory_care"}, {**CASELOAD[1], "care_type": "skilled_nursing"}]


async def test_care_type_stays_off_while_the_staff_api_does_not_send_it():
    plan = _plan(action="count", filters={"care_type": {"include": ["memory_care"]}})
    with pytest.raises(NotSupported) as exc:
        await execute(plan, _repo(STAFF)[0])  # today's backend: no careType in /staff/my-residents
    assert exc.value.fields == ("care_type",)


async def test_care_type_stays_off_while_any_assigned_resident_has_none():
    partial = [WITH_CARE[0], {**CASELOAD[1], "care_type": None}]
    with pytest.raises(NotSupported):  # a missing value is never read as "does not match"
        await execute(_plan(action="count", filters={"care_type": {"include": ["memory_care"]}}),
                      _repo(STAFF, caseload=partial)[0])  # fmt: skip


async def test_once_sent_care_type_filters_counts_lists_and_groups_the_staff_members_residents():
    repo, sent = _repo(STAFF, caseload=WITH_CARE)
    count = await execute(_plan(action="count", filters={"care_type": {"include": ["memory_care"]}}), repo)
    assert (count.matched, count.total) == (1, 2)  # of the caller's 2 assigned residents, not the facility
    listed = await execute(_plan(action="list", filters={"care_type": {"include": ["skilled_nursing"]}}), repo)
    assert [r["firstName"] for r in listed.rows] == ["Tom"]
    groups = await execute(_plan(action="breakdown", group_by=["care_type"]), repo)
    assert sorted((g.key, g.count) for g in groups.groups) == [(("memory_care",), 1), (("skilled_nursing",), 1)]
    assert not [t for t, _ in sent if t in ("run_query", "get_field_values")]  # still only the API tool


# ======================================================================== staff: age and dates from the API tool

DATED = [{**CASELOAD[0], "created_date": "2025-12-01", "age": 82},
         {**CASELOAD[1], "created_date": "2025-10-01", "birth_date": "1950-01-01", "discharge_date": "2026-03-01"}]  # fmt: skip


async def test_staff_list_by_admission_and_created_date_now():
    repo, _ = _repo(STAFF)  # today's API: admission dates (dateOfJoining); no created dates in CASELOAD
    joined = await execute(_plan(action="list", filters={"admission_date": {"start": "2026-02-01"}}), repo)
    assert [r["firstName"] for r in joined.rows] == ["Tom"]
    created = await execute(_plan(action="count", filters={"created_date": {"start": "2025-11-01"}}), _repo(STAFF, caseload=DATED)[0])
    assert created.matched == 1
    with pytest.raises(NotSupported, match="created_date"):  # not sent (an old snf_mcp): never a wrong 0
        await execute(_plan(action="count", filters={"created_date": {"start": "2025-11-01"}}), repo)


async def test_staff_age_and_discharge_date_switch_on_only_when_sent_for_everyone_who_needs_them():
    over = _plan(action="list", filters={"age": {"min": 80}})
    with pytest.raises(NotSupported, match="age"):
        await execute(over, _repo(STAFF)[0])  # today's API: no age or birth date
    rows = (await execute(over, _repo(STAFF, caseload=DATED)[0])).rows  # age, or computed from the birth date
    assert [r["firstName"] for r in rows] == ["Ann"]
    discharged = _plan(action="count", filters={"discharge_date": {"start": "2026-01-01"}})
    assert (await execute(discharged, _repo(STAFF, caseload=DATED)[0])).matched == 1  # only Tom is discharged
    no_date = [DATED[0], {**DATED[1], "discharge_date": None}]  # a discharged resident without one
    with pytest.raises(NotSupported, match="discharge_date"):
        await execute(discharged, _repo(STAFF, caseload=no_date)[0])


# ======================================================================== the token and facility come from the UI


def test_a_request_without_the_users_token_or_facility_is_refused(settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    repo, sent = _repo(ADMIN)
    with TestClient(create_app(settings=settings, repo=repo, use_default_translator=False)) as c:
        sent.clear()  # only what the requests below cause
        for headers in ({}, {"Authorization": "Bearer t"}, {"x-facility-id": "fac-1"}, {"Authorization": "t", "x-facility-id": "f"}):
            r = c.post("/api/query", json={"query": "How many residents are there?"}, headers=headers)
            assert r.status_code == 401 and "Sign in to use the assistant" in r.json()["error"]
    assert sent == []  # nothing reached snf_mcp


def test_each_request_uses_its_own_users_token_facility_and_caches():
    import asyncio

    repo = SnfMcpRepository("http://mcp.test/mcp", 5)
    seen: list = []

    async def session(action):  # what snf_mcp received: the request's own headers
        headers = repo._headers()
        seen.append(headers)
        admin = headers["Authorization"] == "Bearer admin-token"

        class Result:
            is_error, content = False, []
            structured_content = ADMIN if admin else STAFF

        return Result()

    repo._session = session

    async def ask(token, facility):
        use_credentials(token, facility)  # each request runs in its own context
        return (await repo.caller())["groups"]

    async def both():
        return await asyncio.gather(ask("admin-token", "fac-1"), ask("staff-token", "fac-2"), ask("admin-token", "fac-1"))

    assert asyncio.run(both()) == [["ADMIN"], ["STAFF"], ["ADMIN"]]
    assert {(h["Authorization"], h["x-facility-id"]) for h in seen} == {("Bearer admin-token", "fac-1"),
                                                                       ("Bearer staff-token", "fac-2")}  # fmt: skip
    assert len(seen) == 2  # the repeated admin call came from that user's own cache entry
