"""Questions about ONE named person — "Who is X?", "What is X's designation?", "Is X active?", "Which facility does X
belong to?", "What is X's role?", "What are X's assigned residents?" — answered by the fast path from the people the app
knows (no model call), and person_N masking so the model never reads a staff name in other questions.

Fixtures: tests/data/residents.json, staffs.json, familymembers.json (fictional).
  "Maya"         resident Maya Angelou + staff Care Coordinator Maya Patel + Wellness Coordinator Maya Patel
  "Owen Hart"    one staff member (Doctor, inactive, Garden Terrace)
  "Priya Raman"  one staff member (Physician, active, Garden Terrace), assigned to RES-1001, RES-1002, RES-1004
  "Margaret"     residents Margaret Thatcher, Margaret Hamilton + family member Margaret Vance (Eleanor Vance's daughter)
"""

import json

import pytest
import yaml

from app.data.memory_repository import MemoryResidentRepository
from app.nlu.person_question import PersonQuestionParser
from app.nlu.vocabulary import EntityIndex
from tests.conftest import FakeTranslator

ROUTES = yaml.safe_load(open("app/data/intent_routes.yaml", encoding="utf-8"))["direct"]
PARSER = PersonQuestionParser(ROUTES["person_questions"], ROUTES["lookup_attributes"])


async def ask(make_service, q, plans=None):
    fake = FakeTranslator(plans or {})
    service = await make_service(fake)
    return await service.handle(q), fake


def text(r) -> str:
    return json.dumps(r["response"])


# ======================================================================== 1. the grammar


@pytest.mark.parametrize(
    ("question", "name", "attributes", "check", "kinds"),
    [
        ("Who is Jone?", "Jone", [], None, {"resident", "staff", "family"}),
        ("What is Jone's designation?", "Jone", ["staff_designation"], None, {"staff"}),
        ("Is Jone active?", "Jone", [], "active", {"resident", "staff"}),
        ("Which facility does Jone belong to?", "Jone", ["facility"], None, {"resident", "staff"}),
        ("What is Jone's role?", "Jone", ["staff_role"], None, {"staff"}),
        ("What are Jone's assigned residents?", "Jone", ["staff_residents"], None, {"staff"}),
        ("What is Eleanor Vance's room number?", "Eleanor Vance", ["room_number"], None, {"resident"}),
        ("who is jone", "jone", [], None, {"resident", "staff", "family"}),
    ],
)
def test_person_questions_are_recognised(question, name, attributes, check, kinds):
    asked = PARSER.parse(question)
    assert asked is not None and asked.name == name and [a for a, _ in asked.attributes] == attributes
    assert asked.check == check and asked.kinds == kinds


def test_kind_word_and_detail():
    asked = PARSER.parse("What is staff member Jone (Transport Driver)'s designation?")
    assert (asked.name, asked.kind, asked.detail) == ("Jone", "staff", "Transport Driver")


@pytest.mark.parametrize(
    "question",
    [
        "How many residents are active?",  # no name
        "Who is in room 101?",  # a number
        "Which resident is Margaret Vance related to?",  # asks for residents, not about the person
        "What are the designations of Owen Hart and Priya Raman?",  # two people
    ],
)
def test_other_questions_are_not_person_questions(question):
    assert PARSER.parse(question) is None


async def test_people_named_across_kinds():
    index = EntityIndex(await MemoryResidentRepository().vocabulary())
    maya = index.people_named("Maya")
    assert [r["name"] for r in maya["residents"]] == ["Maya Angelou"] and len(maya["staff"]) == 2
    assert [s["name"] for s in index.people_named("Care Coordinator Maya Patel")["staff"]] == ["Care Coordinator Maya Patel"]
    assert index.people_named("Zzyzx") == {"residents": [], "staff": [], "family": []}


# ======================================================================== 2. fast path answers (no model)


async def test_staff_designation_is_answered_without_the_model(make_service):
    r, fake = await ask(make_service, "What is Owen Hart's designation?")
    assert fake.calls == [] and r["tier"] == "direct" and r["modelCalls"] == 0 and r["tokensConsumed"] == 0
    assert r["response"]["attributeSpotlight"]["value"] == "Doctor"
    assert r["policy"]["policyCategory"] == "STAFF_INTENT" and r["intent"]["category"] == "STAFF_LOOKUP"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Who is Owen Hart?", ["Doctor", "Inactive", "Garden Terrace"]),
        ("What is Priya Raman's role?", ["Physician"]),
        ("Which facility does Priya Raman belong to?", ["Garden Terrace"]),
        ("What are Priya Raman's assigned residents?", ["Eleanor Vance", "Arthur Pendleton", "Harold Finch"]),
    ],
)
async def test_staff_attribute_questions(make_service, question, expected):
    r, fake = await ask(make_service, question)
    assert fake.calls == [] and not r["response"].get("needsClarification")
    assert all(e in text(r) for e in expected), text(r)
    assert "@" not in text(r)  # contact details only when asked for


async def test_is_x_active_is_a_yes_no_answer(make_service):
    r, fake = await ask(make_service, "Is Owen Hart active?")
    assert fake.calls == [] and r["response"]["headline"].startswith("No — Owen Hart")
    r2, _ = await ask(make_service, "Is Priya Raman active?")
    assert r2["response"]["headline"].startswith("Yes — Priya Raman")


async def test_same_named_staff_are_all_answered(make_service):
    """'Maya' as staff: two Maya Patels; the designation question can only mean staff, so both are answered."""
    r, fake = await ask(make_service, "What is Maya's designation?")
    assert fake.calls == [] and "Care Coordinator" in text(r) and "Activities Coordinator" in text(r)


async def test_several_kinds_are_listed_without_a_model_call(make_service):
    r, fake = await ask(make_service, "Who is Maya?")
    resp = r["response"]
    assert fake.calls == [] and r["tier"] == "direct" and r["tokensConsumed"] == 0
    assert resp["needsClarification"] and '"Maya" matches 1 resident and 2 staff members' in resp["summaryText"]
    labels = [o["label"] for o in resp["clarificationOptions"]]
    assert "Resident · Maya Angelou" in labels and "Staff · Care Coordinator Maya Patel (Care Coordinator)" in labels


async def test_each_option_answers_exactly_one_person(make_service):
    fake = FakeTranslator({})
    service = await make_service(fake)
    options = (await service.handle("Who is Maya?"))["response"]["clarificationOptions"]
    for option in options:
        r = await service.handle(option["query"])
        assert not r["response"].get("needsClarification"), option
    assert fake.calls == []


async def test_family_option_is_answered_deterministically(make_service):
    fake = FakeTranslator({})
    service = await make_service(fake)
    r = await service.handle("Who is Margaret?")
    assert "2 residents and 1 family member" in r["response"]["summaryText"]
    family = next(o for o in r["response"]["clarificationOptions"] if o["label"].startswith("Family"))
    assert family["label"] == "Family · Margaret Vance (Daughter of Eleanor Vance)"
    r2 = await service.handle(family["query"])
    assert fake.calls == [] and "Margaret Vance is Eleanor Vance's daughter." in text(r2)


async def test_same_name_same_kind_is_told_apart_by_designation(settings):
    """Staging has two different staff members recorded as just "Jone": a detail picks exactly one of them."""
    from app.engine.service import QueryService
    from tests.test_staff import STAFF, _repo_with

    twins = [{"_id": "j1", "cName": "c-j1", "name": "Jone", "designation": "Transport Driver", "active": True},
             {"_id": "j2", "cName": "c-j2", "name": "Jone", "designation": "Social Worker", "active": False}]  # fmt: skip
    fake = FakeTranslator({})
    service = QueryService(_repo_with([*STAFF, *twins]), fake, settings)
    await service.refresh_vocabulary()
    both = await service.handle("Who is Jone?")
    assert "Transport Driver" in text(both) and "Social Worker" in text(both)
    one = await service.handle("Who is staff member Jone (Social Worker)?")
    assert "Social Worker" in text(one) and "Transport Driver" not in text(one)
    assert fake.calls == []


def test_phone_country_code_is_a_staff_phone_question():
    asked = PARSER.parse("What is Jone's phone country code?")
    assert (asked.name, asked.attributes, asked.kinds) == ("Jone", [("staff_phone", "staff")], {"staff"})


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        ({"phone": "5550100", "countryCode": "+1"}, "+1 5550100"),
        ({"phone": " +44 20 7946 0000 ", "countryCode": "+1"}, "+44 20 7946 0000"),  # already international
        ({"phone": "5550100", "countryCode": ""}, "5550100"),
        ({"phone": "", "countryCode": "+1"}, ""),  # a code alone is not a phone
    ],
)
def test_staff_phone_is_shown_with_its_country_code(raw, shown):
    from app.data.staff import _py_phone

    assert _py_phone(raw) == shown


async def test_phone_country_code_is_answered_without_the_model(settings):
    """Staging trace q_b7f8073e31a2: "What is Jone's phone country code?" went to the model (3,417 tokens) and failed."""
    from app.engine.service import QueryService
    from tests.test_staff import STAFF, _repo_with

    twins = [{"_id": "j1", "cName": "c-j1", "name": "Jone", "designation": "Transport Driver", "active": True,
              "phone": "5550100", "countryCode": "+1"},
             {"_id": "j2", "cName": "c-j2", "name": "Jone", "designation": "Social Worker", "active": True,
              "phone": "5550199", "countryCode": "+91"}]  # fmt: skip
    fake = FakeTranslator({})
    service = QueryService(_repo_with([*STAFF, *twins]), fake, settings)
    await service.refresh_vocabulary()
    both = await service.handle("What is Jone's phone country code?")
    assert both["response"].get("needsClarification") and "Transport Driver" in text(both)
    assert "555" not in text(both)  # never several people's contacts
    one = await service.handle("What is staff member Jone (Transport Driver)'s phone country code?")
    assert "+1 5550100" in text(one) and "5550199" not in text(one)
    assert fake.calls == [] and one["tokensConsumed"] == 0


async def test_only_residents_keep_the_usual_resident_path(make_service):
    plan = {"action": "lookup", "lookup": {"residents": ["resident_1"]}}
    r, _ = await ask(make_service, "Who is Eleanor Vance?", {"Who is resident_1?": plan})
    assert "Resident: Eleanor Vance" in " ".join(r["interpretation"]) and r["policy"]["policyCategory"] == "RESIDENT_INTENT"


async def test_a_name_nobody_has_is_not_found_without_the_model(make_service):
    # changed deliberately (2026-11-26): the model only sees known names as placeholders, so it could never answer
    r, fake = await ask(make_service, "Who is Zzyzx?", {"Who is Zzyzx?": {"action": "clarify", "message": "?"}})
    assert fake.calls == [] and r["response"]["headline"] == "Not found"


# ======================================================================== 3. person_N for the model (complex questions)


async def test_staff_names_reach_the_model_only_as_person_n(make_service):
    q = "How many active residents does Nurse Sarah Jenkins look after?"
    plan = {"action": "count", "filters": {"status": {"include": ["active"]}, "primary_caregiver": {"include": ["person_1"]}}}
    r, fake = await ask(make_service, q, {"How many active residents does Nurse person_1 look after?": plan})
    assert r["response"]["count"] == 9
    sent = json.dumps(fake.calls)
    assert "Jenkins" not in sent and "Sarah" not in sent


async def test_model_staff_lookup_with_person_placeholders(make_service):
    q = "What are the designations of Owen Hart and Priya Raman?"
    plan = {"action": "lookup", "entity": "staff",
            "lookup": {"staff": ["person_1", "person_2"], "attributes": ["staff_designation"]}}  # fmt: skip
    r, fake = await ask(make_service, q, {"What are the designations of person_1 and person_2?": plan})
    assert "Doctor" in text(r) and "Physician" in text(r) and "Hart" not in json.dumps(fake.calls)
    assert r["intent"]["category"] == "STAFF_LOOKUP"


async def test_follow_up_reuses_the_previous_turns_person(make_service):
    first = {"action": "count", "filters": {"status": {"include": ["active"]}, "primary_caregiver": {"include": ["person_1"]}}}
    follow = {"action": "count", "uses_previous": True, "filters": {
        "status": {"include": ["active"]}, "primary_caregiver": {"include": ["person_1"]}, "care_type": {"include": ["memory_care"]}}}  # fmt: skip
    fake = FakeTranslator({"How many active residents does Nurse person_1 look after?": first,
                           "And how many of them are in memory care?": follow})  # fmt: skip
    service = await make_service(fake)
    await service.handle("How many active residents does Nurse Sarah Jenkins look after?", session_id="s")
    r = await service.handle("And how many of them are in memory care?", session_id="s")
    assert not r["response"].get("needsClarification") and r["response"]["count"] == 9
    assert "Jenkins" not in json.dumps(fake.calls)  # not even in the previous plan the model reads


async def test_declined_staff_question_gets_a_staff_message(make_service):
    q = "What are the designations of Owen Hart and Priya Raman?"
    plans = {"What are the designations of person_1 and person_2?": {"action": "unsupported", "message": "x"}}
    r, _ = await ask(make_service, q, plans)
    assert "staff records" in r["response"]["summaryText"] and "resident records" not in r["response"]["summaryText"]
    assert r["policy"]["policyCategory"] == "STAFF_INTENT"


async def test_model_resident_lookup_of_a_shared_name_is_one_call(make_service):
    """Staging trace "What room or unit is Sagar staying in?": the model rightly looked up person_1 as a resident; that
    must count as using the person (it once triggered a useless identical retry and a clarification)."""
    q = "Which room are Maya and Eleanor Vance in?"  # two people: not a fast-path person question
    plan = {"action": "lookup", "lookup": {"residents": ["person_1", "resident_1"], "attributes": ["room_number"]}}
    r, fake = await ask(make_service, q, {"Which room are person_1 and resident_1 in?": plan})
    assert len(fake.calls) == 1 and not r["response"].get("needsClarification")
    assert "Maya Angelou" in text(r) and "Eleanor Vance" in text(r)


# ======================================================================== 4. masking variants and false positives

COMMON_WORD_STAFF = [{"_id": "w", "cName": "cw", "name": "Will Grant", "designation": "Nurse", "active": True},
                     {"_id": "g", "cName": "cg", "name": "Grace Lee", "designation": "Nurse", "active": True},
                     {"_id": "j", "cName": "cj", "name": "Jone", "designation": "Transport Driver", "active": True}]  # fmt: skip


@pytest.fixture
async def common_word_index():
    from tests.test_staff import STAFF, _repo_with

    return EntityIndex(await _repo_with([*STAFF, *COMMON_WORD_STAFF]).vocabulary())


@pytest.mark.parametrize(
    ("question", "for_model"),
    [
        ("What is Jone's designation?", "What is person_1's designation?"),
        ("what is jone's designation", "what is person_1's designation"),
        ("how many residents does jone look after", "how many residents does person_1 look after"),
        ("is jone active", "is person_1 active"),
        ("who is JONE", "who is person_1"),
        ("residents of nurse jone", "residents of nurse person_1"),
        ("how many residents does owen hart look after", "how many residents does person_1 look after"),
        ("show residents and grace lee's patients", "show residents and person_1's patients"),
    ],
)
async def test_staff_names_are_masked_in_any_capitalisation(common_word_index, question, for_model):
    assert common_word_index.mask(question, family=True).for_model == for_model


@pytest.mark.parametrize(
    "question",
    [
        "will the residents in memory care be counted",  # "will": first word and the assistant's own wording
        "how many residents will be discharged",  # "will": a word the assistant itself uses
        "list residents with grace",  # lowercase, not in a name position
        "how many active residents are there",
    ],
)
async def test_common_words_are_not_masked(common_word_index, question):
    m = common_word_index.mask(question, family=True)
    assert m.for_model == question and not m.model_persons


# ======================================================================== 5. failure cache scope


async def test_a_remembered_failure_is_never_replayed_for_another_person(make_service):
    """The masked text of "Who is the caretaker of X?" is the same for every resident: a failure remembered for one
    resident must not answer the same question about another (the key binds what each placeholder stands for)."""
    drop = {"action": "count"}  # drops the resident: a grounding failure, remembered after the retry
    good = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["primary_caregiver"]}}
    fake = FakeTranslator({"Who is the caretaker of resident_1?": [drop, drop, good]})
    service = await make_service(fake)
    first = await service.handle("Who is the caretaker of Eleanor Vance?")
    assert first["response"].get("needsClarification") and len(fake.calls) == 2
    again = await service.handle("Who is the caretaker of Eleanor Vance?")
    assert again["modelCalls"] == 0  # the same question about the same person: the remembered failure
    other = await service.handle("Who is the caretaker of Harold Finch?")
    assert len(fake.calls) == 3 and not other["response"].get("needsClarification")  # asked afresh, answered


async def test_same_named_residents_are_asked_without_the_model(make_service):
    """Staging: "Dodia Sagar" (T-101) and "Sagar Dodia" (A-504) are both live. "What room is Sagar in?" must ask which
    one, deterministically — never a model call that ends in the same question."""
    # the picked option names ONE resident: the usual resident path answers it (the model here only because this test
    # service has no semantic router; in production the resident fast path does)
    room = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["room_number"]}}
    fake = FakeTranslator({"What room or unit is resident_1 staying in?": room})
    service = await make_service(fake)
    r = await service.handle("What room or unit is Margaret staying in?")  # Margaret Thatcher / Margaret Hamilton (+ family)
    assert fake.calls == [] and r["response"]["needsClarification"] and "2 residents" in r["response"]["summaryText"]
    options = r["response"]["clarificationOptions"]
    assert [o["label"] for o in options] == ["Resident · Margaret Thatcher", "Resident · Margaret Hamilton"]  # rooms: residents only
    assert options[0]["query"] == "What room or unit is Margaret Thatcher staying in?"
    picked = await service.handle(options[0]["query"])
    assert not picked["response"].get("needsClarification") and "Margaret Thatcher" in text(picked)


async def test_residents_with_the_same_words_in_their_names_are_told_apart(settings):
    """Staging: "Dodia Sagar" (T-101) and "Sagar Dodia" (A-504) are both live; the search ignores word order, so each
    option carries the record id and is answered exactly — never the same clarification again."""
    from app.engine.service import QueryService
    from tests.test_staff import STAFF, _repo_with

    repo = _repo_with(STAFF)
    repo._records = [*repo._records,
                     {**repo._records[0], "residentId": "RES-7001", "id": "r7001", "firstName": "Dodia", "lastName": "Sagar", "roomNumber": "T-101"},
                     {**repo._records[0], "residentId": "RES-7002", "id": "r7002", "firstName": "Sagar", "lastName": "Dodia", "roomNumber": "A-504"}]  # fmt: skip
    fake = FakeTranslator({})
    service = QueryService(repo, fake, settings)
    await service.refresh_vocabulary()
    r = await service.handle("What room or unit is Sagar staying in?")
    options = r["response"]["clarificationOptions"]
    assert [o["label"] for o in options] == ["Resident · Dodia Sagar (ID RES-7001)", "Resident · Sagar Dodia (ID RES-7002)"]
    rooms = [(await service.handle(o["query"]))["response"]["attributeSpotlight"]["value"] for o in options]
    assert rooms == ["T-101", "A-504"] and fake.calls == []


# ======================================================================== 7. staging question set (2026-11-24)
# Each of these went to the model (3.3k–7.3k tokens) and most failed. Now: the fast path, 0 model calls.

ACCESS = [
    {"name": "Residents", "allowed": True, "isRead": True, "isWrite": True, "children": []},
    {"name": "Transport", "allowed": True, "isRead": True, "isWrite": True, "children": [
        {"name": "transport-calendar", "allowed": False, "isRead": False, "isWrite": False},
        {"name": "transport-requests", "allowed": True, "isRead": True, "isWrite": True}]},
    {"name": "Staff", "allowed": False, "isRead": False, "isWrite": False, "children": []},
]  # fmt: skip
JONES = [{"_id": "j1", "cName": "c-j1", "name": "Jone", "designation": "Transport Driver", "active": True,
          "phone": "5550100", "countryCode": "+1", "email": "jone.driver@example.com", "accessPermissions": ACCESS},
         {"_id": "j2", "cName": "c-j2", "name": "Jone", "designation": "Social Worker", "active": False,
          "accessPermissions": []},
         {"_id": "j3", "cName": "c-j3", "name": "Hogan Jone", "designation": "Case Manager", "active": True}]  # fmt: skip


def _staging_repo():
    from tests.test_staff import STAFF, _repo_with

    repo = _repo_with([*STAFF, *JONES])
    base = repo._records[0]
    repo._records = [*repo._records,
                     {**base, "residentId": "RES-7001", "id": "r7001", "firstName": "Dodia", "lastName": "Sagar", "roomNumber": "T-101"},
                     {**base, "residentId": "RES-7002", "id": "r7002", "firstName": "Sagar", "lastName": "Dodia", "roomNumber": "A-504"}]  # fmt: skip
    repo._records[1]["primaryCaregiver"] = "Jone"  # the caregiver recorded as just "Jone"
    repo._records[2]["primaryCaregiver"] = "Hogan Jone"  # and one whose SURNAME is Jone
    repo._records[3]["primaryCaregiver"] = "RM staff"  # staging: a caregiver whose "surname" is an ordinary word
    repo._attach_staff()
    return repo


async def _staging_service(settings):
    from app.engine.service import QueryService

    fake = FakeTranslator({})  # any model call fails the test
    service = QueryService(_staging_repo(), fake, settings)
    await service.refresh_vocabulary()
    return service, fake


@pytest.mark.parametrize(
    ("question", "attributes", "sections"),
    [
        ("Which residents are assigned to Jone?", ["staff_residents"], []),
        ("Can you show me all the staff members assigned to Sagar?", ["care_team"], []),
        ("How many staff members are assigned to Sagar?", ["care_team"], []),
        ("What are Jone's contact details and job designation?", ["staff_contact", "staff_designation"], []),
        ("Which facility is Sagar associated with, and what type of care does he receive?", ["facility", "care_type"], []),
        ("Is Jone allowed to access the Residents section?", [], ["Residents"]),
        ("Does Jone have permission to access the Transport section?", [], ["Transport"]),
        ("Can Jone access the transport calendar?", [], ["transport-calendar"]),
        ("Does Jone have access to staff management?", [], ["Staff"]),
    ],
)
def test_staging_questions_are_simple_person_questions(question, attributes, sections):
    asked = PARSER.parse(question)
    assert asked is not None and [a for a, _ in asked.attributes] == attributes and asked.sections == sections


def test_portal_access_is_not_an_app_section():
    assert PARSER.parse("Does Jone have portal access?") is None  # a family flag word, not an app section


@pytest.mark.parametrize(
    ("question", "verdict"),
    [
        ("Is staff member Jone (Transport Driver) allowed to access the Residents section?", "Yes"),
        ("Does staff member Jone (Transport Driver) have permission to access the Transport section?", "Yes"),
        ("Can staff member Jone (Transport Driver) access the transport calendar?", "No"),  # the child is switched off
        ("Does staff member Jone (Transport Driver) have access to staff management?", "No"),
        ("Can staff member Jone (Transport Driver) access dining?", "not recorded"),  # not in the permission list
    ],
)
async def test_app_permissions_yes_or_no_for_one_staff_member(settings, question, verdict):
    service, fake = await _staging_service(settings)
    r = await service.handle(question)
    assert fake.calls == [] and r["tokensConsumed"] == 0
    assert r["response"]["attributeSpotlight"]["value"] == verdict
    assert "isWrite" not in text(r) and "Write" not in text(r)  # never write access, never the whole set


async def test_app_permissions_of_a_shared_name_ask_which_person(settings):
    service, fake = await _staging_service(settings)
    r = await service.handle("Can Jone access the transport calendar?")
    assert fake.calls == [] and r["response"]["needsClarification"]
    queries = [o["query"] for o in r["response"]["clarificationOptions"]]
    assert any("Transport Driver" in q for q in queries) and any("Social Worker" in q for q in queries)
    picked = await service.handle(next(q for q in queries if "Transport Driver" in q))
    assert picked["response"]["attributeSpotlight"]["value"] == "No" and fake.calls == []


async def test_contact_details_and_designation(settings):
    service, fake = await _staging_service(settings)
    both = await service.handle("What are Jone's contact details and job designation?")
    assert both["response"]["needsClarification"] and "555" not in text(both)  # two people named exactly Jone
    one = await service.handle("What are staff member Jone (Transport Driver)'s contact details and job designation?")
    assert "+1 5550100" in text(one) and "jone.driver@example.com" in text(one) and "Transport Driver" in text(one)
    assert fake.calls == []


@pytest.mark.parametrize(
    "question",
    [
        "Can you show me all the staff members assigned to Sagar?",
        "How many staff members are assigned to Sagar?",
        "Which facility is Sagar associated with, and what type of care does he receive?",
    ],
)
async def test_a_shared_resident_name_is_asked_before_any_model_call(settings, question):
    service, fake = await _staging_service(settings)
    r = await service.handle(question)
    assert fake.calls == [] and r["tokensConsumed"] == 0 and r["response"]["needsClarification"]
    queries = [o["query"] for o in r["response"]["clarificationOptions"]]
    assert len(queries) == 2 and all("Sagar" in q for q in queries)  # the question itself is kept
    picked = await service.handle(queries[0])
    assert not picked["response"].get("needsClarification") and fake.calls == []


async def test_a_full_name_written_in_order_is_that_resident():
    index = EntityIndex(await _staging_repo().vocabulary())
    assert index.resolve_resident("Sagar Dodia", {}).values == ["RES-7002"]
    assert index.resolve_resident("Dodia Sagar", {}).values == ["RES-7001"]
    assert [p["id"] for p in index.people_named("Sagar Dodia")["residents"]] == ["RES-7002"]
    assert len(index.people_named("Sagar")["residents"]) == 2  # one word: both


async def test_a_name_recorded_exactly_wins_over_a_surname():
    """Staging: "Which residents are assigned to Jone?" answered Hogan Jone's residents, not those of the staff member
    recorded exactly as "Jone"."""
    index = EntityIndex(await _staging_repo().vocabulary())
    assert index.mentioned_caregivers("Which residents are assigned to Jone?") == {"Jone": ["Jone"]}
    assert index.resolve_caregiver("Jone").values == ["Jone"]


async def test_an_ordinary_word_is_not_a_caregivers_surname():
    """Staging: "How many staff members are assigned to Sagar Dodia?" was checked against caregiver "RM staff"."""
    index = EntityIndex(await _staging_repo().vocabulary())
    assert index.mentioned_caregivers("How many staff members are assigned to Sagar Dodia?") == {}
    assert index.mentioned_caregivers("Which residents does Staff look after?") != {}  # written as a name: still one
