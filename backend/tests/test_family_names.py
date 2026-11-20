"""Family-member NAMES as a third kind of person (family_N / person_N): masking, resolution, grounding, execution,
authorization and privacy. Builds its own fictional fixture with deliberate collisions:

  RES-1001 Eleanor Vance (FAC-A)   Margaret Vance (Daughter) · Arthur Pendleton (Brother; also a RESIDENT) · Sam Lee (Son)
  RES-1002 Arthur Pendleton (FAC-A) Rose (Sister; one-word name) · Linda Pendleton (Daughter in law)
  RES-1003 Martha Stewart-Wells (FAC-A)  Marcus Chen (Son; also STAFF "Care Aide Marcus Chen")
  RES-1004 Harold Finch (FAC-B)     Sam Lee (Nephew; same name as Eleanor's son) · Kim Park (Daughter)
"""

import copy
import json
import logging

import pytest

from app.config import Settings
from app.data.memory_repository import SEED_FILE, MemoryResidentRepository
from app.data.repository import Vocabulary
from app.data.resident_view import to_mongo_filter
from app.engine.caller import CallerContext
from app.engine.service import QueryService
from app.models.plan import Filters
from app.nlu.vocabulary import EntityIndex
from tests.conftest import FakeTranslator, PinnedDateSettings

FAMILY = [
    {"_id": "fm01", "residentId": "RES-1001", "name": "Margaret Vance", "relation": "Daughter", "type": "Family",
     "phone": "555-0101", "email": "margaret@example.com"},
    {"_id": "fm02", "residentId": "RES-1003", "name": "Marcus Chen", "relation": "Son", "type": "Family", "phone": "555-0301"},
    {"_id": "fm03", "residentId": "RES-1001", "name": "Arthur Pendleton", "relation": "Brother", "type": "Family"},
    {"_id": "fm04", "residentId": "RES-1001", "name": "Sam Lee", "relation": "Son", "type": "Family", "phone": "555-0104"},
    {"_id": "fm05", "residentId": "RES-1004", "name": "Sam Lee", "relation": "Nephew", "type": "Family", "phone": "555-0405"},
    {"_id": "fm06", "residentId": "RES-1002", "name": "Rose", "relation": "Sister", "type": "Family"},
    {"_id": "fm07", "residentId": "RES-1002", "name": "Linda Pendleton", "relation": "Daughter in law", "type": "Family"},
    {"_id": "fm08", "residentId": "RES-1004", "name": "Kim Park", "relation": "Daughter", "type": "Family",
     "phone": "555-0408", "email": "kim@example.com"},
]  # fmt: skip
PII = ["Margaret", "Marcus Chen", "Sam Lee", "Linda", "Kim Park", "555-0", "@example.com"]


def residents():
    rows = json.loads(SEED_FILE.read_text(encoding="utf-8"))
    for r in rows:
        r["facilityId"] = "FAC-B" if r["residentId"] == "RES-1004" else "FAC-A"
    return rows


def settings(**kw) -> Settings:
    return PinnedDateSettings(_env_file=None, environment="test", topic_gate_enabled=False, semantic_router_enabled=False,
                              family_name_resolution_enabled=kw.pop("family", True), **kw)  # fmt: skip


async def service(plans, family=FAMILY, **kw):
    fake = FakeTranslator(plans)
    svc = QueryService(MemoryResidentRepository(residents(), copy.deepcopy(family)), fake, settings(**kw))
    await svc.refresh_vocabulary()
    return svc, fake


def relationship(placeholder="person_1"):
    return {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["family_relationship"]},
            "filters": {"family_member": {"include": [placeholder]}}}  # fmt: skip


# ======================================================================== 1. masking


def index(**_):
    repo = MemoryResidentRepository(residents(), copy.deepcopy(FAMILY))
    import asyncio

    return EntityIndex(asyncio.run(repo.vocabulary()))


IDX = index()


@pytest.mark.parametrize(
    ("question", "masked"),
    [
        # family only / staff only / resident only
        ("Which resident is Margaret Vance related to?", "Which resident is family_1 related to?"),
        ("Who does Care Aide Marcus Chen look after?", "Who does Care Aide Marcus Chen look after?"),  # titled: staff
        ("What is Eleanor Vance's room?", "What is resident_1's room?"),
        # staff / family collision
        ("What is the relationship between Marcus Chen and Martha Stewart-Wells?",
         "What is the relationship between person_1 and resident_1?"),
        # resident / family collision: Arthur is a resident AND Eleanor's brother
        ("How is Arthur Pendleton related to Eleanor Vance?", "How is person_1 related to resident_2?"),
        # one-word name: only next to its resident
        ("Is Rose a family member of Arthur Pendleton?", "Is family_1 a family member of resident_1?"),
        ("Show the Rose Garden residents", "Show the Rose Garden residents"),
    ],
)  # fmt: skip
def test_masking(question, masked):
    assert IDX.mask(question, family=True).text == masked


def test_masking_keeps_every_candidate_server_side():
    m = IDX.mask("What is the relationship between Marcus Chen and Martha Stewart-Wells?", family=True)
    assert m.persons["person_1"]["staff"] == ["Care Aide Marcus Chen"]
    assert [c["id"] for c in m.persons["person_1"]["family"]] == ["fm02"]
    assert m.entity_originals["person_1"] == "Marcus Chen"
    shared = IDX.mask("Which resident is Sam Lee related to?", family=True)
    assert sorted(c["residentId"] for c in shared.family["family_1"]) == ["RES-1001", "RES-1004"]


@pytest.mark.parametrize("question", ["What is the relationship between Marcus Chen and Martha Stewart-Wells?",
                                      "How is Arthur Pendleton related to Eleanor Vance?", "Is Rose related to Arthur Pendleton?"])
def test_flag_off_masks_exactly_as_before(question):
    off = IDX.mask(question)
    assert not off.family and not off.persons and "family_" not in off.text and "person_" not in off.text


# ======================================================================== 2. the target question and its variants


async def test_relationship_with_a_staff_name_collision():
    q = "What is the relationship between Marcus Chen and Martha Stewart-Wells?"
    svc, fake = await service({"What is the relationship between person_1 and resident_1?": relationship()})
    r = await svc.handle(q)
    assert r["response"]["attributeSpotlight"]["value"] == "Marcus Chen is Martha Stewart-Wells's son."
    assert len(fake.calls) == 1  # no caregiver condition demanded, no retry


async def test_show_a_residents_daughters_only():
    q = "Show Harold Finch's daughters"
    plan = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["family_members"]},
            "filters": {"family_relation": {"include": ["daughter"]}}}  # fmt: skip
    svc, _ = await service({"Show resident_1's daughters": plan})
    text = (await svc.handle(q))["response"]["attributeSpotlight"]["value"]
    assert "Kim Park (Daughter)" in text and "Sam Lee" not in text


async def test_reverse_question_family_only_name():
    q = "Which resident is Margaret Vance related to?"
    plan = {"action": "list", "filters": {"family_member": {"include": ["family_1"]}}}
    svc, _ = await service({"Which resident is family_1 related to?": plan})
    r = await svc.handle(q)
    assert r["response"]["count"] == 1 and r["response"]["residentsSample"][0]["residentId"] == "RES-1001"


async def test_yes_no_membership():
    q = "Is Rose a family member of Arthur Pendleton?"
    plan = {"action": "lookup", "lookup": {"residents": ["resident_1"]}, "filters": {"family_member": {"include": ["family_1"]}}}
    svc, _ = await service({"Is family_1 a family member of resident_1?": plan})
    assert (await svc.handle(q))["response"]["headline"].startswith("Yes")


async def test_name_of_another_residents_family_member_is_not_found():
    q = "What is the relationship between Linda Pendleton and Eleanor Vance?"  # Linda belongs to Arthur, not Eleanor
    svc, _ = await service({"What is the relationship between family_1 and resident_1?": relationship("family_1")})
    text = (await svc.handle(q))["response"]["summaryText"]
    assert text == 'No family member named "Linda Pendleton" is recorded for Eleanor Vance.'
    assert "Arthur" not in text  # never reveals where the name does exist


async def test_shared_name_across_residents_asks_which_resident():
    q = "Which resident is Sam Lee related to?"
    svc, _ = await service({"Which resident is family_1 related to?": {"action": "list",
                                                                        "filters": {"family_member": {"include": ["family_1"]}}}})
    r = await svc.handle(q)
    assert r["response"]["needsClarification"]
    assert sorted(o["label"] for o in r["response"]["clarificationOptions"]) == ["Eleanor Vance", "Harold Finch"]


async def test_shared_name_resolved_within_the_named_resident():
    q = "What is the relationship between Sam Lee and Harold Finch?"
    svc, _ = await service({"What is the relationship between family_1 and resident_1?": relationship("family_1")})
    assert (await svc.handle(q))["response"]["attributeSpotlight"]["value"] == "Sam Lee is Harold Finch's nephew."


async def test_two_contacts_with_one_name_for_one_resident_asks():
    family = [*FAMILY, {"_id": "fm09", "residentId": "RES-1001", "name": "Margaret Vance", "relation": "Niece", "type": "Family"}]
    q = "What is the relationship between Margaret Vance and Eleanor Vance?"
    svc, _ = await service({"What is the relationship between family_1 and resident_1?": relationship("family_1")}, family)
    r = await svc.handle(q)
    assert r["response"]["needsClarification"] and "more than one family contact" in r["response"]["summaryText"]


# ======================================================================== 3. caregivers and roles


async def test_genuine_caregiver_question_resolves_to_staff():
    q = "How many residents does Marcus Chen care for?"
    plan = {"action": "count", "filters": {"primary_caregiver": {"include": ["person_1"]}}}
    svc, _ = await service({"How many residents does person_1 care for?": plan})
    r = await svc.handle(q)
    direct = await svc.repo.count(Filters.model_validate({"primary_caregiver": {"include": ["Care Aide Marcus Chen"]}}))
    assert r["response"]["count"] == direct > 0


async def test_family_member_is_never_turned_into_a_caregiver():
    q = "Which resident is Margaret Vance related to?"
    wrong = {"action": "list", "filters": {"primary_caregiver": {"include": ["family_1"]}}}
    right = {"action": "list", "filters": {"family_member": {"include": ["family_1"]}}}
    svc, fake = await service({"Which resident is family_1 related to?": [wrong, right]})
    r = await svc.handle(q)
    assert len(fake.calls) == 2 and "not a caregiver" in fake.calls[1]["hint"]
    assert r["response"]["residentsSample"][0]["residentId"] == "RES-1001"


async def test_unused_person_is_never_dropped():
    q = "What is the relationship between Marcus Chen and Martha Stewart-Wells?"
    dropped = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["family_members"]}}
    svc, fake = await service({"What is the relationship between person_1 and resident_1?": [dropped, dropped]})
    r = await svc.handle(q)
    assert len(fake.calls) == 2 and "person_1" in fake.calls[1]["hint"]
    assert r["response"]["needsClarification"] and '"Marcus Chen"' in r["response"]["summaryText"]


# ======================================================================== 4. authorization and privacy


async def test_facility_isolation():
    caller = CallerContext(facilities=frozenset({"FAC-A"}))
    # Kim Park belongs to Harold Finch (FAC-B): invisible to a FAC-A caller, reported exactly like an unknown name
    svc, _ = await service({"Which resident is family_1 related to?": {"action": "list",
                                                                        "filters": {"family_member": {"include": ["family_1"]}}}})
    r = await svc.handle("Which resident is Kim Park related to?", caller=caller)
    assert r["response"]["summaryText"] == 'No family member named "Kim Park" is recorded.'
    # a shared name: only the authorised resident is a candidate, so there is nothing to ask
    r = await svc.handle("Which resident is Sam Lee related to?", caller=caller)
    assert not r["response"].get("needsClarification")
    assert [x["residentId"] for x in r["response"]["residentsSample"]] == ["RES-1001"]


async def test_family_permission_is_required():
    q = "What is the relationship between Marcus Chen and Martha Stewart-Wells?"
    svc, _ = await service({"What is the relationship between person_1 and resident_1?": relationship()})
    r = await svc.handle(q, caller=CallerContext(can_read_family=False, can_read_family_contacts=False))
    assert r["response"]["summaryText"] == "You don't have access to family information."


async def test_contact_details_need_their_own_permission():
    q = "Who are Harold Finch's family members?"
    plan = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["family_members"]}}
    svc, _ = await service({"Who are resident_1's family members?": plan})
    names_only = json.dumps((await svc.handle(q, caller=CallerContext(can_read_family_contacts=False)))["response"])
    assert "Kim Park (Daughter)" in names_only and "555-0408" not in names_only and "kim@example.com" not in names_only
    full = json.dumps((await svc.handle(q))["response"])
    assert "555-0408" in full


async def test_relationship_question_never_reads_contact_details():
    q = "What is the relationship between Marcus Chen and Martha Stewart-Wells?"
    svc, _ = await service({"What is the relationship between person_1 and resident_1?": relationship()})
    seen = []
    original = svc.repo.find_by_ids

    async def spy(ids, contacts=True):
        seen.append(contacts)
        return await original(ids, contacts=contacts)

    svc.repo.find_by_ids = spy
    r = await svc.handle(q)
    assert seen == [False] and "555-0301" not in json.dumps(r)


async def test_no_family_pii_reaches_the_model_or_the_logs(caplog):
    caplog.set_level(logging.INFO)
    q = "What is the relationship between Marcus Chen and Martha Stewart-Wells?"
    svc, fake = await service({"What is the relationship between person_1 and resident_1?": relationship()})
    await svc.handle(q)
    sent = json.dumps(fake.calls)
    assert not any(p in sent for p in PII)
    assert "Marcus Chen" not in caplog.text and "555-0" not in caplog.text


async def test_no_record_ids_in_responses():
    for q, key, plan in [
        ("What is the relationship between Marcus Chen and Martha Stewart-Wells?",
         "What is the relationship between person_1 and resident_1?", relationship()),
        ("Is Rose a family member of Arthur Pendleton?", "Is family_1 a family member of resident_1?",
         {"action": "lookup", "lookup": {"residents": ["resident_1"]}, "filters": {"family_member": {"include": ["family_1"]}}}),
    ]:  # fmt: skip
        svc, _ = await service({key: plan})
        text = json.dumps((await svc.handle(q))["response"])
        assert "fm0" not in text and "familyIds" not in text and "familyContacts" not in text


def test_enabled_by_default_in_every_environment():
    for env in ("development", "test", "production"):
        assert Settings(_env_file=None, environment=env).family_name_resolution_enabled is True  # nothing to configure


# ======================================================================== 5. execution parity and compiler


def test_family_member_filter_is_the_same_in_memory_and_mongodb():
    f = Filters.model_validate({"family_member": {"include": ["fm02"]}})
    assert to_mongo_filter(f) == {"familyIds": {"$in": ["fm02"]}}
    repo = MemoryResidentRepository(residents(), copy.deepcopy(FAMILY))
    import asyncio

    rows = asyncio.run(repo.find(f))
    assert [r["residentId"] for r in rows] == ["RES-1003"]


async def test_names_and_foreign_placeholders_are_rejected_by_the_compiler():
    q = "What is the relationship between Marcus Chen and Martha Stewart-Wells?"
    raw_name = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["family_relationship"]},
                "filters": {"family_member": {"include": ["Marcus Chen"]}}}  # fmt: skip
    svc, fake = await service({"What is the relationship between person_1 and resident_1?": [raw_name, relationship()]})
    r = await svc.handle(q)
    assert len(fake.calls) == 2 and "placeholder" in json.dumps(r["modelPlans"][0].get("compileError"))
    assert r["response"]["attributeSpotlight"]["value"] == "Marcus Chen is Martha Stewart-Wells's son."


async def test_flag_off_keeps_todays_behaviour():
    q = "What is the relationship between Marcus Chen and Martha Stewart-Wells?"
    svc, fake = await service({"What is the relationship between Marcus Chen and resident_1?": relationship("Marcus Chen")},
                              family=FAMILY, **{})  # fmt: skip
    svc.settings.family_name_resolution_enabled = False
    await svc.handle(q)
    assert "Marcus Chen" in fake.calls[0]["question"]  # unmasked, exactly as before the feature (flag off)


def test_vocabulary_rows_carry_no_contact_details():
    repo = MemoryResidentRepository(residents(), copy.deepcopy(FAMILY))
    import asyncio

    vocab: Vocabulary = asyncio.run(repo.vocabulary())
    assert vocab.family_members and all(set(m) == {"id", "name", "rel", "relation", "residentId", "facilityId"}
                                        for m in vocab.family_members)  # fmt: skip


def test_family_wording_is_never_a_name_even_if_a_record_is_called_that():
    """Staging has a family record named like "Family Member": the question's words must not be masked as that person."""
    import asyncio

    family = [*FAMILY, {"_id": "fm10", "residentId": "RES-1001", "name": "Family Members", "relation": "Other"},
              {"_id": "fm11", "residentId": "RES-1001", "name": "Next Kin", "relation": "Other"}]  # fmt: skip
    idx = EntityIndex(asyncio.run(MemoryResidentRepository(residents(), family).vocabulary()))
    assert idx.mask("Who are Eleanor Vance's family members?", family=True).text == "Who are resident_1's family members?"
    assert idx.mask("Is Margaret Vance a family member of Eleanor Vance?", family=True).text == (
        "Is family_1 a family member of resident_1?"
    )
    assert idx.mask("Who is Eleanor Vance's next of kin?", family=True).text == "Who is resident_1's next of kin?"
