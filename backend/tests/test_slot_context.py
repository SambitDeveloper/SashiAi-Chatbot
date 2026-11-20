"""Template slots: a synonym that only modifies an attribute ("current status") is not a value; real values still are.

Groups: 1 attribute context · 2 genuine synonym values · 3 reuse across wordings with zero model calls ·
4 no wrong reuse across attributes / status values · (5 = the full suite). Uses the semantic-cache test harness
(FakeStore, an embedder that scores every pair 1.0, so reuse is decided by slots, logic words and validation only).
"""

import pytest

from app.engine.semantic_cache import TEMPLATE_DOC_VERSION, mentions, slot_text
from tests.test_semantic_cache import _view, ask, build

STATUS_LOOKUP = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["status"]}}
CARE_LOOKUP = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["care_type"]}}


def kinds(service, question):
    return [(m.kind, m.value) for m in mentions(_view(service, question))]


# ======================================================================== 1. attribute context: not a value


@pytest.mark.parametrize(
    "question",
    [
        "What is Eleanor Vance's current status?",
        "What is Eleanor Vance's current room?",
        "Who is Eleanor Vance's current caregiver?",
        "Who was Eleanor Vance's former caregiver?",
        "What is Eleanor Vance's discharge date?",
        "What is Eleanor Vance's current care type?",
    ],
)
async def test_synonym_modifying_an_attribute_is_not_a_slot(repo, settings, question):
    service, _, _ = await build(repo, settings, {})
    view = _view(service, question)
    assert [m for m in mentions(view) if m.kind == "status"] == []
    assert "{status}" not in slot_text(view, mentions(view))  # the stored text keeps the words


async def test_stored_text_keeps_the_modifier(repo, settings):
    service, _, _ = await build(repo, settings, {})
    view = _view(service, "What is Eleanor Vance's current status?")
    assert slot_text(view, mentions(view)) == "What is resident_1's current status?"


# ======================================================================== 2. genuine values still recognised


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How many current residents are in memory care?", [("status", "active"), ("care_type", "memory_care")]),
        ("How many discharged residents are there?", [("status", "discharged")]),
        ("How many residents have left?", [("status", "discharged")]),  # synonym of a status in the data
        ("How many discharges were there?", [("status", "discharged")]),  # a lone synonym used as a value
        ("What is the discharged status count?", [("status", "discharged")]),  # canonical word: always a value
    ],
)
async def test_values_are_still_slots(repo, settings, question, expected):
    service, _, _ = await build(repo, settings, {})
    assert [k for k in kinds(service, question) if k[0] in ("status", "care_type")] == expected


# ======================================================================== 3. one template, three wordings, 0 model calls


async def test_three_status_wordings_share_one_template(repo, settings):
    service, fake, store = await build(repo, settings, {"What is resident_1's current status?": STATUS_LOOKUP})
    first = await ask(service, "What is Eleanor Vance's current status?")
    assert first["tier"] == "llm" and len(store.docs) == 1
    doc = next(iter(store.docs.values()))
    assert doc["doc_version"] == TEMPLATE_DOC_VERSION == 2 and doc["mention_counts"] == {}
    for q in ("What is Arthur Pendleton's status?", "Tell me Arthur Pendleton's status right now.",
              "What is Arthur Pendleton's current status?"):  # fmt: skip
        r = await ask(service, q)  # the fake model has no plan for these: a model call would fail the test
        assert r["tier"] == "template" and r["modelCalls"] == 0, q
        assert r["intent"]["parameters"]["lookup"]["residents"] == ["RES-1002"], q  # the right resident, refilled
        assert "Arthur Pendleton" in r["response"]["supportingDetail"] if "supportingDetail" in r["response"] else True
    assert len(fake.calls) == 1


async def test_cache_hits_still_run_validation(repo, settings):
    """A hit is re-resolved and re-grounded: a refill that does not fit the new question is rejected, not served."""
    service, fake, store = await build(repo, settings, {
        "What is resident_1's current status?": STATUS_LOOKUP,
        "What is resident_1's status and room?": {"action": "lookup", "lookup": {"residents": ["resident_1"],
                                                                                 "attributes": ["status", "room_number"]}},
    })  # fmt: skip
    await ask(service, "What is Eleanor Vance's current status?")
    r = await ask(service, "What is Arthur Pendleton's status and room?")  # asks for MORE than the template shows
    assert r["tier"] == "llm" and len(fake.calls) == 2


async def test_old_template_version_is_ignored(repo, settings):
    service, fake, store = await build(repo, settings, {"What is resident_1's current status?": STATUS_LOOKUP,
                                                       "What is resident_1's status?": STATUS_LOOKUP})  # fmt: skip
    await ask(service, "What is Eleanor Vance's current status?")
    for doc in store.docs.values():
        doc["doc_version"] = 1  # a template written before this change
    r = await ask(service, "What is Arthur Pendleton's status?")
    assert r["tier"] == "llm" and len(fake.calls) == 2


# ======================================================================== 4. no wrong reuse


async def test_status_template_never_answers_a_care_type_question(repo, settings):
    service, fake, _ = await build(repo, settings, {"What is resident_1's current status?": STATUS_LOOKUP,
                                                   "What is resident_1's care type?": CARE_LOOKUP})  # fmt: skip
    await ask(service, "What is Eleanor Vance's current status?")
    r = await ask(service, "What is Arthur Pendleton's care type?")
    assert r["tier"] == "llm" and len(fake.calls) == 2
    assert r["intent"]["parameters"]["lookup"]["attributes"] == ["care_type"]


async def test_status_value_is_refilled_never_kept(repo, settings):
    plan = {"action": "count", "filters": {"status": {"include": ["active"]}, "care_type": {"include": ["memory_care"]}}}
    service, fake, store = await build(repo, settings, {"How many current residents are in memory care?": plan})
    first = await ask(service, "How many current residents are in memory care?")
    assert first["tier"] == "llm" and len(store.docs) == 1
    r = await ask(service, "How many discharged residents are in memory care?")
    assert r["tier"] == "template" and r["modelCalls"] == 0
    assert r["intent"]["parameters"]["filters"]["status"] == {"include": ["discharged"]}  # refilled, not "active"
    direct = await service.repo.count(
        __import__("app.models.plan", fromlist=["Filters"]).Filters.model_validate(
            {"status": {"include": ["discharged"]}, "care_type": {"include": ["memory_care"]}}
        )
    )
    assert r["response"]["count"] == direct


async def test_question_without_the_status_value_does_not_reuse_a_status_filter(repo, settings):
    plan = {"action": "count", "filters": {"status": {"include": ["active"]}, "care_type": {"include": ["memory_care"]}}}
    service, fake, _ = await build(repo, settings, {
        "How many current residents are in memory care?": plan,
        "How many residents are in memory care?": {"action": "count", "filters": {"care_type": {"include": ["memory_care"]}}},
    })  # fmt: skip
    await ask(service, "How many current residents are in memory care?")
    r = await ask(service, "How many residents are in memory care?")  # no status word: must not inherit "active"
    assert r["tier"] == "llm" and "status" not in (r["intent"]["parameters"].get("filters") or {})
