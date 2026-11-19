"""Semantic plan-template cache (app/engine/semantic_cache.py).

Most tests use an embedder that maps EVERY question to the same vector, so every stored template is a perfect
similarity match (1.0): the worst case. They show that slot shapes, logic words and the existing validator — not the
similarity score — decide reuse. A separate real-BGE test measures false hits and missed paraphrases.

The store is an in-memory stand-in for the Atlas collection (tests only; production uses $vectorSearch).
"""

import asyncio
import copy
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from app.engine.grounding import scan_question
from app.engine.semantic_cache import (
    TEMPLATE_DOC_VERSION,
    Ineligible,
    QuestionView,
    SemanticPlanCache,
    logic_signature,
    mentions,
    slot_text,
    templatize,
)
from app.engine.service import QueryService
from tests.conftest import TODAY, FakeTranslator

MODELS_DIR = Path(__file__).resolve().parents[1] / ".models"

ROOM = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["room_number"]}}
CAREGIVER = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["primary_caregiver"]}}
CARE_LEVEL = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["care_type"]}}
ADMITTED = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["admission_date"]}}


class FakeStore:
    """In-memory stand-in for the Atlas collection: the same status/compat filter as the $vectorSearch stage."""

    def __init__(self, ignore_filter: bool = False):
        self.docs: dict[str, dict] = {}
        self.ignore_filter = ignore_filter  # a store that returns everything: the cache must still check status/compat
        self.fail: Exception | None = None
        self.delay = 0.0
        self.searches = 0

    async def search(self, vector, compat_key, limit, num_candidates):
        self.searches += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise self.fail
        q = np.asarray(vector)
        out = []
        for d in self.docs.values():
            if not self.ignore_filter and (d.get("revoked") is not False or d["compat_key"] != compat_key):
                continue
            e = np.asarray(d["embedding"])
            sim = float(q @ e / (np.linalg.norm(q) * np.linalg.norm(e)))
            out.append(({k: v for k, v in d.items() if k != "embedding"}, sim))
        return sorted(out, key=lambda x: -x[1])[:limit]

    async def store(self, doc):
        if doc["_id"] not in self.docs:  # like the Atlas upsert: revoked and expires_at are set only on insert
            now = datetime.now(UTC)
            self.docs[doc["_id"]] = {**copy.deepcopy(doc), "revoked": False, "seen_count": 0, "usage_count": 0,
                                     "created_at": now, "expires_at": now + timedelta(hours=12)}  # fmt: skip
        self.docs[doc["_id"]]["seen_count"] += 1
        return doc["_id"]

    async def mark_used(self, template_id):
        if template_id in self.docs:
            self.docs[template_id]["usage_count"] += 1

    async def close(self):
        pass



class SameVector:
    def embed(self, texts):
        return np.ones((len(texts), 8), dtype=np.float32)


class Broken:
    def embed(self, texts):
        raise RuntimeError("model file missing")


async def build(repo, settings, plans, store=None, embedder=None, threshold=0.0):
    store = store or FakeStore()
    cache = SemanticPlanCache(store, lambda: embedder if embedder is not None else SameVector(), threshold=threshold,
                              top_k=5, max_candidates=50, timeout_s=1.0, embedding_model="test-embedder")  # fmt: skip
    fake = FakeTranslator(plans)
    service = QueryService(repo, fake, settings, semantic_cache=cache)
    await service.refresh_vocabulary()
    return service, fake, store


async def ask(service, question, session_id=None):
    result = await service.handle(question, session_id)
    await service.semantic_cache.drain()
    return result


async def seed(service, store, question):
    """Ask once: the model answers and the checked plan is stored as a reusable template (no review step)."""
    first = await ask(service, question)
    assert first["tier"] == "llm" and len(store.docs) >= 1
    return first


# ------------------------------------------------------------------------------------------- A. valid paraphrases
async def test_room_paraphrases_reuse_one_stored_template(repo, settings):
    service, fake, store = await build(repo, settings, {"What is the room number of resident_1?": ROOM})
    first = await seed(service, store, "What is the room number of Eleanor Vance?")
    assert first["response"]["headline"].startswith("MC-101")
    for q in ("In which room is Eleanor Vance staying?", "Which room is allocated to Eleanor Vance?"):
        r = await ask(service, q)  # the fake model has no plan for these: any model call would fail the test
        assert r["tier"] == "template" and r["modelCalls"] == 0, q
        assert r["response"]["headline"].startswith("MC-101")
    assert len(fake.calls) == 1
    assert list(store.docs.values())[0]["usage_count"] == 2


# ------------------------------------------------------------------------------------------- C. slots are refilled
async def test_a_different_resident_is_resolved_fresh(repo, settings):
    service, fake, store = await build(repo, settings, {"What is the room number of resident_1?": ROOM})
    await seed(service, store, "What is the room number of Eleanor Vance?")
    r = await ask(service, "In which room is Arthur Pendleton staying?")
    assert r["tier"] == "template"
    assert r["response"]["headline"].startswith("MC-104"), "must be Arthur Pendleton's room, not Eleanor Vance's"


async def test_status_and_care_level_values_come_from_the_new_question(repo, settings):
    plan = {"action": "count", "filters": {"status": {"include": ["active"]}, "care_type": {"include": ["memory_care"]}}}
    service, fake, store = await build(repo, settings, {"How many active residents are in memory care?": plan})
    await seed(service, store, "How many active residents are in memory care?")
    r = await ask(service, "How many discharged residents are in assisted living?")
    assert r["tier"] == "template"
    f = r["intent"]["parameters"]["filters"]
    assert f["status"]["include"] == ["discharged"] and f["care_type"]["include"] == ["assisted_living"]


async def test_caregiver_and_facility_values_come_from_the_new_question(repo, settings):
    plan = {"action": "count", "filters": {"primary_caregiver": {"include": ["Nurse Sarah Jenkins, RN"]},
                                           "facility": {"include": ["North Wing"]}}}  # fmt: skip
    q = "How many residents in North Wing does Sarah Jenkins look after?"
    service, fake, store = await build(repo, settings, {q: plan})
    await seed(service, store, q)
    r = await ask(service, "How many residents in East Wing does Marcus Chen look after?")
    assert r["tier"] == "template"
    f = r["intent"]["parameters"]["filters"]
    assert f["primary_caregiver"]["include"] == ["Care Aide Marcus Chen"] and f["facility"]["include"] == ["East Wing"]
    doc = json.dumps(list(store.docs.values()), default=str)
    assert "Sarah" not in doc and "Jenkins" not in doc and "North Wing" not in doc  # F. privacy


async def test_dates_are_recomputed_for_the_new_question(repo, settings):
    plan = {"action": "count", "filters": {"admission_date": {"start": "2025-01-02"}}}
    q = "How many residents were admitted after January 1, 2025?"
    service, fake, store = await build(repo, settings, {q: plan})
    await seed(service, store, q)
    r = await ask(service, "How many residents were admitted after March 1, 2025?")
    assert r["tier"] == "template"
    assert r["intent"]["parameters"]["filters"]["admission_date"] == {"start": "2025-03-02"}


# ------------------------------------------------------------------------------------------- B. different meaning
@pytest.mark.parametrize(
    "question, plan",
    [
        ("Who is the primary caregiver of Eleanor Vance?", ("Who is the primary caregiver of resident_1?", CAREGIVER)),
        ("What is the care level of Eleanor Vance?", ("What is the care level of resident_1?", CARE_LEVEL)),
        ("When was Eleanor Vance admitted?", ("When was resident_1 admitted?", ADMITTED)),
    ],
)
async def test_a_different_lookup_attribute_never_reuses_the_room_template(repo, settings, question, plan):
    service, fake, store = await build(repo, settings, {"What is the room number of resident_1?": ROOM, plan[0]: plan[1]})
    await seed(service, store, "What is the room number of Eleanor Vance?")
    r = await ask(service, question)
    assert r["tier"] == "llm", "the room template must be rejected and the model asked"
    assert len(fake.calls) >= 2  # the model was asked (what IT answers is the existing pipeline's business)
    assert (r["intent"]["parameters"].get("lookup") or {}).get("attributes") != ["room_number"]


@pytest.mark.parametrize(
    "other, other_plan",
    [
        ("How many residents are younger than 75?", {"action": "count", "filters": {"age": {"max": 74}}}),
    ],
)
async def test_opposite_comparison_is_rejected(repo, settings, other, other_plan):
    q = "How many residents are older than 75?"
    service, fake, store = await build(repo, settings, {q: {"action": "count", "filters": {"age": {"min": 76}}},
                                                        other: other_plan})  # fmt: skip
    await seed(service, store, q)
    r = await ask(service, other)
    assert r["tier"] == "llm" and r["intent"]["parameters"]["filters"]["age"] == {"max": 74}


async def test_inclusive_bound_reuses_the_template_with_the_new_boundary(repo, settings):
    # same meaning class (a lower age bound); the boundary comes from THIS question: 75 or older → min 75, not 76
    q = "How many residents are older than 75?"
    service, fake, store = await build(repo, settings, {q: {"action": "count", "filters": {"age": {"min": 76}}}})
    await seed(service, store, q)
    r = await ask(service, "How many residents are aged 75 or older?")
    assert r["tier"] == "template" and r["intent"]["parameters"]["filters"]["age"] == {"min": 75}


async def test_or_and_and_are_different_requests(repo, settings):
    q_or, q_and = "How many residents are active or discharged?", "How many residents are active and discharged?"
    plan = {"action": "count", "filters": {"status": {"include": ["active", "discharged"]}}}
    service, fake, store = await build(repo, settings, {q_or: plan, q_and: {"action": "unsupported", "message": "x"}})
    await seed(service, store, q_or)
    r = await ask(service, q_and)
    assert r["tier"] != "template"
    assert logic_signature(q_or) != logic_signature(q_and)


@pytest.mark.parametrize(
    "other, plan",
    [
        ("How many residents were admitted before January 1, 2025?", {"action": "count", "filters": {"admission_date": {"end": "2024-12-31"}}}),
        ("How many residents were admitted between January 1, 2024, and December 31, 2025?",
         {"action": "count", "filters": {"admission_date": {"start": "2024-01-01", "end": "2025-12-31"}}}),
    ],
)  # fmt: skip
async def test_date_direction_and_range_are_not_reused(repo, settings, other, plan):
    q = "How many residents were admitted after January 1, 2025?"
    service, fake, store = await build(repo, settings, {q: {"action": "count", "filters": {"admission_date": {"start": "2025-01-02"}}},
                                                        other: plan})  # fmt: skip
    await seed(service, store, q)
    r = await ask(service, other)
    assert r["tier"] == "llm" and r["intent"]["parameters"]["filters"]["admission_date"] == plan["filters"]["admission_date"]


@pytest.mark.parametrize(
    "other, plan",
    [
        ("How many residents are in each status within every facility?", {"action": "breakdown", "group_by": ["facility", "status"]}),
        ("What is the total number of residents?", {"action": "count"}),
    ],
)  # fmt: skip
async def test_grouping_templates_are_not_reused_for_other_groupings(repo, settings, other, plan):
    q = "How many residents belong to each care level?"
    service, fake, store = await build(repo, settings, {q: {"action": "breakdown", "group_by": ["care_type"]}, other: plan})
    await seed(service, store, q)
    r = await ask(service, other)
    assert r["tier"] == "llm"


# ------------------------------------------------------------------------------------------- D. states, versions
async def test_stored_templates_are_reusable_immediately(repo, settings):
    # no review step (2026-11-18): a checked, executed plan is stored with revoked=False and reused at once
    service, fake, store = await build(repo, settings, {"What is the room number of resident_1?": ROOM})
    await ask(service, "What is the room number of Eleanor Vance?")
    assert [d["revoked"] for d in store.docs.values()] == [False] and "status" not in next(iter(store.docs.values()))
    r = await ask(service, "In which room is Eleanor Vance staying?")
    assert r["tier"] == "template" and len(fake.calls) == 1


async def test_templates_expire_after_their_ttl(repo, settings):
    plans = {"What is the room number of resident_1?": ROOM, "In which room is resident_1 staying?": ROOM}
    service, fake, store = await build(repo, settings, plans)
    await seed(service, store, "What is the room number of Eleanor Vance?")
    (doc,) = store.docs.values()
    assert abs((doc["expires_at"] - doc["created_at"]) - timedelta(hours=12)) < timedelta(seconds=1)
    doc["expires_at"] = datetime.now(UTC) - timedelta(seconds=1)  # past its TTL, MongoDB has not deleted it yet
    r = await ask(service, "In which room is Eleanor Vance staying?")
    assert r["tier"] == "llm" and len(fake.calls) == 2


async def test_atlas_store_writes_expires_at_and_ensures_the_ttl_index():
    from app.engine.semantic_cache import AtlasTemplateStore

    calls = {}

    class Coll:
        async def update_one(self, flt, update, upsert):
            calls["update"] = update

        async def create_index(self, key, **kw):
            calls["index"] = (key, kw)

    class Admin:
        async def command(self, name):
            return {"ok": 1}

    store = AtlasTemplateStore.__new__(AtlasTemplateStore)
    store._coll, store._client, store.ttl_seconds = Coll(), type("C", (), {"admin": Admin()})(), 12 * 3600
    await store.connect()
    assert calls["index"] == ("expires_at", {"expireAfterSeconds": 0, "name": "expires_at_ttl"})
    await store.store({"_id": "t1", "question_text": "q"})
    ins = calls["update"]["$setOnInsert"]
    assert abs((ins["expires_at"] - ins["created_at"]) - timedelta(hours=12)) < timedelta(seconds=1)
    assert "expires_at" not in calls["update"]["$set"]  # storing again never extends the lifetime


async def test_documents_without_the_revoked_flag_are_never_reused(repo, settings):
    # e.g. written by an older version: only an explicit revoked=False makes a template usable
    plans = {"What is the room number of resident_1?": ROOM, "In which room is resident_1 staying?": ROOM}
    service, fake, store = await build(repo, settings, plans, store=FakeStore(ignore_filter=True))
    await seed(service, store, "What is the room number of Eleanor Vance?")
    for d in store.docs.values():
        d.pop("revoked")
    r = await ask(service, "In which room is Eleanor Vance staying?")
    assert r["tier"] == "llm" and len(fake.calls) == 2


async def test_revoked_templates_are_never_reused(repo, settings):
    plans = {"What is the room number of resident_1?": ROOM, "In which room is resident_1 staying?": ROOM}
    service, fake, store = await build(repo, settings, plans, store=FakeStore(ignore_filter=True))
    await seed(service, store, "What is the room number of Eleanor Vance?")
    for d in store.docs.values():
        d.update(revoked=True, revoked_at="2026-11-18T00:00:00Z")
    r = await ask(service, "In which room is Eleanor Vance staying?")
    assert r["tier"] == "llm"


@pytest.mark.parametrize("field", ["request_ir_version", "vocabulary_hash", "schema_hash", "model_version", "doc_version"])
async def test_incompatible_templates_are_rejected(repo, settings, field):
    plans = {"What is the room number of resident_1?": ROOM, "In which room is resident_1 staying?": ROOM}
    service, fake, store = await build(repo, settings, plans, store=FakeStore(ignore_filter=True))
    await seed(service, store, "What is the room number of Eleanor Vance?")
    for d in store.docs.values():  # a template written by another app/model/schema version
        if field == "doc_version":
            d["doc_version"] = TEMPLATE_DOC_VERSION + 1
        else:
            d["compat_key"] = "other-" + field
    r = await ask(service, "In which room is Eleanor Vance staying?")
    assert r["tier"] == "llm"


async def test_a_different_llm_model_does_not_see_old_templates(repo, settings):
    store = FakeStore()
    service, _, _ = await build(repo, settings, {"What is the room number of resident_1?": ROOM}, store=store)
    await seed(service, store, "What is the room number of Eleanor Vance?")
    other = settings.model_copy(update={"llm_model_name": "another-model"})
    service2, fake2, _ = await build(repo, other, {"In which room is resident_1 staying?": ROOM}, store=store)
    r = await ask(service2, "In which room is Eleanor Vance staying?")
    assert r["tier"] == "llm" and len(fake2.calls) == 1


# ------------------------------------------------------------------------------------------- E. failures → Qwen
@pytest.mark.parametrize("failure", ["embedder", "store_error", "store_timeout"])
async def test_cache_failures_fall_back_to_the_model(repo, settings, failure):
    store = FakeStore()
    plans = {"What is the room number of resident_1?": ROOM, "In which room is resident_1 staying?": ROOM}
    service, fake, _ = await build(repo, settings, plans, store=store)
    await seed(service, store, "What is the room number of Eleanor Vance?")
    if failure == "embedder":
        service.semantic_cache._embedder_provider = lambda: Broken()
    elif failure == "store_error":
        store.fail = ConnectionError("Atlas unreachable / index missing")
    else:
        store.delay = 5.0  # longer than the 1 s cache timeout
    r = await ask(service, "In which room is Eleanor Vance staying?")
    assert r["tier"] == "llm" and r["response"]["headline"].startswith("MC-101")
    assert service.semantic_cache.stats.errors == 1


async def test_low_similarity_is_not_a_candidate(repo, settings):
    plans = {"What is the room number of resident_1?": ROOM, "In which room is resident_1 staying?": ROOM}
    service, fake, store = await build(repo, settings, plans, threshold=1.01 - 1e-9)
    await seed(service, store, "What is the room number of Eleanor Vance?")
    r = await ask(service, "In which room is Eleanor Vance staying?")
    assert r["tier"] == "llm"


async def test_two_different_plans_that_both_fit_are_ambiguous():
    store = FakeStore()
    cache = SemanticPlanCache(store, SameVector, threshold=0.0, top_k=5, max_candidates=50, timeout_s=1.0,
                              embedding_model="t")  # fmt: skip
    from app.engine.semantic_cache import Compatibility

    compat = Compatibility("v", "h", "s", "m", "e")
    for i, attrs in enumerate((["room_number"], ["care_type"])):
        store.docs[str(i)] = {"_id": str(i), "revoked": False, "compat_key": compat.key, "doc_version": TEMPLATE_DOC_VERSION,
                              "logic_signature": [], "embedding": [1.0] * 8, "mention_counts": {},
                              "plan_template": {"action": "lookup", "filters": {}, "lookup": {"residents": ["resident_1"], "attributes": attrs}},
                              "slots": [{"name": "residents", "type": "resident", "placeholders": ["resident_1"]}]}  # fmt: skip

    class Index:
        statuses, care_types = [], []

        def mentioned_wings(self, t):
            return {}

        def mentioned_caregivers(self, t):
            return {}

    view = QuestionView("details of resident_1", ["resident_1"], scan_question("details of resident_1", __import__("datetime").date(2026, 11, 9)), Index(), {})

    async def always_ok(plan):
        return None

    res = await cache.lookup(view, compat, always_ok)
    assert res.plan is None and res.reason == "ambiguous_candidates"


# ------------------------------------------------------------------------------------------- G. fast-path seeding
async def test_fast_path_plan_is_stored_and_serves_a_paraphrase(repo, settings):
    from app.nlu.intent_router import SemanticIntentRouter
    from tests.test_direct_query import FakeAurelioRouter

    scores = {"lookup": 0.95}
    router = SemanticIntentRouter(router=FakeAurelioRouter(scores))
    store = FakeStore()
    cache = SemanticPlanCache(store, SameVector, threshold=0.0, top_k=5, max_candidates=50, timeout_s=1.0,
                              embedding_model="t")  # fmt: skip
    fake = FakeTranslator(fail=True)  # the model is down: everything must come from the fast path or a template
    service = QueryService(repo, fake, settings, intent_router=router, semantic_cache=cache)
    await service.refresh_vocabulary()
    first = await ask(service, "What is the room number of Eleanor Vance?")
    assert first["tier"] == "direct" and first["response"]["headline"].startswith("MC-101")
    (doc,) = store.docs.values()
    assert doc["revoked"] is False and doc["source"] == "fast_path"
    assert doc["plan_template"]["lookup"]["residents"] == ["resident_1"]  # the id was mapped back to the placeholder
    assert "RES-" not in json.dumps(doc, default=str) and "Eleanor" not in json.dumps(doc, default=str)
    scores.clear()
    scores["help"] = 0.10  # from now on the fast path is not confident: only a template (or the model) can answer
    calls_before = len(fake.calls)
    r = await ask(service, "In which room is Arthur Pendleton staying?")
    assert r["tier"] == "template" and r["response"]["headline"].startswith("MC-104")
    assert len(fake.calls) == calls_before  # answered without the model


# ------------------------------------------------------------------------------------------- G. eligibility
async def test_follow_up_questions_are_neither_recorded_nor_served(repo, settings):
    plans = {
        "How many residents are active?": {"action": "count", "filters": {"status": {"include": ["active"]}}},
        "How many of them are in memory care?": {"action": "count", "uses_previous": True,
                                                 "filters": {"status": {"include": ["active"]}, "care_type": {"include": ["memory_care"]}}},
    }  # fmt: skip
    service, fake, store = await build(repo, settings, plans)
    await ask(service, "How many residents are active?", session_id="s")
    before = set(store.docs)
    r = await ask(service, "How many of them are in memory care?", session_id="s")
    assert r["tier"] == "llm" and set(store.docs) == before


def _view(service, question):
    mask = service.index.mask(question)
    from datetime import date

    return QuestionView(mask.text, list(mask.placeholders), scan_question(mask.text, date.fromisoformat(TODAY)),
                        service.index, service.date_anchors)  # fmt: skip


async def test_templates_store_no_values_and_must_reproduce_their_own_plan(repo, settings):
    service, _, _ = await build(repo, settings, {})
    view = _view(service, "How many active residents does Sarah Jenkins look after in North Wing?")
    plan = {"action": "count", "filters": {"status": {"include": ["active"]},
                                           "primary_caregiver": {"include": ["Nurse Sarah Jenkins, RN"]},
                                           "facility": {"include": ["North Wing"]}}}  # fmt: skip
    template, slots = templatize(plan, view)
    text = slot_text(view, mentions(view))
    assert "Sarah" not in text and "North" not in text and "active" not in text.lower()
    assert "Jenkins" not in json.dumps(template) and {s["type"] for s in slots} == {"status", "primary_caregiver", "facility"}
    with pytest.raises(Ineligible):  # a value the question never names cannot be filled from a later question
        templatize({"action": "count", "filters": {"status": {"include": ["discharged"]}}}, view)
    with pytest.raises(Ineligible):  # room numbers are not slot-able
        templatize({"action": "count", "filters": {"room_number": {"include": ["MC-101"]}}}, _view(service, "Who is in room MC-101?"))


async def test_step_log_shows_template_hit_and_recording(repo, settings, caplog):
    service, fake, store = await build(repo, settings, {"What is the room number of resident_1?": ROOM})
    with caplog.at_level(logging.INFO, logger="app.engine.service"):
        first = await ask(service, "What is the room number of Eleanor Vance?")
    rec = {r.step_name: r for r in caplog.records if getattr(r, "query_id", None) == first["queryId"] and hasattr(r, "step_name")}
    assert rec["semantic_cache"].status == "miss" and rec["remember"].template_recorded is True
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="app.engine.service"):
        second = await ask(service, "In which room is Arthur Pendleton staying?")
    rec = {r.step_name: r for r in caplog.records if getattr(r, "query_id", None) == second["queryId"] and hasattr(r, "step_name")}
    assert rec["semantic_cache"].status == "hit" and rec["local_model"].status == "skipped"
    assert rec["validate_plan"].status == "passed" and rec["remember"].template_used


# ------------------------------------------------------------------------------------------- real BGE measurement
@pytest.mark.skipif(not MODELS_DIR.exists(), reason="embedding model not downloaded")
async def test_real_bge_false_hits_and_missed_paraphrases(repo, settings):
    """Approved templates for 6 questions; every probe below is asked with the REAL embedder and threshold 0.88.
    Probes marked True must reuse the template, False must not. False hits are counted separately (must be 0)."""
    from app.nlu.topic_gate import FastEmbedEmbedder

    bge = FastEmbedEmbedder("BAAI/bge-small-en-v1.5", str(MODELS_DIR))
    seeds = {
        "What is the room number of resident_1?": ROOM,
        "How many residents are older than 75?": {"action": "count", "filters": {"age": {"min": 76}}},
        "How many residents are active or discharged?": {"action": "count", "filters": {"status": {"include": ["active", "discharged"]}}},
        "How many residents were admitted after January 1, 2025?": {"action": "count", "filters": {"admission_date": {"start": "2025-01-02"}}},
        "How many residents belong to each care level?": {"action": "breakdown", "group_by": ["care_type"]},
    }  # fmt: skip
    probes = [  # (question with real names, expected reuse?, model plan when not reused)
        ("In which room is Eleanor Vance staying?", True, None),
        ("Which room is allocated to Eleanor Vance?", True, None),
        ("In which room is Arthur Pendleton staying?", True, None),
        ("Who is the primary caregiver of Eleanor Vance?", False, CAREGIVER),
        ("What is the care level of Eleanor Vance?", False, CARE_LEVEL),
        ("When was Eleanor Vance admitted?", False, ADMITTED),
        ("How many residents are younger than 75?", False, {"action": "count", "filters": {"age": {"max": 74}}}),
        ("How many residents are active and discharged?", False, {"action": "unsupported", "message": "x"}),
        ("How many residents were admitted before January 1, 2025?", False, {"action": "count", "filters": {"admission_date": {"end": "2024-12-31"}}}),
        ("How many residents were admitted between January 1, 2024, and December 31, 2025?", False,
         {"action": "count", "filters": {"admission_date": {"start": "2024-01-01", "end": "2025-12-31"}}}),
        ("How many residents are in each status within every facility?", False, {"action": "breakdown", "group_by": ["facility", "status"]}),
        ("What is the total number of residents?", False, {"action": "count"}),
    ]  # fmt: skip
    plans = dict(seeds)
    for q, _, p in probes:
        if p is not None:
            plans[service_mask(repo, q)] = p
    service, fake, store = await build(repo, settings, plans, embedder=bge, threshold=0.88)
    for q in seeds:
        await ask(service, q.replace("resident_1", "Eleanor Vance"))
    false_hits, misses = [], []
    for q, should_reuse, _ in probes:
        r = await ask(service, q)
        reused = r["tier"] == "template"
        if reused and not should_reuse:
            false_hits.append(q)
        if should_reuse and not reused:
            misses.append(q)
    print(f"\nreal BGE @0.88: false hits {len(false_hits)} {false_hits}; missed paraphrases {len(misses)} {misses}")
    assert false_hits == []
    assert misses == []


def service_mask(repo, question):
    """The masked form the fake model is keyed by (each question names at most one resident → resident_1)."""
    return question.replace("Eleanor Vance", "resident_1").replace("Arthur Pendleton", "resident_1")
