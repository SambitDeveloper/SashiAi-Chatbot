"""Retry policy and failure cache: retry only what a model can correct, detect a retry that changes nothing, and never
pay twice for a failure in the same context."""

import json

from tests.conftest import FakeTranslator
from tests.test_service import ACTIVE_OCTOBER, ELENA_MODEL_Q, ELENA_Q


async def test_non_correctable_failure_is_not_retried(make_service):
    """'them' with no earlier turn: no plan can fix that, so one model call and then a clarifying question."""
    q = "How many of them are female?"
    fake = FakeTranslator({q: {"action": "count", "filters": {"gender": {"include": ["Female"]}}}})
    service = await make_service(fake)
    r = await service.handle(q, session_id="fresh")
    assert len(fake.calls) == 1 and r["modelCalls"] == 1
    assert r["response"]["needsClarification"] and r["mcpCall"] is None
    assert any(t.get("retry_skipped") for t in r["modelPlans"])
    assert "'them'" in r["response"]["summaryText"]


async def test_correctable_failure_is_retried_once(make_service):
    fake = FakeTranslator({ELENA_MODEL_Q: [ACTIVE_OCTOBER, ACTIVE_OCTOBER]})
    service = await make_service(fake)
    r = await service.handle(ELENA_Q)
    assert len(fake.calls) == 2 and fake.calls[1]["hint"]  # the retry carried the problems as feedback
    assert not any(t.get("retry_skipped") for t in r["modelPlans"])


async def test_identical_retry_is_detected_after_normalisation(make_service):
    """Same plan with keys and conditions in a different order is still the same plan."""
    reordered = {
        "filters": {"admission_date": {"end": "2026-10-31", "start": "2026-10-01"}, "status": {"include": ["active"]}},
        "action": "count",
    }
    fake = FakeTranslator({ELENA_MODEL_Q: [ACTIVE_OCTOBER, reordered]})
    service = await make_service(fake)
    r = await service.handle(ELENA_Q)
    attempts = [t for t in r["modelPlans"] if "attempt" in t]
    assert [t["attempt"] for t in attempts] == [1, 2]
    assert attempts[1].get("identical_to_previous") is True


async def test_different_retry_is_not_flagged_identical(make_service):
    other = {"action": "count", "filters": {"status": {"include": ["active"]}}}
    fake = FakeTranslator({ELENA_MODEL_Q: [ACTIVE_OCTOBER, other]})
    service = await make_service(fake)
    r = await service.handle(ELENA_Q)
    assert not any(t.get("identical_to_previous") for t in r["modelPlans"])


async def test_retry_receives_candidate_and_preserves_its_valid_date_filter(make_service):
    question = (
        "Among residents admitted after August 1, 2026, how many are Active and how many are Discharged? "
        "Show the names and room numbers in each group."
    )
    filters = {
        "admission_date": {"start": "2026-08-02"},
        "status": {"include": ["active", "discharged"]},
    }
    first = {"action": "list", "filters": filters, "fields": ["primary_caregiver"]}
    repaired = {
        "action": "breakdown",
        "filters": {
            "admission_date": {"start": "2026-08-02"},
            "status": {"include": ["active", "discharged"]},
        },
        "group_by": ["status"],
        "fields": ["room_number"],
    }
    fake = FakeTranslator({question: [first, repaired]})
    service = await make_service(fake)

    result = await service.handle(question)

    assert len(fake.calls) == 2
    assert fake.calls[1]["retry_ir_json"]
    assert "2026-08-02" in fake.calls[1]["retry_ir_json"]
    assert fake.calls[1]["dates"]
    assert result["intent"]["parameters"]["filters"]["admission_date"]["start"] == "2026-08-02"
    assert result["intent"]["parameters"]["group_by"] == ["status"]
    assert result["response"]["columns"][0]["key"] == "roomNumber"
    assert "2026-08-02" in json.dumps(result["modelPlans"][1]["requestIR"])


async def test_retry_with_a_date_changed_by_backend_validation_fails_closed(make_service):
    question = (
        "Among residents admitted after August 1, 2026, how many are Active and how many are Discharged? "
        "Show the names and room numbers in each group."
    )
    first = {
        "action": "breakdown",
        "filters": {
            "admission_date": {"start": "2026-08-02"},
            "status": {"include": ["active", "discharged"]},
        },
        "group_by": ["status"],
    }
    retry = {
        "action": "breakdown",
        "filters": {
            "admission_date": {"start": "2026-08-03"},
            "status": {"include": ["active", "discharged"]},
        },
        "group_by": ["status"],
        "fields": ["room_number"],
    }
    fake = FakeTranslator({question: [first, retry]})
    service = await make_service(fake)

    result = await service.handle(question)

    assert len(fake.calls) == 2
    assert result["response"].get("needsClarification")
    assert result["mcpCall"] is None


async def test_fabricated_resident_reference_is_rejected_and_retried(make_service):
    question = "How many active residents are there?"
    fake = FakeTranslator(
        {
            question: [
                {"action": "lookup", "lookup": {"residents": ["resident_1"]}},
                {"action": "count", "filters": {"status": {"include": ["active"]}}},
            ]
        }
    )
    service = await make_service(fake)

    result = await service.handle(question)

    assert len(fake.calls) == 2
    assert fake.calls[1]["hint"] and "Do not invent resident references" in fake.calls[1]["hint"]
    assert result["intent"]["parameters"]["action"] == "count"
    assert not result["response"].get("needsClarification")


async def test_date_filtered_status_breakdown_returns_group_members_and_projection(make_service):
    question = (
        "Among residents admitted after August 1, 2026, how many are Active and how many are Discharged? "
        "Show the names and room numbers in each group."
    )
    fake = FakeTranslator(
        {
            question: {
                "action": "breakdown",
                "filters": {
                    "admission_date": {"start": "2026-08-02"},
                    "status": {"include": ["active", "discharged"]},
                },
                "group_by": ["status"],
                "fields": ["room_number"],
            }
        }
    )
    service = await make_service(fake)

    result = await service.handle(question)

    assert result["intent"]["parameters"]["filters"]["admission_date"]["start"] == "2026-08-02"
    assert result["intent"]["parameters"]["filters"]["status"]["include"] == ["active", "discharged"]
    assert result["intent"]["parameters"]["group_by"] == ["status"]
    assert result["intent"]["parameters"]["fields"] == ["room_number"]
    groups = {group["label"]: group for group in result["response"]["breakdown"]}
    assert set(groups) == {"Active", "Discharged"}
    assert all(group["count"] == len(group["residents"]) for group in groups.values())
    assert all(
        {"firstName", "lastName", "roomNumber"} <= set(resident)
        for group in groups.values()
        for resident in group["residents"]
    )


async def test_failure_cache_is_scoped_to_conversation_context(make_service):
    active = {"action": "count", "filters": {"status": {"include": ["active"]}}}
    fake = FakeTranslator({ELENA_MODEL_Q: [ACTIVE_OCTOBER, ACTIVE_OCTOBER], "How many residents are active?": active})
    service = await make_service(fake)
    await service.handle(ELENA_Q, session_id="a")
    again = await service.handle(ELENA_Q, session_id="b")  # also no previous turn: same context → cached failure
    assert len(fake.calls) == 2 and again["tier"] == "cache" and again["modelCalls"] == 0
    assert any(t.get("cached_failure") for t in again["modelPlans"])
    # a different conversation (there is a previous turn now): the failure is not reused
    await service.handle("How many residents are active?", session_id="c")
    calls_before = len(fake.calls)
    other = await service.handle(ELENA_Q, session_id="c")
    assert len(fake.calls) > calls_before and other["modelCalls"] >= 1


async def test_failure_cache_is_scoped_to_schema_version(make_service):
    fake = FakeTranslator({ELENA_MODEL_Q: [ACTIVE_OCTOBER, ACTIVE_OCTOBER]})
    service = await make_service(fake)
    await service.handle(ELENA_Q)
    service.vocab_version = "changed"  # vocabulary/schema changed: earlier interpretations no longer apply
    await service.handle(ELENA_Q)
    assert len(fake.calls) == 4


async def test_template_compatibility_follows_meaningful_vocabulary_only(make_service):
    # replaces the exact plan-cache scoping test (cache removed 2026-11-18). Stored semantic templates are scoped by
    # statuses + care types (what a plan MEANS); a new resident/room/caregiver must not invalidate every template.
    import dataclasses

    service = await make_service(FakeTranslator({}))
    base = service._template_compat().key
    vocab = await service.repo.vocabulary()
    more_residents = dataclasses.replace(vocab, residents=vocab.residents + [{"residentId": "RES-9999", "firstName": "New", "lastName": "Person"}])
    from app.nlu.vocabulary import EntityIndex

    service.index = EntityIndex(more_residents)
    assert service._template_compat().key == base
    fewer_statuses = [s for s in vocab.statuses if s != "pending"]  # the database no longer has "pending"
    service.index = EntityIndex(dataclasses.replace(vocab, statuses=fewer_statuses))
    assert service._template_compat().key != base


def test_cache_key_changes_with_version_scope_and_model():
    from app.engine.service import _cache_key, _session_cache_key

    assert _cache_key("2026-11-13", "q", None, "v1|mongo:db.c|m") != _cache_key("2026-11-13", "q", None, "v2|mongo:db.c|m")
    assert _cache_key("2026-11-13", "q", None, "v1|mongo:db.c|m") != _cache_key("2026-11-13", "q", None, "v1|mongo:db.d|m")
    assert _cache_key("2026-11-13", "q", None, "v1|mongo:db.c|m") != _cache_key("2026-11-13", "q", None, "v1|mongo:db.c|n")
    assert _session_cache_key("session", "mongo:db.c") != _session_cache_key("session", "mongo:db.d")


async def test_fast_path_blocking_problem_asks_without_the_model():
    """A list that asks for a single-record-only field: the fast path sees it can't be done and asks, no model call."""
    from tests.test_direct_query import make_service as make_routed_service

    fake = FakeTranslator({})
    service = await make_routed_service({"list": 0.93, "count": 0.80}, fake)
    r = await service.handle("List active residents and show their emergency contacts", session_id="s")
    assert fake.calls == [] and r["modelCalls"] == 0
    assert r["response"].get("needsClarification") and r["mcpCall"] is None
    assert "emergency contact" in r["response"]["summaryText"].lower()
