"""Numbered step logs (app/engine/steps.py): every question gets step-01 … step-19, once each, in order, without PII."""

import logging

import pytest

from app.data.memory_repository import MemoryResidentRepository
from app.engine.service import QueryService
from app.engine.steps import STEPS
from app.nlu.topic_gate import TopicGate
from tests.conftest import FakeTranslator, PinnedDateSettings
from tests.test_direct_query import FakeAurelioRouter
from tests.test_topic_gate import FakeEmbedder

ALL = [f"step-{i:02d} {name}" for i, name in enumerate(STEPS, 1)]
COUNT_PLAN = {"action": "count", "conditions": [{"field": "status", "include": ["active"]}]}


async def service_with(translator, gate=None, router_scores=None) -> QueryService:
    from app.nlu.intent_router import SemanticIntentRouter

    settings = PinnedDateSettings(_env_file=None, environment="test", topic_gate_enabled=False,
                                  semantic_router_enabled=False)  # fmt: skip
    router = SemanticIntentRouter(router=FakeAurelioRouter(router_scores)) if router_scores else None
    service = QueryService(MemoryResidentRepository(), translator, settings, gate=gate, intent_router=router)
    await service.refresh_vocabulary()
    return service


def steps(caplog, query_id):
    return [r for r in caplog.records if getattr(r, "step_name", None) and r.query_id == query_id]


def assert_complete(records):
    """Exactly 19 lines, numbered 01–19, in pipeline order."""
    assert [r.getMessage().split(":")[0] for r in records] == ALL
    assert [r.step for r in records] == list(range(1, 20))


def by_name(records):
    return {r.step_name: r for r in records}


async def ask(service, caplog, question, session_id=None):
    with caplog.at_level(logging.INFO, logger="app.engine.service"):
        result = await service.handle(question, session_id)
    return result, steps(caplog, result["queryId"])


async def test_model_path_logs_every_step_in_order(caplog):
    q = "How many residents are active?"
    service = await service_with(FakeTranslator({q.lower(): COUNT_PLAN}))
    result, recs = await ask(service, caplog, q)
    assert_complete(recs)
    s = by_name(recs)
    assert s["plan_cache"].status == "miss"
    assert s["local_model"].status == "planned" and s["local_model"].completion_tokens == 40
    assert s["validate_plan"].status == "passed"
    assert s["execute_query"].status == "ok" and s["execute_query"].matched == result["response"]["count"]
    assert s["remember"].plan_cached is True
    assert s["finished"].status == "answered" and s["finished"].tier == "llm" and s["finished"].model_calls == 1
    assert isinstance(s["finished"].total_ms, int)


async def test_cache_hit_skips_model_steps_with_reason(caplog):
    q = "How many residents are active?"
    service = await service_with(FakeTranslator({q.lower(): COUNT_PLAN}))
    await ask(service, caplog, q)
    caplog.clear()
    result, recs = await ask(service, caplog, q)
    assert_complete(recs)
    s = by_name(recs)
    assert s["plan_cache"].status == "hit" and result["tier"] == "cache"
    for name in ("topic_check", "policy_check", "intent_classifier", "fast_path", "local_model"):
        assert s[name].status == "skipped" and s[name].reason == "plan cache hit"
    assert s["execute_query"].status == "ok"


async def test_off_topic_stops_at_topic_check_and_still_logs_19(caplog):
    gate = TopicGate(0.02, embedder_factory=FakeEmbedder)
    gate.load()
    fake = FakeTranslator({})
    service = await service_with(fake, gate=gate)
    _, recs = await ask(service, caplog, "What's the weather today?")
    assert_complete(recs)
    s = by_name(recs)
    assert s["topic_check"].status == "blocked" and s["topic_check"].method == "embedding (bge-small)"
    assert isinstance(s["topic_check"].resident_score, float)
    assert s["local_model"].status == "skipped" and s["local_model"].reason == "not reached (off_topic)"
    assert s["finished"].status == "off_topic" and fake.calls == []


async def test_direct_answer_logs_classifier_and_fast_path(caplog):
    fake = FakeTranslator({})
    service = await service_with(fake, router_scores={"count": 0.93, "list": 0.80})
    _, recs = await ask(service, caplog, "how many active residents does Care Aide Elena Rostova look after")
    assert_complete(recs)
    s = by_name(recs)
    assert s["intent_classifier"].status == "scored" and s["intent_classifier"].route == "count"
    assert s["fast_path"].status == "answered"
    assert s["local_model"].status == "skipped" and fake.calls == []
    assert s["finished"].tier == "direct"


async def test_model_unavailable_still_logs_19(caplog):
    service = await service_with(FakeTranslator(fail=True))
    _, recs = await ask(service, caplog, "How many residents are active?")
    assert_complete(recs)
    s = by_name(recs)
    assert s["local_model"].status == "unavailable"
    assert s["execute_query"].reason == "not reached (model_unavailable)"
    assert s["finished"].status == "model_unavailable"


async def test_error_inside_a_step_logs_failed_and_reraises(caplog, monkeypatch):
    q = "How many residents are active?"
    service = await service_with(FakeTranslator({q.lower(): COUNT_PLAN}))

    async def boom(*a, **k):
        raise RuntimeError("database down")

    monkeypatch.setattr("app.engine.service.execute", boom)
    with caplog.at_level(logging.INFO, logger="app.engine.service"), pytest.raises(RuntimeError):
        await service.handle(q)
    recs = [r for r in caplog.records if getattr(r, "step_name", None)]
    assert_complete(recs)
    s = by_name(recs)
    assert s["execute_query"].status == "skipped" and s["finished"].status == "failed"
    assert s["finished"].error == "RuntimeError"


async def test_step_logs_never_contain_names_or_question_text(caplog):
    q = "Who is the primary caretaker of Eleanor Vance ?"
    plan = {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["primary_caregiver"]}}
    service = await service_with(FakeTranslator({"who is the primary caretaker of resident_1 ?": plan}))
    _, recs = await ask(service, caplog, q, session_id="s1")
    assert_complete(recs)
    assert by_name(recs)["mask_names"].residents_masked == 1
    dump = " ".join(str(r.__dict__) for r in recs)
    assert "Eleanor" not in dump and "Vance" not in dump and "caretaker" not in dump
