"""Direct answers (semantic router → MongoDB MCP, no language model): safety rules, service flow, accuracy."""

import logging
import os
import shutil
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.data.memory_repository import MemoryResidentRepository
from app.engine.executor import execute
from app.engine.resolver import resolve_plan
from app.engine.service import QueryService
from app.models.plan import QueryPlan
from app.nlu.direct_query import DirectQueryBuilder
from app.nlu.grammar import skeleton as grammar_skeleton
from app.nlu.intent_router import IntentScore, SemanticIntentRouter
from app.nlu.vocabulary import EntityIndex
from tests.conftest import FakeTranslator, PinnedDateSettings
from tests.data.direct_cases import CASES

MODELS_DIR = Path(__file__).resolve().parents[1] / ".models"
PIN = date(2026, 11, 9)  # reference "today" for deterministic relative-date parsing


def confident(route: str) -> IntentScore:
    return IntentScore(route, 0.95, 0.20, {route: 0.95})


@pytest.fixture
async def index():
    return EntityIndex(await MemoryResidentRepository().vocabulary())


@pytest.fixture
def builder():
    return DirectQueryBuilder(SemanticIntentRouter().config)


def build(builder, index, question, route):
    return builder.build(question, index.mask(question), confident(route), index, PIN)


# ---------------------------------------------------------------- answers
async def test_count_with_database_values(builder, index):
    r = build(builder, index, "how many active residents does Care Aide Elena Rostova look after", "count")
    assert r.plan == {
        "action": "count",
        "filters": {"status": {"include": ["active"]}, "primary_caregiver": {"include": ["Care Aide Elena Rostova"]}},
    }


async def test_breakdown_uses_route_metadata(builder, index):
    assert build(builder, index, "how many residents in each wing", "breakdown_wing").plan == {
        "action": "breakdown",
        "filters": {},
        "group_by": ["facility"],  # canonical name (the data has no wing; "wing" is accepted as an alias)
    }


async def test_lookup_answers_only_what_was_asked(builder, index):
    # Previously every lookup returned the full profile; now the asked-for attribute is the answer.
    r = build(builder, index, "Who is the primary caretaker of Eleanor Vance ?", "lookup")
    assert r.plan == {"action": "lookup", "lookup": {"residents": ["RES-1001"], "attributes": ["primary_caregiver"]}}


@pytest.mark.parametrize(
    ("question", "attributes"),
    [
        ("What is the room number of Eleanor Vance?", ["room_number"]),
        ("Tell me about Robert MacKenzi category and staus", ["care_type", "status"]),  # typo "staus" still counts
        ("how old is Eleanor Vance", ["age"]),
        ("emergency contact for Eleanor Vance", ["emergency_contact"]),
        ("Tell me about Eleanor Vance", []),  # nothing specific asked → the full profile
    ],
)
async def test_lookup_attributes_follow_the_question(builder, index, question, attributes):
    r = build(builder, index, question, "lookup")
    assert r.plan is not None and r.plan["lookup"]["attributes"] == attributes and "filters" not in r.plan


async def test_when_admitted_asks_for_the_date_not_a_status_check(builder, index):
    r = build(builder, index, "when was Eleanor Vance admitted", "lookup")
    assert r.plan["lookup"]["attributes"] == ["admission_date"] and "filters" not in r.plan
    # the yes/no form stays a status check
    r = build(builder, index, "is Eleanor Vance admitted", "lookup")
    assert r.plan.get("filters") == {"status": {"include": ["admitted"]}}


async def test_resident_name_never_counts_as_an_attribute_word(builder, index):
    # "Carter"-style surnames must not be read as "carer"; names are excluded from attribute matching
    r = build(builder, index, "Tell me about Rosalind Franklin", "lookup")
    assert r.plan["lookup"]["attributes"] == []


# ---------------------------------------------------------------- must go to the language model
@pytest.mark.parametrize(
    ("question", "route", "reason"),
    [
        ("And how many of them are in memory care?", "count", "refers_to_previous"),  # follow-up: needs the conversation
        ("the 3 oldest residents", "oldest", "has_number"),  # ranking by count is not inferred
        ("active residents with diabetes", "list", "not_understood"),  # unknown attribute, never dropped
        ("how many residents in the north", "count", "not_understood"),  # wing word without a wing: never dropped
        ("Tell me about Roosevelt", "lookup", "ambiguous_resident"),
        ("tell me about the resident", "lookup", "missing_resident"),
        ("how many active residents are there", "lookup", "missing_resident"),  # a condition but no resident named
        ("RES-9999", "lookup", "unknown_resident_id"),
        ("hi active", "help", "field_not_allowed:status"),  # help route accepts no conditions
    ],
)
async def test_unsure_questions_go_to_the_model(builder, index, question, route, reason):
    r = build(builder, index, question, route)
    assert r.plan is None and r.reason.startswith(reason.split(":")[0]) and r.reason == reason


async def test_confidence_and_margin_rules(builder, index):
    q = "How many residents are active?"
    assert builder.build(q, index.mask(q), IntentScore("count", 0.70, 0.2, {}), index, PIN).reason == "low_confidence"
    assert builder.build(q, index.mask(q), IntentScore("count", 0.95, 0.01, {}), index, PIN).reason == "low_margin"


# ---------------------------------------------------------------- service flow
class FakeAurelioRouter:
    def __init__(self, scores):
        self.scores = scores

    def __call__(self, text, limit=None):
        return [SimpleNamespace(name=n, confidence=c) for n, c in self.scores.items()]


async def make_service(scores, translator):
    settings = PinnedDateSettings(
        _env_file=None, environment="test", topic_gate_enabled=False, semantic_router_enabled=False
    )
    router = SemanticIntentRouter(router=FakeAurelioRouter(scores))
    service = QueryService(MemoryResidentRepository(), translator, settings, intent_router=router)
    await service.refresh_vocabulary()
    return service


async def test_direct_answer_never_calls_the_model(caplog):
    fake = FakeTranslator({})
    service = await make_service({"count": 0.93, "list": 0.80}, fake)
    with caplog.at_level(logging.INFO, logger="app.engine.service"):
        result = await service.handle("how many active residents does Care Aide Elena Rostova look after")
    assert result["tier"] == "direct" and result["tokensConsumed"] == 0 and fake.calls == []
    assert result["response"]["count"] == 11  # all 11 of Care Aide Elena Rostova's residents are active
    assert any(r.getMessage() == "direct answer via MongoDB MCP (no language model)" for r in caplog.records)


async def test_unsure_question_goes_to_the_model_and_logs_why(caplog):
    # An interpreter-reported unresolved term is non-executable; unknown vocabulary is not treated as a plan.
    q = "active residents with profiles"
    from app.nlu.request_ir import request_ir_from_plan

    candidate = request_ir_from_plan(
        {"action": "count", "filters": {"status": {"include": ["active"]}}},
        q,
    )
    candidate["unresolved"] = [{"reason": "unresolved request material", "evidence": "profiles"}]
    fake = FakeTranslator({q: {"request_ir": candidate}})
    service = await make_service({"count": 0.93, "list": 0.80}, fake)
    with caplog.at_level(logging.INFO, logger="app.engine.service"):
        result = await service.handle(q)
    assert result["tier"] == "llm" and len(fake.calls) == 1
    assert result["response"]["needsClarification"] and result["mcpCall"] is None
    assert result["failureAttribution"]["requestIR"] == "rejected"
    rec = next(r for r in caplog.records if r.getMessage() == "direct answer not possible; asking language model")
    assert rec.reason == "not_understood"


async def test_direct_answers_can_be_switched_off():
    q = "How many residents are active?"
    fake = FakeTranslator({q: {"action": "count", "conditions": [{"field": "status", "include": ["active"]}]}})
    settings = PinnedDateSettings(
        _env_file=None,
        environment="test",
        topic_gate_enabled=False,
        semantic_router_enabled=False,
        direct_answers_enabled=False,
    )
    router = SemanticIntentRouter(router=FakeAurelioRouter({"count": 0.95, "list": 0.7}))
    service = QueryService(MemoryResidentRepository(), fake, settings, intent_router=router)
    await service.refresh_vocabulary()
    assert (await service.handle(q))["tier"] == "llm"


# ---------------------------------------------------------------- accuracy with the real router
@pytest.mark.skipif(not MODELS_DIR.exists(), reason="embedding model not downloaded")
async def test_no_wrong_direct_answers_on_labeled_questions(index):
    router = SemanticIntentRouter(cache_dir=str(MODELS_DIR))
    router.load()
    builder = DirectQueryBuilder(router.config)
    repo = MemoryResidentRepository()
    wrong, direct = [], 0
    for question, expected in CASES:
        mask = index.mask(question)
        result = builder.build(question, mask, router.score(grammar_skeleton(mask.text, PIN)), index, PIN)
        if result.plan is None:
            continue
        direct += 1
        got = resolve_plan(result.plan, index, mask, question)
        if expected is None or got.problem:
            wrong.append(question)
            continue
        a, b = await execute(got.plan, repo), await execute(QueryPlan.model_validate(expected), repo)
        if (a.matched, a.aggregate_value, [r["residentId"] for r in a.rows][:5]) != (
            b.matched,
            b.aggregate_value,
            [r["residentId"] for r in b.rows][:5],
        ):
            wrong.append(question)
    assert wrong == [], f"wrong direct answers: {wrong}"
    assert direct >= 8, f"only {direct} questions answered directly"


# ---------------------------------------------------------------- real MongoDB MCP server (opt-in)
MCP_URI = os.environ.get("MCP_TEST_MONGODB_URI")


@pytest.mark.skipif(
    not (MCP_URI and shutil.which("mongodb-mcp-server")), reason="set MCP_TEST_MONGODB_URI to a seeded MongoDB"
)
async def test_mcp_repository_matches_reference():
    from app.data.mcp_repository import McpResidentRepository

    mcp = McpResidentRepository("mongodb-mcp-server", MCP_URI, "care_app", "residents")
    await mcp.init()
    try:
        memory = MemoryResidentRepository()
        for p in [
            {
                "action": "count",
                "filters": {
                    "status": {"include": ["active"]},
                    "primary_caregiver": {"include": ["Care Aide Elena Rostova"]},
                },
            },
            {"action": "breakdown", "group_by": ["status", "care_type"]},
            {"action": "aggregate", "aggregate": {"op": "avg", "field": "age"}, "group_by": ["wing"]},
            {"action": "list", "sort": {"field": "age", "direction": "desc"}, "limit": 3},
            {"action": "lookup", "lookup": {"residents": ["RES-1001"], "attributes": []}},
        ]:
            plan = QueryPlan.model_validate(p)
            a, b = await execute(plan, mcp), await execute(plan, memory)
            assert a.matched == b.matched and a.aggregate_value == b.aggregate_value
            assert sorted((g.key, g.count) for g in a.groups or []) == sorted((g.key, g.count) for g in b.groups or [])
            assert sorted((g.key, g.count, g.value, g.n_values) for g in a.group_values or []) == sorted(
                (g.key, g.count, g.value, g.n_values) for g in b.group_values or []
            )
            assert [r["residentId"] for r in a.rows] == [r["residentId"] for r in b.rows]
    finally:
        await mcp.close()


async def test_surname_first_question_is_answered_without_the_model(builder, index):
    r = build(builder, index, "What is the admission date of Vance Eleanor?", "lookup")
    assert r.plan == {"action": "lookup", "lookup": {"residents": ["RES-1001"], "attributes": ["admission_date"]}}


@pytest.mark.parametrize(
    "question",
    ["What is the resident Id of Eleanor Vance?", "What is the resident Id of Eleanor Vance V?", "Eleanor Vance record id"],
)
async def test_resident_id_is_answered_without_the_model(builder, index, question):
    # Regression: "resident ID" was not a lookup attribute and a trailing initial ("V") was an unknown word,
    # so the question went to the model, which answered "not available".
    r = build(builder, index, question, "lookup")
    assert r.plan == {"action": "lookup", "lookup": {"residents": ["RES-1001"], "attributes": ["resident_id"]}}, r
