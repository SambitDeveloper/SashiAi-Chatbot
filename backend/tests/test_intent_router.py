"""Semantic router: scoring and logging only (routing to the language model is unchanged)."""

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from app.config import Settings
from app.data.memory_repository import MemoryResidentRepository
from app.engine.service import QueryService
from app.nlu.intent_router import DEFAULT_ROUTES_FILE, SemanticIntentRouter
from app.nlu.topic_gate import TopicGate
from tests.conftest import FakeTranslator, PinnedDateSettings
from tests.test_topic_gate import FakeEmbedder

MODELS_DIR = Path(__file__).resolve().parents[1] / ".models"


class FakeAurelioRouter:
    """Mimics IndexedRouter(text, limit=None): one choice per route with a confidence."""

    def __init__(self, scores):
        self.scores = scores
        self.seen: list[str] = []

    def __call__(self, text, limit=None):
        self.seen.append(text)
        return [SimpleNamespace(name=n, confidence=c) for n, c in self.scores.items()]


def test_routes_file_is_valid_and_uses_the_server_model():
    config = yaml.safe_load(DEFAULT_ROUTES_FILE.read_text(encoding="utf-8"))
    names = [r["name"] for r in config["routes"]]
    assert len(names) == len(set(names)) >= 5
    assert all(r["utterances"] for r in config["routes"])
    assert config["encoder"]["name"] == Settings(_env_file=None).topic_gate_model


def test_score_reports_best_route_margin_and_all_scores():
    router = SemanticIntentRouter(router=FakeAurelioRouter({"count": 0.81, "list": 0.74, "help": 0.40}))
    s = router.score("how many residents are active")
    assert (s.route, s.confidence, s.margin) == ("count", 0.81, 0.07)
    assert s.scores == {"count": 0.81, "list": 0.74, "help": 0.40}


def test_not_ready_returns_none():
    assert SemanticIntentRouter().score("anything") is None


async def make_service(fake_router, gate=None):
    settings = PinnedDateSettings(
        _env_file=None,
        environment="test",
        topic_gate_enabled=False,
        semantic_router_enabled=False,
        direct_answers_enabled=False,  # these tests cover scoring + logging only; direct answers: test_direct_query.py
    )
    intent_router = SemanticIntentRouter(router=fake_router) if fake_router else None
    translator = FakeTranslator(
        {
            "Who is the primary caretaker of resident_1 ?": {
                "action": "lookup",
                "lookup": {"residents": ["resident_1"], "attributes": ["primary_caregiver"]},
            },
        }
    )
    service = QueryService(MemoryResidentRepository(), translator, settings, gate=gate, intent_router=intent_router)
    await service.refresh_vocabulary()
    return service, translator


async def test_on_topic_question_is_scored_and_logged_without_names(caplog):
    fake = FakeAurelioRouter({"lookup": 0.86, "list": 0.71})
    service, translator = await make_service(fake)
    with caplog.at_level(logging.INFO, logger="app.engine.service"):
        result = await service.handle("Who is the primary caretaker of Eleanor Vance ?")
    records = [r for r in caplog.records if r.getMessage() == "semantic router score"]
    assert len(records) == 1
    rec = records[0]
    assert rec.route == "lookup" and rec.confidence == 0.86 and rec.margin == 0.15
    assert rec.scores == {"lookup": 0.86, "list": 0.71} and rec.query_id == result["queryId"]
    # the router saw the masked question: no resident name in what was scored or logged
    assert fake.seen == ["Who is the primary caretaker of resident_1 ?"]
    assert "Eleanor" not in str(rec.__dict__)
    # routing unchanged: the question still went to the language model
    assert result["tier"] == "llm" and len(translator.calls) == 1


async def test_off_topic_question_is_not_scored(caplog):
    gate = TopicGate(0.02, embedder_factory=FakeEmbedder)
    gate.load()
    fake = FakeAurelioRouter({"count": 0.8})
    service, _ = await make_service(fake, gate=gate)
    with caplog.at_level(logging.INFO, logger="app.engine.service"):
        result = await service.handle("What's the weather today?")
    assert result["tier"] == "gate" and fake.seen == []
    assert not [r for r in caplog.records if r.getMessage() == "semantic router score"]


async def test_router_still_loading_does_not_block_questions():
    service, translator = await make_service(None)
    service.intent_router = SemanticIntentRouter()  # not loaded yet
    result = await service.handle("Who is the primary caretaker of Eleanor Vance ?")
    assert result["tier"] == "llm"


@pytest.mark.skipif(not MODELS_DIR.exists(), reason="embedding model not downloaded")
def test_real_router_scores_from_the_yaml_file():
    """Clear-cut questions only: closer intents (list vs breakdown) often have small margins, which is what
    the logged margin is for. Improve those by adding example questions to intent_routes.yaml."""
    router = SemanticIntentRouter(cache_dir=str(MODELS_DIR))
    router.load()
    count = router.score("how many residents are in memory care")
    assert count.route == "count" and count.margin > 0.05
    assert router.score("hi").route == "help"
    assert router.score("who is the primary caretaker of resident_1 ?").route == "lookup"
    assert set(count.scores) <= set(router.route_names)
