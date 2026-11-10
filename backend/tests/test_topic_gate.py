"""Topic gate logic and service integration, using a fake embedder (fast, no model download)."""

import numpy as np

from app.config import Settings
from app.data.memory_repository import MemoryResidentRepository
from app.engine.service import QueryService
from app.nlu.topic_examples import OFF_TOPIC, RESIDENT
from app.nlu.topic_gate import TopicGate
from tests.conftest import FakeTranslator, PinnedDateSettings


class FakeEmbedder:
    """2-D 'meaning': x = resident-ness, y = off-topic-ness."""

    def embed(self, texts):
        out = []
        for t in texts:
            low = t.lower()
            if t in RESIDENT:
                out.append([1.0, 0.0])
            elif t in OFF_TOPIC:
                out.append([0.0, 1.0])
            elif "weather" in low or "joke" in low:
                out.append([0.1, 1.0])  # clearly off-topic
            elif "music" in low:
                out.append([0.70, 0.71])  # borderline: must NOT be blocked
            else:
                out.append([1.0, 0.1])  # resident-like
        return np.array(out, dtype=np.float32)


def ready_gate(margin=0.02) -> TopicGate:
    gate = TopicGate(margin, embedder_factory=FakeEmbedder)
    gate.load()
    return gate


def test_blocks_clearly_off_topic():
    assert ready_gate().check("What's the weather in Pune?").blocked


def test_passes_resident_question():
    assert not ready_gate().check("How many residents are active?").blocked


def test_borderline_goes_to_model():
    assert not ready_gate().check("play some music").blocked


async def test_fails_open_when_model_cannot_load():
    def broken():
        raise RuntimeError("no model")

    gate = TopicGate(0.02, embedder_factory=broken)
    await gate.start()
    assert not gate.ready
    assert not gate.check("What's the weather?").blocked


async def make_service(translator, gate, **settings_overrides) -> QueryService:
    settings = PinnedDateSettings(_env_file=None, environment="test", topic_gate_enabled=False, **settings_overrides)
    service = QueryService(MemoryResidentRepository(), translator, settings, gate=gate)
    await service.refresh_vocabulary()
    return service


async def test_off_topic_never_reaches_the_model():
    fake = FakeTranslator({})
    service = await make_service(fake, ready_gate())
    result = await service.handle("Tell me a joke about cats")
    assert result["tier"] == "gate" and result["tokensConsumed"] == 0
    assert result["response"]["isPolicyViolation"]
    assert fake.calls == []
    assert service.stats.by_tier["gate"] == 1


async def test_off_topic_is_handled_even_without_a_model():
    service = await make_service(None, ready_gate())
    assert (await service.handle("What's the weather today?"))["tier"] == "gate"


async def test_questions_naming_a_resident_skip_the_gate():
    class AlwaysBlock(FakeEmbedder):
        def embed(self, texts):
            return np.array([[0.0, 1.0] if t not in RESIDENT else [1.0, 0.0] for t in texts], dtype=np.float32)

    gate = TopicGate(0.02, embedder_factory=AlwaysBlock)
    gate.load()
    fake = FakeTranslator(
        {
            "What's the weather like in resident_1's room?": {"action": "unsupported", "message": "Not available."},
            "weather for RES-1001": {"action": "unsupported", "message": "Not available."},
        }
    )
    service = await make_service(fake, gate)
    assert (await service.handle("What's the weather like in Eleanor Vance's room?"))["tier"] == "llm"
    assert (await service.handle("weather for RES-1001"))["tier"] == "llm"


async def test_daily_model_call_limit():
    fake = FakeTranslator(
        {
            "How many residents are active?": {"action": "count", "filters": {"status": {"include": ["active"]}}},
            "How many residents are inactive?": {"action": "count", "filters": {"status": {"include": ["inactive"]}}},
        }
    )
    service = await make_service(fake, None, llm_daily_call_limit=1)
    assert (await service.handle("How many residents are active?"))["tier"] == "llm"
    limited = await service.handle("How many residents are inactive?")
    assert limited["tier"] == "unavailable" and "daily question limit" in limited["response"]["summaryText"]
    # cached answers still work after the limit is reached
    assert (await service.handle("How many residents are active?"))["tier"] == "cache"


def test_gate_settings_defaults():
    s = Settings(_env_file=None)
    assert s.topic_gate_enabled and s.topic_gate_model == "BAAI/bge-small-en-v1.5" and s.topic_gate_margin == 0.02
    assert s.llm_daily_call_limit == 0 and s.rate_limit_per_day == 2000
