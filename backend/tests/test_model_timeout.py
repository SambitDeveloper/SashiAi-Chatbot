"""Model-call deadline (slot wait + request share one budget, hard total cap) and model-call telemetry."""

import asyncio
import json
import logging
import time

import httpx
import pytest

from app.config import Settings
from app.nlu.qwen_translator import QwenTranslator
from app.nlu.translator import SchemaContext, TranslationRequest, TranslatorUnavailable
from tests.conftest import FakeTranslator

CTX = SchemaContext(["active", "discharged"], ["memory_care"], ["North Wing"], ["Nurse Sarah Jenkins, RN"])
OK = {
    "choices": [{"message": {"content": '{"action": "count"}'}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
}


def translator(handler, **kw) -> QwenTranslator:
    s = Settings(_env_file=None, environment="test", **kw)
    client = httpx.AsyncClient(base_url=s.llm_server_url, transport=httpx.MockTransport(handler))
    return QwenTranslator(s, client=client)


def request() -> TranslationRequest:
    return TranslationRequest("how many residents", "2026-11-17", CTX)


def test_default_model_timeout_is_120_seconds():  # raised from 60 s on 2026-11-19 (requested)
    assert Settings(_env_file=None).llm_timeout_seconds == 120.0


class SlowServer:
    """A real local HTTP server (asyncio, no dependencies) that answers each request after `delays[i]` seconds and
    records whether the client hung up first, so timeouts and connection cleanup are tested over a real socket."""

    def __init__(self, delays):
        self.delays = list(delays)
        self.disconnected = 0
        self.answered = 0

    async def handle(self, reader, writer):
        head = await reader.readuntil(b"\r\n\r\n")
        length = next((int(x.split(b":")[1]) for x in head.split(b"\r\n") if x.lower().startswith(b"content-length")), 0)
        await reader.readexactly(length)
        delay = self.delays.pop(0) if self.delays else 0
        try:
            if await asyncio.wait_for(reader.read(1), delay) == b"":  # client closed the connection while we worked
                self.disconnected += 1
                writer.close()
                return
        except TimeoutError:
            pass
        body = json.dumps(OK).encode()
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        await writer.drain()
        self.answered += 1
        writer.close()

    async def __aenter__(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        self.url = f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"
        return self

    async def __aexit__(self, *exc):
        self.server.close()


async def test_timed_out_request_closes_the_connection():
    async with SlowServer([10]) as srv:
        t = QwenTranslator(Settings(_env_file=None, environment="test", llm_server_url=srv.url, llm_timeout_seconds=1.0))
        started = time.perf_counter()
        try:
            await t.translate(request())
            raise AssertionError("expected a timeout")
        except TranslatorUnavailable as exc:
            assert str(exc) == "timeout"
        assert time.perf_counter() - started < 1.6
        await asyncio.sleep(0.2)
        assert srv.disconnected == 1 and srv.answered == 0  # the server sees the hang-up (llama-server then cancels)
        await t.close()


async def test_queued_call_gets_only_the_remaining_budget():
    # first call holds the only slot for 0.8 s; the second has 2.5 s in total, so ~1.7 s once it gets the slot,
    # and its own 5 s answer must time out at the shared deadline — not after a fresh 2.5 s
    async with SlowServer([0.8, 5]) as srv:
        s = Settings(_env_file=None, environment="test", llm_server_url=srv.url, llm_timeout_seconds=2.5,
                     llm_max_concurrency=1)  # fmt: skip
        t = QwenTranslator(s)
        started = time.perf_counter()
        results = await asyncio.gather(t.translate(request()), t.translate(request()), return_exceptions=True)
        assert not isinstance(results[0], Exception)
        assert str(results[1]) == "timeout"
        assert time.perf_counter() - started < 2.5 + 0.4
        await t.close()


async def test_slot_wait_and_request_share_one_deadline():
    async def handler(_req):
        await asyncio.sleep(1.0)
        return httpx.Response(200, json=OK)

    t = translator(handler, llm_timeout_seconds=1.6, llm_max_concurrency=1)
    started = time.perf_counter()
    results = await asyncio.gather(t.translate(request()), t.translate(request()), return_exceptions=True)
    elapsed = time.perf_counter() - started
    assert not isinstance(results[0], Exception)
    # the second call waited ~1 s for the slot; only ~0.6 s of its budget was left, so it is not sent at all.
    # Before: it got a fresh full timeout after the wait (≈ 2× the setting in total).
    assert isinstance(results[1], TranslatorUnavailable) and str(results[1]) == "overloaded"
    assert elapsed < 1.6 + 0.3


async def test_cancellation_releases_the_inference_slot_promptly():
    entered = asyncio.Event()

    async def handler(_req):
        entered.set()
        await asyncio.Event().wait()

    t = translator(handler, llm_timeout_seconds=5.0, llm_max_concurrency=1)
    task = asyncio.create_task(t.translate(request()))
    await asyncio.wait_for(entered.wait(), 1.0)
    started = time.perf_counter()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert time.perf_counter() - started < 0.2
    assert t._slots._value == 1
    await t.close()


# ---- telemetry --------------------------------------------------------------------------------------------------

Q = "Tell me the count of active residents whose care taker is Nurse Sarah Jenkins?"
PLAN = {"action": "count", "filters": {"status": {"include": ["active"]}, "primary_caregiver": {"include": ["Sarah Jenkins"]}}}


class TimingOut(FakeTranslator):
    async def translate(self, request):
        self.calls.append({"question": request.masked_question})
        raise TranslatorUnavailable("timeout")


async def test_timed_out_call_is_counted_as_attempted(make_service, caplog):
    service = await make_service(TimingOut())
    with caplog.at_level(logging.INFO, logger="app.engine.service"):
        r = await service.handle(Q)
    assert r["tier"] == "unavailable"
    assert r["modelCalls"] == 0  # completed calls (unchanged meaning)
    assert (r["modelCallsAttempted"], r["modelCallsSucceeded"], r["modelCallsTimedOut"], r["modelCallsFailed"]) == (1, 0, 1, 0)
    assert r["model"] == "fake-local-model"  # a model was asked, even though it did not answer
    finished = next(rec for rec in caplog.records if rec.getMessage().startswith("step-19"))
    assert finished.model_calls_attempted == 1 and finished.model_calls_timed_out == 1
    local = next(rec for rec in caplog.records if rec.getMessage().startswith("step-12"))
    assert local.status == "timeout"
    assert service.stats.model_calls_attempted == 1 and service.stats.model_calls_timed_out == 1


async def test_other_failures_are_counted_as_failed(make_service):
    service = await make_service(FakeTranslator(fail=True))  # "connection"
    r = await service.handle(Q)
    assert (r["modelCallsAttempted"], r["modelCallsTimedOut"], r["modelCallsFailed"]) == (1, 0, 1)


async def test_successful_call_counts(make_service):
    service = await make_service(FakeTranslator({Q: PLAN}))
    r = await service.handle(Q)
    assert r["modelCalls"] == 1
    assert (r["modelCallsAttempted"], r["modelCallsSucceeded"], r["modelCallsTimedOut"]) == (1, 1, 0)


async def test_no_model_needed_counts_nothing(make_service):
    # answered from a stored semantic plan template (the exact plan cache was removed on 2026-11-18)
    from app.engine.semantic_cache import SemanticPlanCache
    from tests.test_semantic_cache import FakeStore, SameVector

    service = await make_service(FakeTranslator({Q: PLAN}))
    store = FakeStore()
    service.semantic_cache = SemanticPlanCache(store, SameVector, threshold=0.0, top_k=5, max_candidates=50,
                                               timeout_s=1.0, embedding_model="t")  # fmt: skip
    await service.handle(Q)
    await service.semantic_cache.drain()
    again = await service.handle(Q)
    assert again["tier"] == "template" and again["modelCallsAttempted"] == 0 and again["model"] is None
