"""Startup warm-up that is allowed to finish, fast "still starting" replies while it runs, and total-time logging."""

import asyncio
import json
import logging

import httpx
import pytest

from app.config import Settings
from app.nlu.qwen_translator import WARM_UP_ATTEMPTS, QwenTranslator
from app.nlu.translator import SchemaContext, TranslationRequest, TranslatorUnavailable
from tests.conftest import FakeTranslator

CTX = SchemaContext(["active"], ["memory_care"], ["North Wing"])
OK = {"choices": [{"message": {"content": json.dumps({"action": "count"})}, "finish_reason": "stop"}],
      "usage": {"prompt_tokens": 10, "completion_tokens": 5}}  # fmt: skip


def translator(handler, **kw) -> QwenTranslator:
    s = Settings(_env_file=None, environment="test", **kw)
    return QwenTranslator(s, client=httpx.AsyncClient(base_url=s.llm_server_url, transport=httpx.MockTransport(handler)))


async def test_warm_up_uses_its_own_long_limit_and_retries():
    seen_timeouts, attempts = [], []

    def handler(req):
        attempts.append(1)
        seen_timeouts.append(req.extensions["timeout"]["read"])
        if len(attempts) < 3:  # the server is still loading: first two attempts fail
            raise httpx.ReadTimeout("cold")
        return httpx.Response(200, json=OK)

    t = translator(handler, llm_max_concurrency=1, llm_warmup_timeout_seconds=600)
    await t.warm_up(CTX, "2026-11-09")
    assert t.warm is True and t.warming is False and len(attempts) == 3
    assert all(x > 60 for x in seen_timeouts)  # never the 60 s per-question limit


async def test_failed_warm_up_gives_up_after_the_attempts_and_never_raises():
    calls = []

    def down(req):
        calls.append(1)
        raise httpx.ConnectError("refused")

    t = translator(down)
    await t.warm_up(CTX, "2026-11-09")
    assert t.warm is False and t.warming is False and len(calls) == WARM_UP_ATTEMPTS


async def test_questions_during_warm_up_are_declined_at_once():
    release = asyncio.Event()

    async def slow(req):
        await release.wait()
        return httpx.Response(200, json=OK)

    s = Settings(_env_file=None, environment="test", llm_max_concurrency=2)
    t = QwenTranslator(s, client=httpx.AsyncClient(base_url=s.llm_server_url, transport=httpx.MockTransport(slow)))
    task = asyncio.create_task(t.warm_up(CTX, "2026-11-09"))
    await asyncio.sleep(0.05)
    with pytest.raises(TranslatorUnavailable, match="warming_up"):
        await t.translate(TranslationRequest("how many residents", "2026-11-09", CTX))
    release.set()
    await task
    assert t.warm
    await t.translate(TranslationRequest("how many residents", "2026-11-09", CTX))  # works once warm


class WarmingTranslator(FakeTranslator):
    async def translate(self, request):
        raise TranslatorUnavailable("warming_up")


async def test_warming_up_reply_is_clear(make_service):
    service = await make_service(WarmingTranslator())
    r = await service.handle("How many residents are active?")
    assert r["tier"] == "unavailable" and "still starting up" in r["response"]["summaryText"]


async def test_every_question_logs_its_total_time(make_service, caplog):
    plan = {"action": "count", "filters": {"status": {"include": ["active"]}}}
    service = await make_service(FakeTranslator({"how many residents are active?": plan}))
    with caplog.at_level(logging.INFO, logger="app.engine.service"):
        r = await service.handle("How many residents are active?")
    done = [x for x in caplog.records if x.getMessage() == "query completed"]
    assert len(done) == 1 and done[0].query_id == r["queryId"]
    assert isinstance(done[0].total_ms, int) and done[0].tier == "llm" and done[0].outcome == "answered"
    assert "active" not in json.dumps({k: str(v) for k, v in done[0].__dict__.items() if k not in ("msg", "message")})


async def test_watchdog_rewarms_after_a_llama_server_restart():
    counters = iter([50, 60, 3, 4, 5, 6, 7, 8])  # 60 → 3: the server restarted (its counters reset)
    completions = []

    def handler(req):
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if req.url.path == "/metrics":
            return httpx.Response(200, text=f"llamacpp:n_decode_total {next(counters, 9)}\n")
        completions.append(1)
        return httpx.Response(200, json=OK)

    t = translator(handler, llm_max_concurrency=1)
    t.warm = True  # warmed against the OLD server
    task = asyncio.create_task(t.keep_warm(lambda: CTX, lambda: "2026-11-09", interval_s=0.01))
    for _ in range(200):
        await asyncio.sleep(0.01)
        if completions:
            break
    task.cancel()
    assert completions and t.warm  # the reset was noticed and the new server was warmed
