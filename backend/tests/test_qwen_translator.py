"""Local Qwen translator (llama.cpp server client): request shape, constrained output, failure handling.

The llama.cpp server is replaced by an httpx MockTransport, so these tests run without the model.
"""

import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.nlu.qwen_translator import QwenTranslator
from app.nlu.translator import SchemaContext, TranslationRequest, TranslatorUnavailable

CTX = SchemaContext(["active", "discharged"], ["memory_care"], ["North Wing"], ["Nurse Sarah Jenkins, RN"])
REQUEST_IR = {
    "action": "count",
    "action_status": "supported",
    "action_evidence": "How many",
    "where": {
        "kind": "predicate",
        "status": "supported",
        "field": "status",
        "operator": "eq",
        "value": "active",
        "evidence": "active",
    },
    "group_by": [],
    "aggregate": None,
    "projection": [],
    "sort": None,
    "limit": None,
    "lookup": None,
    "message": None,
    "uses_previous": False,
    "unsupported": [],
    "unresolved": [],
    "ambiguous": [],
}


def settings(**kw) -> Settings:
    return Settings(_env_file=None, environment="test", **kw)


def ok_response(content: str, finish: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 812, "completion_tokens": 21},
            "timings": {"cache_n": 790},
        },
    )


def translator(handler, **kw) -> QwenTranslator:
    s = settings(**kw)
    client = httpx.AsyncClient(base_url=s.llm_server_url, transport=httpx.MockTransport(handler))
    return QwenTranslator(s, client=client)


def request(**kw) -> TranslationRequest:
    return TranslationRequest("How many active residents?", "2026-11-09", CTX, **kw)


async def test_request_is_deterministic_and_grammar_constrained():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["body"] = json.loads(req.content)
        return ok_response(json.dumps(REQUEST_IR))

    t = translator(handler)
    result = await t.translate(request(date_hints=["'last month' = 2026-10-01 to 2026-10-31"]))
    body = seen["body"]
    assert seen["path"] == "/v1/chat/completions"
    assert body["temperature"] == 0.0 and body["top_k"] == 1 and body["cache_prompt"] is True
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["action"]["enum"][0] == "count"
    assert "$defs" in schema and "IRNode" in schema["$defs"]
    # the real request is the LAST message; system prompt + few-shot examples come first (cacheable prefix)
    assert body["messages"][0]["role"] == "system"
    last = body["messages"][-1]["content"]
    assert "Question: How many active residents?" in last and "Dates: 'last month' = 2026-10-01" in last
    assert result.request_ir == REQUEST_IR and result.usage.calls == 1 and result.usage.prompt_tokens == 812


async def test_system_prompt_and_examples_are_identical_across_requests():
    """A stable prefix is what lets llama.cpp reuse its KV cache between questions."""
    prefixes = []

    def handler(req):
        prefixes.append(json.loads(req.content)["messages"][:-1])
        return ok_response(json.dumps(REQUEST_IR))

    t = translator(handler)
    await t.translate(request())
    await t.translate(TranslationRequest("list discharged residents", "2026-11-09", CTX))
    assert prefixes[0] == prefixes[1]


@pytest.mark.parametrize(
    ("response", "kind"),
    [
        (ok_response("{not json"), "invalid_json"),
        (ok_response('["a list"]'), "invalid_json"),
        (ok_response('{"action": "count", "condi', finish="length"), "max_tokens"),
        (httpx.Response(503, json={"error": "loading model"}), "unavailable"),
        (httpx.Response(500, text="boom"), "api_error"),
    ],
)
async def test_bad_responses_raise_unavailable(response, kind):
    t = translator(lambda req: response)
    with pytest.raises(TranslatorUnavailable, match=kind):
        await t.translate(request())


async def test_unreachable_server_and_timeout():
    def refuse(req):
        raise httpx.ConnectError("refused")

    def slow(req):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(TranslatorUnavailable, match="connection"):
        await translator(refuse).translate(request())
    with pytest.raises(TranslatorUnavailable, match="timeout"):
        await translator(slow).translate(request())


async def test_concurrency_is_capped_and_overflow_fails_fast():
    release = asyncio.Event()
    active = {"now": 0, "max": 0}

    async def handler(req):
        active["now"] += 1
        active["max"] = max(active["max"], active["now"])
        await release.wait()
        active["now"] -= 1
        return ok_response(json.dumps(REQUEST_IR))

    t = translator(handler, llm_max_concurrency=1, llm_timeout_seconds=0.2)
    first = asyncio.create_task(t.translate(request()))
    await asyncio.sleep(0.05)
    with pytest.raises(TranslatorUnavailable, match="overloaded"):
        await t.translate(request())  # no free slot within the timeout
    release.set()
    assert (await first).request_ir == REQUEST_IR
    assert active["max"] == 1


async def test_health_reflects_server_state():
    assert await translator(lambda r: httpx.Response(200, json={"status": "ok"})).health()
    assert not await translator(lambda r: httpx.Response(503, json={"status": "loading model"})).health()


async def test_chat_template_kwargs_are_sent_only_when_configured():
    """Qwen3 needs {"enable_thinking": false}; Qwen2.5 requests stay unchanged."""
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return ok_response(json.dumps(REQUEST_IR))

    await translator(handler, llm_chat_template_kwargs={}).translate(request())  # e.g. Qwen2.5 configured explicitly
    await translator(handler).translate(request())  # default: Qwen3 non-thinking
    assert "chat_template_kwargs" not in bodies[0]
    assert bodies[1]["chat_template_kwargs"] == {"enable_thinking": False}
    # everything else (prompt, schema, sampling) is identical between the two models
    assert {k: v for k, v in bodies[1].items() if k != "chat_template_kwargs"} == bodies[0]


def test_chat_template_kwargs_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": false}')
    assert Settings(_env_file=None).llm_chat_template_kwargs == {"enable_thinking": False}


def test_default_model_is_qwen3_non_thinking():
    s = Settings(_env_file=None)
    assert s.llm_model_name == "qwen3-4b-instruct-2507-q4_k_m" and s.llm_chat_template_kwargs == {
        "enable_thinking": False
    }


async def test_warm_up_fills_every_slot_and_never_raises():
    calls = []

    def handler(req):
        calls.append(1)
        return ok_response(json.dumps(REQUEST_IR))

    await translator(handler, llm_max_concurrency=2).warm_up(CTX, "2026-11-09")
    assert len(calls) == 2

    def down(req):
        raise httpx.ConnectError("refused")

    await translator(down).warm_up(CTX, "2026-11-09")  # logged, not raised


async def test_thinking_output_is_never_parsed_or_returned():
    """Thoughts arrive in reasoning_content (or, defensively, inline before </think>); only the final JSON counts."""

    def handler(req):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "reasoning_content": 'Okay, the user wants {"action": "list"} maybe...',
                            "content": "<think>stray {not json}</think>\n" + json.dumps(REQUEST_IR),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1170, "completion_tokens": 314},
            },
        )

    result = await translator(handler).translate(request())
    assert result.request_ir == REQUEST_IR and "reasoning" not in json.dumps(result.request_ir)


async def test_sampling_settings_are_sent():
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return ok_response(json.dumps(REQUEST_IR))

    await translator(handler, llm_temperature=0.6, llm_top_p=0.95, llm_top_k=20).translate(request())
    assert (bodies[0]["temperature"], bodies[0]["top_p"], bodies[0]["top_k"]) == (0.6, 0.95, 20)
    await translator(handler).translate(request())  # default stays greedy
    assert (bodies[1]["temperature"], bodies[1]["top_k"]) == (0.0, 1)


async def test_prompt_does_not_grow_with_the_facility_list():
    """Regression: the full facility list was sent with every question (~16 tokens per name); with a real
    database that overflowed the 2,048-token slot and made every new prompt slow. Only relevant wings are sent."""
    from app.data.memory_repository import MemoryResidentRepository
    from app.data.repository import Vocabulary
    from app.engine.service import QueryService
    from app.nlu.vocabulary import EntityIndex

    many = [f"(train) FACILITY_{i}" for i in range(300)]
    service = QueryService(MemoryResidentRepository(), None, settings())
    service.index = EntityIndex(Vocabulary(wings=many, statuses=["active"], care_types=["memory_care"]))
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return ok_response(json.dumps(REQUEST_IR))

    t = translator(handler)
    from datetime import date

    from app.engine.grounding import scan_question

    for q in ("how many residents are active", "how many residents are in (train) FACILITY_42"):
        await t.translate(service._request(q, "2026-11-13", None, scan_question(q, date(2026, 11, 13))))
    first, second = bodies
    # the shared prefix (system prompt + few-shot examples) is identical, so llama.cpp keeps it cached
    assert first["messages"][:-1] == second["messages"][:-1]
    # no facility list unless the question names one; then only that one
    assert "Wings:" not in first["messages"][-1]["content"]
    assert "Wings: (train) FACILITY_42" in second["messages"][-1]["content"]
    assert "FACILITY_7," not in json.dumps(second)
    # the whole request stays small whatever the database size (300 facilities here)
    assert len(json.dumps(first["messages"])) < 7000
