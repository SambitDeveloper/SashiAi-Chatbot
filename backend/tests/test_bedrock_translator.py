"""Bedrock (Gemma 4 on bedrock-mantle) translator: request shape, auth, parsing, failure handling, provider selection.

bedrock-mantle is replaced by an httpx MockTransport, so these tests never call AWS and need no real key.
"""

import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.main import build_translator
from app.nlu.bedrock_translator import BedrockTranslator
from app.nlu.qwen_translator import QwenTranslator
from app.nlu.translator import SchemaContext, TranslationRequest, TranslatorUnavailable
from tests.test_qwen_translator import REQUEST_IR

CTX = SchemaContext(["active", "discharged"], ["memory_care"], ["North Wing"], ["Nurse Sarah Jenkins, RN"])
KEY = "test-bedrock-key-not-real-0123456789"
BASE = "https://bedrock-mantle.us-west-2.api.aws/openai/v1"


def settings(**kw) -> Settings:
    kw.setdefault("llm_provider", "bedrock")
    kw.setdefault("bedrock_api_key", KEY)
    return Settings(_env_file=None, environment="test", **kw)


def ok_response(content: str, finish: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "google.gemma-4-31b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1800, "completion_tokens": 60, "total_tokens": 1860},
        },
    )


def translator(handler, **kw) -> BedrockTranslator:
    s = settings(**kw)
    client = httpx.AsyncClient(base_url=s.bedrock_endpoint, transport=httpx.MockTransport(handler))
    return BedrockTranslator(s, client=client)


def request(**kw) -> TranslationRequest:
    return TranslationRequest("how many active residents", "2026-11-19", CTX, **kw)


# ---------------------------------------------------------------- endpoint, auth, request body


def test_defaults_point_at_gemma_on_mantle_openai_prefix():
    s = settings()
    assert s.bedrock_model_id == "google.gemma-4-31b" and s.bedrock_region == "us-west-2"
    assert s.bedrock_endpoint == BASE  # /openai/v1, not /v1 (AWS: Gemma 4 is rejected on /v1)
    assert s.model_version_tag == "bedrock:us-west-2:google.gemma-4-31b"


def test_local_model_version_tag_is_unchanged():
    s = Settings(_env_file=None, environment="test")
    assert s.llm_provider == "local"
    assert s.model_version_tag == f"{s.llm_model_name}:{json.dumps(s.llm_chat_template_kwargs, sort_keys=True)}"


async def test_request_goes_to_mantle_chat_completions_with_bearer_key():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["auth"] = req.headers.get("authorization")
        seen["body"] = json.loads(req.content)
        return ok_response(json.dumps(REQUEST_IR))

    t = translator(handler)
    result = await t.translate(request())
    assert seen["url"] == f"{BASE}/chat/completions"
    assert seen["auth"] == f"Bearer {KEY}"
    body = seen["body"]
    assert body["model"] == "google.gemma-4-31b"
    assert body["temperature"] == 0.0 and body["top_p"] == 1.0 and body["max_tokens"] == 512
    # llama.cpp-only fields are never sent to Bedrock; no response_format by default (structured outputs not listed)
    for key in ("top_k", "seed", "cache_prompt", "chat_template_kwargs", "response_format", "reasoning_effort"):
        assert key not in body
    assert body["messages"][0]["role"] == "system" and body["messages"][-1]["role"] == "user"
    assert "Question: how many active residents" in body["messages"][-1]["content"]
    assert result.request_ir == REQUEST_IR and result.model == "google.gemma-4-31b"
    assert result.usage.calls == 1 and result.usage.prompt_tokens == 1800 and result.usage.completion_tokens == 60


async def test_same_prompt_as_the_local_model_plus_the_operator_table():
    """Same few-shot examples and request as the local Qwen client. The ONLY difference: Bedrock has no output grammar,
    so its system prompt spells out every allowed operator per field (generated from request_ir.operators_for)."""
    bodies = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(REQUEST_IR)}, "finish_reason": "stop"}]}
        )

    local = Settings(_env_file=None, environment="test")
    qwen = QwenTranslator(local, httpx.AsyncClient(base_url=local.llm_server_url, transport=httpx.MockTransport(handler)))
    await qwen.translate(request())
    await translator(handler).translate(request())
    local_msgs, bedrock_msgs = bodies[0]["messages"], bodies[1]["messages"]
    assert local_msgs[1:] == bedrock_msgs[1:]  # examples + the question: identical
    local_sys, bedrock_sys = local_msgs[0]["content"], bedrock_msgs[0]["content"]
    assert "Predicate operators — use ONLY these, per field:" in bedrock_sys
    assert "room_number: eq | ne | in | not_in | missing | present | starts_with | contains" in bedrock_sys
    assert "age, admission_date, created_date, discharge_date: gt | gte | lt | lte | between | missing | present" in bedrock_sys
    assert "Predicate operators" not in local_sys  # Qwen's grammar enforces the operator enum; its prompt stays small
    assert "starts_with/contains: only facility, primary_caregiver, room_number" in local_sys


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("json_object", {"type": "json_object"}), ("json_schema", "json_schema")],
)
async def test_response_format_is_opt_in(mode, expected):
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return ok_response(json.dumps(REQUEST_IR))

    await translator(handler, bedrock_response_format=mode, bedrock_reasoning_effort="low").translate(request())
    if expected == "json_schema":
        assert seen["response_format"]["type"] == "json_schema"
        assert seen["response_format"]["json_schema"]["name"] == "request_ir"
    else:
        assert seen["response_format"] == expected
    assert seen["reasoning_effort"] == "low"


def test_key_from_aws_standard_variable(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", KEY)
    s = Settings(_env_file=None, environment="test")
    assert s.bedrock_api_key.get_secret_value() == KEY


def test_key_is_never_shown_in_settings_repr():
    s = settings()
    assert KEY not in repr(s) and KEY not in str(s.model_dump())


def test_bedrock_without_key_fails_at_startup():
    with pytest.raises(ValueError, match="BEDROCK_API_KEY"):
        Settings(_env_file=None, environment="test", llm_provider="bedrock")


# ---------------------------------------------------------------- reply parsing


@pytest.mark.parametrize(
    "content",
    [
        json.dumps(REQUEST_IR),
        "```json\n" + json.dumps(REQUEST_IR, indent=2) + "\n```",
        "Here is the RequestIR:\n" + json.dumps(REQUEST_IR),
    ],
)
async def test_json_reply_is_parsed_with_or_without_a_fence(content):
    result = await translator(lambda req: ok_response(content)).translate(request())
    assert result.request_ir == REQUEST_IR


# ---------------------------------------------------------------- failures → TranslatorUnavailable (never Qwen)


@pytest.mark.parametrize(
    ("response", "kind"),
    [
        (httpx.Response(401, json={"message": "invalid key"}), "auth"),
        (httpx.Response(403, json={"message": "denied"}), "auth"),
        (httpx.Response(429, json={"message": "throttled"}), "overloaded"),
        (httpx.Response(500, text="boom"), "unavailable"),
        (httpx.Response(503, text="busy"), "unavailable"),
        (httpx.Response(400, json={"error": "model isn't supported on this route"}), "api_error"),
        (ok_response("not json at all"), "invalid_json"),
        (ok_response("[1, 2]"), "invalid_json"),
        (ok_response('{"action": "count"', finish="length"), "max_tokens"),
    ],
)
async def test_bad_responses_raise_unavailable(response, kind):
    with pytest.raises(TranslatorUnavailable, match=kind):
        await translator(lambda req: response).translate(request())


async def test_unreachable_endpoint_and_timeout():
    def unreachable(req):
        raise httpx.ConnectError("no route")

    def slow(req):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(TranslatorUnavailable, match="connection"):
        await translator(unreachable).translate(request())
    with pytest.raises(TranslatorUnavailable, match="timeout"):
        await translator(slow).translate(request())


async def test_error_body_and_logs_never_contain_the_key(caplog):
    def handler(req):
        return httpx.Response(401, json={"message": "bad token"})

    with pytest.raises(TranslatorUnavailable) as exc:
        await translator(handler).translate(request())
    assert KEY not in str(exc.value) and KEY not in caplog.text


async def test_concurrency_is_capped_and_overflow_fails_fast():
    release = asyncio.Event()

    async def handler(req):
        await release.wait()
        return ok_response(json.dumps(REQUEST_IR))

    t = translator(handler, llm_max_concurrency=1)
    first = asyncio.create_task(t.translate(request()))
    await asyncio.sleep(0.05)
    with pytest.raises(TranslatorUnavailable, match="overloaded"):
        await t.translate(request(timeout_s=0.2))
    release.set()
    assert (await first).request_ir == REQUEST_IR


async def test_remaining_request_budget_caps_the_call():
    """A retry only gets what is left of the question's budget (MockTransport does not enforce timeouts, so the
    deadline handed to httpx is checked instead)."""
    seen = {}

    def handler(req):
        seen.update(req.extensions["timeout"])
        return ok_response(json.dumps(REQUEST_IR))

    await translator(handler, llm_timeout_seconds=120).translate(request(timeout_s=7.0))
    assert 0 < seen["read"] <= 7.0


async def test_complete_text_for_phrasing():
    t = translator(lambda req: ok_response("There are 9 active residents."))
    assert await t.complete_text([{"role": "user", "content": "x"}], max_tokens=64, timeout=5) == (
        "There are 9 active residents."
    )


async def test_health_uses_documented_models_api_and_caches():
    calls = []

    def handler(req):
        calls.append((str(req.url), req.headers.get("authorization")))
        return httpx.Response(200, json={"object": "list", "data": [{"id": "openai.gpt-oss-120b"}]})

    t = translator(handler)
    assert await t.health() is True and await t.health() is True
    assert calls == [("https://bedrock-mantle.us-west-2.api.aws/v1/models", f"Bearer {KEY}")]  # cached: one call


async def test_health_false_on_bad_key():
    assert await translator(lambda req: httpx.Response(401)).health() is False


# ---------------------------------------------------------------- provider selection and no fallback


def test_provider_selection():
    bedrock = build_translator(settings())
    local = build_translator(Settings(_env_file=None, environment="test", llm_provider="local"))
    off = build_translator(settings(llm_enabled=False))
    assert isinstance(bedrock, BedrockTranslator) and bedrock.provider == "aws-bedrock-mantle"
    assert isinstance(local, QwenTranslator) and local.provider == "local-llama.cpp"
    assert off is None


async def test_bedrock_failure_never_invokes_qwen(make_service, monkeypatch):
    """A Bedrock outage gives the normal 'unavailable' answer; the local Qwen client is never built or called."""
    qwen_used = []
    monkeypatch.setattr(QwenTranslator, "__init__", lambda *a, **k: qwen_used.append("init"))
    monkeypatch.setattr(QwenTranslator, "translate", lambda *a, **k: qwen_used.append("translate"))
    sent = []

    def handler(req):
        sent.append(str(req.url))
        return httpx.Response(503, text="service unavailable")

    service = await make_service(translator(handler))
    result = await service.handle("How many residents are active?")
    assert result["tier"] == "unavailable" and result["response"]["headline"] == "Assistant unavailable"
    assert sent and all(u.startswith(BASE) for u in sent)
    assert qwen_used == []
    assert result["externalLlmCalls"] == len(sent) and service.stats.external_llm_calls == len(sent)


async def test_bedrock_answer_is_validated_and_counted_as_external(make_service):
    def handler(req):
        return ok_response(json.dumps(REQUEST_IR))

    service = await make_service(translator(handler))
    result = await service.handle("How many residents are active?")
    assert result["tier"] == "llm"
    assert result["model"] == "google.gemma-4-31b"
    assert result["externalLlmCalls"] == result["modelCallsAttempted"] >= 1
