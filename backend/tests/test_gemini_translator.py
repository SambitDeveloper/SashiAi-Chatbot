"""Gemini translator tests against a mocked HTTP transport (no network, no cost)."""

import json

import httpx
import pytest

from app.config import Settings
from app.main import build_translator
from app.nlu.gemini_translator import GeminiTranslator
from app.nlu.prompt import SYSTEM_PROMPT
from app.nlu.translator import ClaudeTranslator, TranslatorUnavailable


def make_translator(handler, **overrides) -> GeminiTranslator:
    settings = Settings(_env_file=None, gemini_api_key="gem-test", gemini_max_retries=0, **overrides)
    return GeminiTranslator(settings, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def gemini_response(text: str | None, finish="STOP", block_reason=None, cached=0):
    body = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": text}] if text is not None else []},
                "finishReason": finish,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 700,
            "candidatesTokenCount": 40,
            "thoughtsTokenCount": 5,
            "cachedContentTokenCount": cached,
            "totalTokenCount": 745,
        },
        "modelVersion": "gemini-3.5-flash-lite",
    }
    if block_reason:
        body = {"promptFeedback": {"blockReason": block_reason}, "usageMetadata": body["usageMetadata"]}
    return body


async def test_request_shape_and_parsed_plan():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["body"] = json.loads(request.content)
        plan = {"action": "count", "filters": {"status": {"include": ["active"]}}}
        return httpx.Response(200, json=gemini_response(json.dumps(plan), cached=600))

    t = make_translator(handler)
    result = await t.translate("How many residents are active?", "2026-11-09", ["North Wing"], None)

    assert seen["url"].endswith("/models/gemini-3.5-flash-lite:generateContent")
    assert seen["key"] == "gem-test"
    body = seen["body"]
    assert body["systemInstruction"]["parts"][0]["text"] == SYSTEM_PROMPT  # static, cacheable prefix
    gen = body["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    assert gen["responseJsonSchema"]["required"] == ["action"]
    thinking = gen["thinkingConfig"]  # the SDK may serialise either casing; Google JSON APIs accept both
    assert (thinking.get("thinkingLevel") or thinking.get("thinking_level")) == "MINIMAL"
    assert "Today: 2026-11-09" in body["contents"][0]["parts"][0]["text"]

    assert result.plan["filters"]["status"]["include"] == ["active"]
    assert result.usage.cache_read_input_tokens == 600
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 45  # answer + thinking tokens


async def test_default_thinking_level_omits_thinking_config():
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=gemini_response('{"action": "help"}'))

    t = make_translator(handler, gemini_thinking_level="default")
    await t.translate("hi", "2026-11-09", [], None)
    assert "thinkingConfig" not in seen["body"]["generationConfig"]


async def test_blocked_prompt_becomes_unsupported():
    t = make_translator(lambda r: httpx.Response(200, json=gemini_response(None, block_reason="SAFETY")))
    assert (await t.translate("q", "2026-11-09", [], None)).plan["action"] == "unsupported"


async def test_safety_finish_becomes_unsupported():
    t = make_translator(lambda r: httpx.Response(200, json=gemini_response(None, finish="SAFETY")))
    assert (await t.translate("q", "2026-11-09", [], None)).plan["action"] == "unsupported"


async def test_truncated_output_is_an_error():
    t = make_translator(lambda r: httpx.Response(200, json=gemini_response('{"action": "co', finish="MAX_TOKENS")))
    with pytest.raises(TranslatorUnavailable):
        await t.translate("q", "2026-11-09", [], None)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 503])
async def test_api_errors_map_to_unavailable(status):
    error = {"error": {"code": status, "message": "nope", "status": "ERROR"}}
    t = make_translator(lambda r: httpx.Response(status, json=error))
    with pytest.raises(TranslatorUnavailable):
        await t.translate("q", "2026-11-09", [], None)


async def test_connection_error_maps_to_unavailable():
    def handler(request):
        raise httpx.ConnectError("down")

    with pytest.raises(TranslatorUnavailable):
        await make_translator(handler).translate("q", "2026-11-09", [], None)


def test_provider_selection(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert isinstance(build_translator(Settings(_env_file=None, gemini_api_key="g")), GeminiTranslator)
    assert isinstance(build_translator(Settings(_env_file=None, anthropic_api_key="a")), ClaudeTranslator)
    both = Settings(_env_file=None, gemini_api_key="g", anthropic_api_key="a", llm_provider="anthropic")
    assert isinstance(build_translator(both), ClaudeTranslator)
    assert build_translator(Settings(_env_file=None)) is None


def test_google_api_key_alias(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "from-google-var")
    assert Settings(_env_file=None).gemini_api_key == "from-google-var"
