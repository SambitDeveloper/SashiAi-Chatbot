"""Query translator on AWS Bedrock: Gemma 4 31B through the bedrock-mantle OpenAI-compatible endpoint.

Selected with LLM_PROVIDER=bedrock (app/main.py). The local Qwen client (qwen_translator.py) is untouched and is used
only with LLM_PROVIDER=local; this client NEVER falls back to it.

API facts, checked against the AWS Bedrock user guide on 2026-11-19 (model card "Gemma 4 31B", "Chat Completions
API", "Get validated JSON results from models"):
  • Gemma 4 is served only on bedrock-mantle, under the /openai/v1 prefix:
    POST https://bedrock-mantle.{region}.api.aws/openai/v1/chat/completions (model "google.gemma-4-31b").
    /v1/chat/completions rejects it ("model ... isn't supported on this route"); bedrock-runtime, Converse and
    InvokeModel are not supported for this model. In-Region only (no geo/global inference IDs); us-west-2 is listed.
  • Authentication: an Amazon Bedrock API key as `Authorization: Bearer <key>`. The key is a SecretStr in Settings,
    put only in that header, and never logged or placed in an exception message.
  • Structured outputs are not in the model card's feature list, and Bedrock's JSON-Schema subset rejects parts of the
    RequestIR schema (numeric bounds, more than 12 optional properties, objects without additionalProperties: false).
    So by default no response_format is sent: the system prompt already ends "Output only the RequestIR JSON object",
    the reply is parsed here (a ```json fence or surrounding text is tolerated), and the SAME RequestIR validation,
    grounding check and retry as for Qwen decide whether it is used. BEDROCK_RESPONSE_FORMAT opts into json_object /
    json_schema for experiments; a Bedrock 400 for an unsupported schema surfaces as "api_error".
  • The first request with a new json_schema can take up to ~90 s (grammar compilation); text mode has no such cost.
  • Reasoning: honored on Chat Completions but not returned. BEDROCK_REASONING_EFFORT sets it; unset = model default.
  • Prompt caching on Bedrock is implicit; llama.cpp-only fields (top_k, seed, cache_prompt, chat_template_kwargs)
    are not sent.
  • Throughput on bedrock-mantle ramps up over time, so throttling (HTTP 429) can happen within quota.

Each call has ONE deadline (slot wait + request), as in the local client. Every failure raises TranslatorUnavailable:
"overloaded" (no slot / HTTP 429), "timeout", "connection", "auth" (401/403), "unavailable" (5xx), "api_error" (other
4xx), "max_tokens", "invalid_json". The service answers those with its normal controlled message.
"""

import asyncio
import json
import logging
import time

import httpx

from app.config import Settings
from app.nlu.prompt import build_messages, build_output_schema, build_user_message
from app.nlu.translator import ModelUsage, SchemaContext, Translation, TranslationRequest, TranslatorUnavailable

log = logging.getLogger(__name__)

MIN_REQUEST_SECONDS = 1.0  # a call left with less than this after waiting for a slot is not sent
HEALTH_TTL_SECONDS = 30.0  # /ready is polled; do not list models on every poll


class BedrockTranslator:
    provider = "aws-bedrock-mantle"
    external = True  # requests leave this server (counted as externalLlmCalls)

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        if settings.bedrock_api_key is None:
            raise ValueError("BedrockTranslator needs BEDROCK_API_KEY")
        self._s = settings
        self.model_name = settings.bedrock_model_id
        self.endpoint = settings.bedrock_endpoint
        base = self.endpoint[: -len("/openai/v1")] if self.endpoint.endswith("/openai/v1") else self.endpoint
        self.models_url = f"{base.removesuffix('/v1')}/v1/models"
        self._client = client or httpx.AsyncClient(
            base_url=self.endpoint,
            timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=5.0),
        )
        self._headers = {"Authorization": f"Bearer {settings.bedrock_api_key.get_secret_value()}"}
        self._slots = asyncio.Semaphore(settings.llm_max_concurrency)
        self._schema_cache: tuple[tuple, dict] | None = None
        self._health: tuple[float, bool] | None = None
        self.warm = True  # nothing to pre-load: no local prompt cache to fill

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        """True when bedrock-mantle is reachable and accepts the key (cached for 30 s).

        Uses the documented OpenAI-compatible Models API (GET .../v1/models; documented under /v1, not /openai/v1). It
        proves endpoint + key only, not that this model answers: AWS does not document whether /openai/v1 models
        appear in that listing, so the model id is not required to be in it.
        """
        now = time.monotonic()
        if self._health and now - self._health[0] < HEALTH_TTL_SECONDS:
            return self._health[1]
        ok = False
        try:
            r = await self._client.get(self.models_url, headers=self._headers, timeout=5.0)
            ok = r.status_code == 200
            if not ok:
                log.warning("Bedrock health check: HTTP %s", r.status_code)
        except httpx.HTTPError as exc:
            log.warning("Bedrock health check failed: %s", type(exc).__name__)
        self._health = (now, ok)
        return ok

    def _schema(self, ctx: SchemaContext) -> dict:
        key = (tuple(ctx.statuses), tuple(ctx.care_types), tuple(ctx.wings))
        if self._schema_cache is None or self._schema_cache[0] != key:
            self._schema_cache = (key, build_output_schema(ctx))
        return self._schema_cache[1]

    def _body(self, messages: list[dict], max_tokens: int) -> dict:
        s = self._s
        body: dict = {
            "model": self.model_name,
            "messages": messages,
            "temperature": s.llm_temperature,
            "top_p": s.llm_top_p,
            "max_tokens": max_tokens,
        }
        if s.bedrock_reasoning_effort:
            body["reasoning_effort"] = s.bedrock_reasoning_effort
        return body

    async def _post(self, body: dict, timeout: float) -> tuple[dict, int]:
        """One call under ONE deadline (slot wait + request). Returns the parsed 200 body and inference ms."""
        deadline = time.perf_counter() + timeout
        try:
            await asyncio.wait_for(self._slots.acquire(), timeout=timeout)
        except TimeoutError as exc:
            log.warning("Bedrock model busy: no free request slot within %ss", timeout)
            raise TranslatorUnavailable("overloaded") from exc
        started = time.perf_counter()
        remaining = deadline - started
        try:
            if remaining < min(MIN_REQUEST_SECONDS, timeout / 2):
                raise TranslatorUnavailable("overloaded")
            response = await self._client.post(
                "/chat/completions", json=body, headers=self._headers, timeout=httpx.Timeout(remaining, connect=5.0)
            )
        except httpx.TimeoutException as exc:
            log.warning("Bedrock model timed out after %.1fs (deadline %ss)", time.perf_counter() - started, timeout)
            raise TranslatorUnavailable("timeout") from exc
        except httpx.HTTPError as exc:
            log.warning("Bedrock endpoint unreachable at %s: %s", self.endpoint, type(exc).__name__)
            raise TranslatorUnavailable("connection") from exc
        finally:
            self._slots.release()
        elapsed = round((time.perf_counter() - started) * 1000)
        if response.status_code != 200:
            raise TranslatorUnavailable(_http_reason(response))
        try:
            data = response.json()
        except ValueError as exc:
            raise TranslatorUnavailable("api_error") from exc
        return data, elapsed

    async def translate(self, request: TranslationRequest, timeout: float | None = None) -> Translation:
        s = self._s
        timeout = timeout or s.llm_timeout_seconds
        if request.timeout_s is not None:
            timeout = min(timeout, max(request.timeout_s, 0.001))
        ctx = request.context
        user = build_user_message(
            request.masked_question,
            request.today,
            ctx.wings,
            request.previous_ir_json,
            request.date_hints,
            request.hint,
            request.brief,
            request.retry_ir_json,
        )
        # no output grammar here, so the prompt spells out every allowed operator per field
        body = self._body(build_messages(ctx, user, operator_table=True), s.llm_max_tokens)
        if s.bedrock_response_format == "json_object":
            body["response_format"] = {"type": "json_object"}
        elif s.bedrock_response_format == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "request_ir", "schema": self._schema(ctx)},
            }
        data, elapsed = await self._post(body, timeout)

        u = data.get("usage") or {}
        usage = ModelUsage(1, u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0, elapsed)
        choice = (data.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "length":
            log.warning("Bedrock model hit max_tokens before finishing RequestIR")
            raise TranslatorUnavailable("max_tokens")
        request_ir = _json_object((choice.get("message") or {}).get("content") or "")
        if request_ir is None:
            log.warning("Bedrock model returned no parsable RequestIR JSON object")
            raise TranslatorUnavailable("invalid_json")
        details = u.get("prompt_tokens_details") or {}
        log.info(
            "Bedrock model RequestIR",
            extra={
                "inference_ms": elapsed,
                "prompt_tokens": usage.prompt_tokens,
                "cached_tokens": details.get("cached_tokens"),
                "completion_tokens": usage.completion_tokens,
            },
        )
        return Translation(request_ir=request_ir, usage=usage, model=self.model_name)

    async def complete_text(self, messages: list[dict], max_tokens: int, timeout: float) -> str:
        """Free-text completion for answer phrasing (engine/phraser.py). Raises TranslatorUnavailable on any failure."""
        data, _ms = await self._post(self._body(messages, max_tokens), timeout)
        choice = (data.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "length":
            raise TranslatorUnavailable("max_tokens")
        return (choice.get("message") or {}).get("content") or ""


def _http_reason(response: httpx.Response) -> str:
    """Map an error status to a failure reason. The body is logged truncated; it never contains the request key."""
    code = response.status_code
    log.warning("Bedrock HTTP %s: %s", code, response.text[:200])
    if code in (401, 403):
        return "auth"
    if code == 429:
        return "overloaded"
    if code >= 500:
        return "unavailable"
    return "api_error"


def _json_object(text: str) -> dict | None:
    """The RequestIR object from a text reply: plain JSON, a ```json fenced block, or the outermost {...} span."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0].strip()
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None
