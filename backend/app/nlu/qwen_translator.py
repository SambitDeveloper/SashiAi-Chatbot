"""Local query translator: a small Qwen model (GGUF) served by llama.cpp's `llama-server`.

Default: Qwen2.5-0.5B-Instruct Q4_K_M. Under evaluation: Qwen3-0.6B Q4_K_M in non-thinking mode
(LLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'). Same client, prompt, schema and validation for both.

  • Runs on this server's private network only (LLM_SERVER_URL, default http://127.0.0.1:8081); nothing leaves it.
  • The model is loaded once by llama-server and reused; this client never downloads or loads weights.
  • Output is grammar-constrained to the plan JSON Schema (llama.cpp converts it to GBNF), with greedy decoding
    (temperature 0, fixed seed) so the same question gives the same plan.
  • A semaphore caps concurrent generations to the server's parallel slots; a request that cannot get a slot
    within the timeout fails fast with TranslatorUnavailable("overloaded") instead of piling up.
  • Any failure (unreachable, timeout, HTTP error, invalid JSON, truncated output) raises TranslatorUnavailable:
    the service answers with a controlled message. There is no fallback to any external model.
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

WARM_UP_TIMEOUT_S = 120.0


class QwenTranslator:
    provider = "local-llama.cpp"

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self._s = settings
        self.model_name = settings.llm_model_name
        self._client = client or httpx.AsyncClient(
            base_url=settings.llm_server_url,
            timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=2.0),
        )
        self._slots = asyncio.Semaphore(settings.llm_max_concurrency)
        self._schema_cache: tuple[tuple, dict] | None = None

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        """True when llama-server has loaded the model and accepts requests."""
        try:
            r = await self._client.get("/health", timeout=2.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def warm_up(self, ctx: SchemaContext, today: str) -> None:
        """Fill every inference slot's prompt cache with the shared system prompt + examples.

        A cold first request must process the whole ~1,200-token prefix and was measured at >15 s (the default
        timeout) right after llama-server starts; warm requests reuse the cached prefix. The warm-up therefore uses a
        long timeout of its own (llama-server drops a request whose client disconnects, so a timed-out warm-up would
        cache nothing). Failures are only logged.
        """
        started = time.perf_counter()
        request = TranslationRequest("how many residents are there", today, ctx)
        results = await asyncio.gather(
            *(self.translate(request, timeout=WARM_UP_TIMEOUT_S) for _ in range(self._s.llm_max_concurrency)),
            return_exceptions=True,
        )
        ok = sum(not isinstance(r, Exception) for r in results)
        log.info("local model warm-up", extra={"slots_warmed": ok, "ms": round((time.perf_counter() - started) * 1000)})

    def _schema(self, ctx: SchemaContext) -> dict:
        key = (tuple(ctx.statuses), tuple(ctx.care_types), tuple(ctx.wings))
        if self._schema_cache is None or self._schema_cache[0] != key:
            self._schema_cache = (key, build_output_schema(ctx))
        return self._schema_cache[1]

    async def translate(self, request: TranslationRequest, timeout: float | None = None) -> Translation:
        """timeout: overrides LLM_TIMEOUT_SECONDS (used only by warm_up)."""
        s = self._s
        timeout = timeout or s.llm_timeout_seconds
        ctx = request.context
        user = build_user_message(
            request.masked_question,
            request.today,
            ctx.wings,
            request.previous_plan_json,
            request.date_hints,
            request.hint,
        )
        body = {
            "model": self.model_name,
            "messages": build_messages(ctx, user),
            "temperature": s.llm_temperature,
            "top_p": s.llm_top_p,
            "top_k": s.llm_top_k,
            "seed": s.llm_seed,
            "max_tokens": s.llm_max_tokens,
            "cache_prompt": True,  # reuse the KV cache of the static system prompt + examples
            "response_format": {"type": "json_schema", "json_schema": {"name": "query_plan", "schema": self._schema(ctx)}},
        }
        if s.llm_chat_template_kwargs:  # e.g. Qwen3 non-thinking mode
            body["chat_template_kwargs"] = dict(s.llm_chat_template_kwargs)
        try:
            await asyncio.wait_for(self._slots.acquire(), timeout=timeout)
        except TimeoutError as exc:
            log.warning("local model busy: no free inference slot")
            raise TranslatorUnavailable("overloaded") from exc
        started = time.perf_counter()
        try:
            response = await self._client.post(
                "/v1/chat/completions", json=body, timeout=httpx.Timeout(timeout, connect=2.0)
            )
        except httpx.TimeoutException as exc:
            log.warning("local model timed out after %ss", timeout)
            raise TranslatorUnavailable("timeout") from exc
        except httpx.HTTPError as exc:
            log.warning("local model unreachable at %s: %s", s.llm_server_url, type(exc).__name__)
            raise TranslatorUnavailable("connection") from exc
        finally:
            self._slots.release()
        elapsed = round((time.perf_counter() - started) * 1000)

        if response.status_code != 200:
            log.warning("local model HTTP %s: %s", response.status_code, response.text[:200])
            raise TranslatorUnavailable("unavailable" if response.status_code == 503 else "api_error")
        data = response.json()
        u = data.get("usage") or {}
        usage = ModelUsage(1, u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0, elapsed)
        choice = (data.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "length":
            log.warning("local model hit max_tokens before finishing the plan")
            raise TranslatorUnavailable("max_tokens")
        message = choice.get("message") or {}
        # Thinking mode: llama-server (--reasoning-format deepseek) returns the thoughts in `reasoning_content`, and
        # the JSON grammar applies only after </think>. Thoughts are never parsed, returned or logged verbatim; as a
        # second guard, anything up to a stray </think> in `content` is dropped and only the final JSON is parsed.
        reasoning_chars = len(message.get("reasoning_content") or "")
        text = _final_answer(message.get("content") or "")
        try:
            plan = json.loads(text)
        except json.JSONDecodeError as exc:
            log.warning("local model returned invalid JSON")
            raise TranslatorUnavailable("invalid_json") from exc
        if not isinstance(plan, dict):
            raise TranslatorUnavailable("invalid_json")
        timings = data.get("timings") or {}
        log.info(
            "local model plan",
            extra={
                "inference_ms": elapsed,
                "prompt_tokens": usage.prompt_tokens,
                "cached_tokens": timings.get("cache_n"),
                "completion_tokens": usage.completion_tokens,
                "reasoning_chars": reasoning_chars,
            },
        )
        return Translation(plan=plan, usage=usage, model=self.model_name)


_THINK_END = "</think>"


def _final_answer(content: str) -> str:
    """The structured answer only: text after the last </think> (if any), with any <think> block removed."""
    if _THINK_END in content:
        content = content.rsplit(_THINK_END, 1)[1]
    return content.replace("<think>", "").strip()
