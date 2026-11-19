"""Local query translator: a small Qwen model (GGUF) served by llama.cpp's `llama-server`.

Default deployment: Qwen3-4B-Instruct Q4_K_M in non-thinking mode
(LLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}').

  • Runs on this server's private network only (LLM_SERVER_URL, default http://127.0.0.1:8081); nothing leaves it.
  • The model is loaded once by llama-server and reused; this client never downloads or loads weights.
  • Output is grammar-constrained to the RequestIR JSON Schema (llama.cpp converts it to GBNF), with greedy decoding
    (temperature 0, fixed seed) so the same question gives the same plan.
  • A semaphore caps concurrent generations to the server's parallel slots. Each call has ONE deadline
    (LLM_TIMEOUT_SECONDS) shared by the slot wait and the request: no slot in time → "overloaded"; no answer in
    time → "timeout", and the connection is closed so llama-server stops working on it (see _post).
  • Any failure (unreachable, timeout, HTTP error, invalid JSON, truncated output) raises TranslatorUnavailable:
    the service answers with a controlled message. There is no fallback to any external model.
"""

import asyncio
import json
import logging
import re
import time

import httpx

from app.config import Settings
from app.nlu.prompt import build_messages, build_output_schema, build_user_message
from app.nlu.translator import ModelUsage, SchemaContext, Translation, TranslationRequest, TranslatorUnavailable

log = logging.getLogger(__name__)

MIN_REQUEST_SECONDS = 1.0  # a call left with less than this after waiting for a slot is not sent
WARM_UP_ATTEMPTS = 3


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
        # warm = llama-server has the shared prompt prefix cached; warming = the startup warm-up is still running
        self.warm = False
        self.warming = False

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
        """Get the shared system prompt + examples into llama-server's prompt cache before users need it.

        A cold server must read the whole ~1,800-token prefix once. On the 4B model on this CPU that took ~220-240 s
        (~8 prompt tokens/s, 2026-11-19), longer than both the old 120 s warm-up limit and the 60 s per-question limit.
        llama-server drops a request whose client gives up, so every attempt then started from zero again and EVERY
        question timed out, after every restart or prompt change. Hence: ONE request first, with its own long limit
        (LLM_WARMUP_TIMEOUT_SECONDS), retried; then the remaining slots, which reuse the cached prefix and are quick.
        While this runs, `warming` is set and questions get an immediate "still starting" reply (translate()).
        """
        started = time.perf_counter()
        request = TranslationRequest("how many residents are there", today, ctx)
        limit = self._s.llm_warmup_timeout_seconds
        self.warming = True
        try:
            for attempt in range(1, WARM_UP_ATTEMPTS + 1):
                try:
                    await self.translate(request, timeout=limit, _warm_up=True)
                    self.warm = True
                    break
                except Exception as exc:  # timeout / server not ready yet: try again, the server keeps loading
                    log.warning("local model warm-up attempt %d/%d failed: %s", attempt, WARM_UP_ATTEMPTS, exc)
            if self.warm and self._s.llm_max_concurrency > 1:  # other slots: the prefix is cached now, so these are fast
                await asyncio.gather(
                    *(self.translate(request, timeout=limit, _warm_up=True) for _ in range(self._s.llm_max_concurrency - 1)),
                    return_exceptions=True,
                )
        finally:
            self.warming = False
        log.info("local model warm-up", extra={"warm": self.warm, "ms": round((time.perf_counter() - started) * 1000)})

    async def _decode_counter(self) -> float | None:
        """llama-server's total decode count (/metrics, needs --metrics). It only grows while the server runs, so a
        smaller value than before means the server restarted and its prompt cache is empty."""
        try:
            r = await self._client.get("/metrics", timeout=2.0)
        except httpx.HTTPError:
            return None
        m = re.search(r"^llamacpp:n_decode_total ([0-9.e+-]+)$", r.text, re.M)
        return float(m.group(1)) if m else None

    async def keep_warm(self, ctx_provider, today_provider, interval_s: float = 15.0) -> None:
        """Background watchdog: re-run the warm-up whenever llama-server restarted (2026-11-19: a restart 33 s before a
        question left the backend believing it was warm, and the question paid the full ~4-minute cold prompt read) or
        when the startup warm-up could not finish. Runs until cancelled; never raises."""
        last: float | None = None
        while True:
            await asyncio.sleep(interval_s)
            try:
                up = await self.health()
                counter = await self._decode_counter() if up else None
                if counter is not None and last is not None and counter < last:
                    log.warning("llama-server restarted (its counters reset): warming its prompt cache again")
                    self.warm = False
                if counter is not None:
                    last = counter
                if up and not self.warm and not self.warming:
                    await self.warm_up(ctx_provider(), today_provider())
                    last = await self._decode_counter()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # the watchdog must never die
                log.warning("model warm-up watchdog: %s", type(exc).__name__)

    def _schema(self, ctx: SchemaContext) -> dict:
        key = (tuple(ctx.statuses), tuple(ctx.care_types), tuple(ctx.wings))
        if self._schema_cache is None or self._schema_cache[0] != key:
            self._schema_cache = (key, build_output_schema(ctx))
        return self._schema_cache[1]

    async def _post(self, body: dict, timeout: float) -> tuple[httpx.Response, int]:
        """One model call under ONE deadline: waiting for a free slot and the request itself share `timeout`.

        Before, the slot wait and the HTTP request each got the full timeout, so a queued call could take twice the
        configured time (measured: 59.5 s with a 30 s setting) before failing. The request gets only what is left.
        llama-server sends nothing until a non-streamed answer is complete, so httpx's read timeout is the request's
        total time (measured: timeouts fire at the deadline). On timeout httpx closes the connection; llama-server
        notices the disconnect and frees the slot at its next token (measured: within ~2 s, after the prompt chunk it
        is processing), so an abandoned request never generates the rest of its plan.

        Raises TranslatorUnavailable: "overloaded" (no slot within the deadline), "timeout", "connection".
        Returns the response and the inference time in ms (slot wait excluded).
        """
        deadline = time.perf_counter() + timeout
        try:
            await asyncio.wait_for(self._slots.acquire(), timeout=timeout)
        except TimeoutError as exc:
            log.warning("local model busy: no free inference slot within %ss", timeout)
            raise TranslatorUnavailable("overloaded") from exc
        started = time.perf_counter()
        remaining = deadline - started
        try:
            if remaining < min(MIN_REQUEST_SECONDS, timeout / 2):  # the wait used the budget: a doomed request
                log.warning("local model busy: %.1fs left after waiting for a slot", remaining)  # would only load the CPU
                raise TranslatorUnavailable("overloaded")
            response = await self._client.post(
                "/v1/chat/completions", json=body, timeout=httpx.Timeout(remaining, connect=2.0)
            )
        except httpx.TimeoutException as exc:
            log.warning(
                "local model timed out after %.1fs (deadline %ss, slot wait %.1fs)",
                time.perf_counter() - started, timeout, started - (deadline - timeout),
            )  # fmt: skip
            raise TranslatorUnavailable("timeout") from exc
        except httpx.HTTPError as exc:
            log.warning("local model unreachable at %s: %s", self._s.llm_server_url, type(exc).__name__)
            raise TranslatorUnavailable("connection") from exc
        finally:
            self._slots.release()
        return response, round((time.perf_counter() - started) * 1000)

    async def translate(self, request: TranslationRequest, timeout: float | None = None, _warm_up: bool = False) -> Translation:
        """timeout: overrides LLM_TIMEOUT_SECONDS (used only by warm_up). request.timeout_s (the request's remaining
        budget) caps it further, so a retry only gets what the first call left."""
        s = self._s
        if self.warming and not self.warm and not _warm_up:
            # the server is still reading the prompt for the first time: a question now could only wait and time out
            # (and its abandoned request would slow the warm-up down) — say so at once instead
            raise TranslatorUnavailable("warming_up")
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
        body = {
            "model": self.model_name,
            "messages": build_messages(ctx, user),
            "temperature": s.llm_temperature,
            "top_p": s.llm_top_p,
            "top_k": s.llm_top_k,
            "seed": s.llm_seed,
            "max_tokens": s.llm_max_tokens,
            "cache_prompt": True,  # reuse the KV cache of the static system prompt + examples
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "request_ir", "schema": self._schema(ctx)},
            },
        }
        if s.llm_chat_template_kwargs:  # e.g. Qwen3 non-thinking mode
            body["chat_template_kwargs"] = dict(s.llm_chat_template_kwargs)
        response, elapsed = await self._post(body, timeout)

        if response.status_code != 200:
            log.warning("local model HTTP %s: %s", response.status_code, response.text[:200])
            raise TranslatorUnavailable("unavailable" if response.status_code == 503 else "api_error")
        data = response.json()
        u = data.get("usage") or {}
        usage = ModelUsage(1, u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0, elapsed)
        choice = (data.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "length":
            log.warning("local model hit max_tokens before finishing RequestIR")
            raise TranslatorUnavailable("max_tokens")
        message = choice.get("message") or {}
        # Thinking mode: llama-server (--reasoning-format deepseek) returns the thoughts in `reasoning_content`, and
        # the JSON grammar applies only after </think>. Thoughts are never parsed, returned or logged verbatim; as a
        # second guard, anything up to a stray </think> in `content` is dropped and only the final JSON is parsed.
        reasoning_chars = len(message.get("reasoning_content") or "")
        text = _final_answer(message.get("content") or "")
        try:
            request_ir = json.loads(text)
        except json.JSONDecodeError as exc:
            log.warning("local model returned invalid JSON")
            raise TranslatorUnavailable("invalid_json") from exc
        if not isinstance(request_ir, dict):
            raise TranslatorUnavailable("invalid_json")
        timings = data.get("timings") or {}
        log.info(
            "local model RequestIR",
            extra={
                "inference_ms": elapsed,
                "prompt_tokens": usage.prompt_tokens,
                "cached_tokens": timings.get("cache_n"),
                "completion_tokens": usage.completion_tokens,
                "reasoning_chars": reasoning_chars,
            },
        )
        return Translation(request_ir=request_ir, usage=usage, model=self.model_name)

    async def complete_text(self, messages: list[dict], max_tokens: int, timeout: float) -> str:
        """Free-text completion (answer phrasing, engine/phraser.py): same server, same slot limit, greedy decoding,
        no JSON grammar. Raises TranslatorUnavailable on any failure; the caller keeps its template answer."""
        s = self._s
        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": s.llm_temperature,
            "top_p": s.llm_top_p,
            "top_k": s.llm_top_k,
            "seed": s.llm_seed,
            "max_tokens": max_tokens,
            "cache_prompt": True,
        }
        if s.llm_chat_template_kwargs:
            body["chat_template_kwargs"] = dict(s.llm_chat_template_kwargs)
        response, _ms = await self._post(body, timeout)
        if response.status_code != 200:
            raise TranslatorUnavailable("unavailable" if response.status_code == 503 else "api_error")
        choice = (response.json().get("choices") or [{}])[0]
        if choice.get("finish_reason") == "length":
            raise TranslatorUnavailable("max_tokens")
        return (choice.get("message") or {}).get("content") or ""


_THINK_END = "</think>"


def _final_answer(content: str) -> str:
    """The structured answer only: text after the last </think> (if any), with any <think> block removed."""
    if _THINK_END in content:
        content = content.rsplit(_THINK_END, 1)[1]
    return content.replace("<think>", "").strip()
