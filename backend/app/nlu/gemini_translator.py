"""Google Gemini translator: masked question → raw query plan (JSON), via the official google-genai SDK.

Token-minimising design (same contract as the other translators):
  • Structured output (`response_json_schema`) — Gemini can only return the plan, never prose or data.
  • Minimal thinking level by default — translation is a simple task.
  • Static system instruction, so Gemini's implicit prompt caching can bill it at the cached rate.
  • Resident names are masked before sending; repeated questions are answered from the plan cache.
"""

import json
import logging

import httpx
from google import genai
from google.genai import errors, types

from app.config import Settings
from app.nlu.prompt import OUTPUT_SCHEMA, SYSTEM_PROMPT, build_user_message
from app.nlu.translator import TokenUsage, Translation, TranslatorUnavailable

log = logging.getLogger(__name__)

_BLOCKED = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}


class GeminiTranslator:
    provider = "gemini"

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None):
        self._settings = settings
        self.model_name = settings.gemini_model
        self._client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=int(settings.gemini_timeout_seconds * 1000),  # milliseconds
                retry_options=types.HttpRetryOptions(attempts=settings.gemini_max_retries + 1),
                httpx_async_client=http_client,
            ),
        )
        thinking = (
            types.ThinkingConfig(thinking_level=settings.gemini_thinking_level.upper())
            if settings.gemini_thinking_level != "default"
            else None
        )
        self._config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_json_schema=OUTPUT_SCHEMA,
            max_output_tokens=settings.gemini_max_tokens,
            thinking_config=thinking,
            # No tools are used, so automatic function calling is off.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    async def close(self) -> None:
        aclose = getattr(self._client.aio, "aclose", None)
        if aclose:
            await aclose()

    async def translate(
        self,
        masked_question: str,
        today: str,
        wings: list[str],
        previous_plan_json: str | None,
        hint: str | None = None,
    ) -> Translation:
        model = self._settings.gemini_model
        try:
            response = await self._client.aio.models.generate_content(
                model=model,
                contents=build_user_message(masked_question, today, wings, previous_plan_json, hint),
                config=self._config,
            )
        except errors.ClientError as exc:  # 4xx
            reason = {
                400: "bad_request",
                401: "authentication",
                403: "permission",
                404: "model_not_found",
                429: "rate_limited",
            }
            kind = reason.get(exc.code, "client_error")
            if exc.code in (401, 403):
                log.error("Gemini rejected the API key or permission (HTTP %s); check GEMINI_API_KEY", exc.code)
            elif exc.code == 404:
                log.error("Gemini model %s not found; set GEMINI_MODEL to a current model ID", model)
            elif exc.code == 429:
                log.warning("Gemini rate limit or quota reached (after SDK retries)")
            else:
                log.error("Gemini rejected the request (HTTP %s): %s", exc.code, exc.message)
            raise TranslatorUnavailable(kind) from exc
        except errors.ServerError as exc:  # 5xx
            log.warning("Gemini server error (HTTP %s)", exc.code)
            raise TranslatorUnavailable("api_error") from exc
        except (httpx.HTTPError, TimeoutError) as exc:  # connection problems and timeouts
            log.warning("Could not reach Gemini: %s", type(exc).__name__)
            raise TranslatorUnavailable("connection") from exc

        usage = _usage(response.usage_metadata)
        served_by = response.model_version or model

        feedback = response.prompt_feedback
        if feedback and feedback.block_reason:
            log.info("Gemini blocked the prompt (%s)", feedback.block_reason)
            return _declined(usage, served_by)
        if not response.candidates:
            raise TranslatorUnavailable("empty_response")
        finish = response.candidates[0].finish_reason
        finish_name = getattr(finish, "name", str(finish or ""))
        if finish_name in _BLOCKED:
            log.info("Gemini declined the request (%s)", finish_name)
            return _declined(usage, served_by)
        if finish_name == "MAX_TOKENS":
            log.error("Gemini hit max_output_tokens before finishing the plan; raise GEMINI_MAX_TOKENS")
            raise TranslatorUnavailable("max_tokens")

        try:
            plan = json.loads(response.text or "")
        except json.JSONDecodeError as exc:
            log.error("Gemini returned non-JSON output")
            raise TranslatorUnavailable("invalid_json") from exc
        return Translation(plan=plan, usage=usage, model=served_by)


def _declined(usage: TokenUsage, model: str) -> Translation:
    return Translation(
        plan={"action": "unsupported", "message": "This request can't be processed by the assistant."},
        usage=usage,
        model=model,
    )


def _usage(u) -> TokenUsage:
    """Map Gemini usage metadata. Cached prompt tokens and thinking tokens are reported separately by Gemini."""
    if u is None:
        return TokenUsage()
    prompt = u.prompt_token_count or 0
    cached = u.cached_content_token_count or 0
    return TokenUsage(
        input_tokens=max(prompt - cached, 0),
        output_tokens=(u.candidates_token_count or 0) + (u.thoughts_token_count or 0),
        cache_read_input_tokens=cached,
    )
