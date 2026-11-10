"""Shared translator types, plus the Claude translator: masked question → raw query plan (JSON).

Token-minimising design:
  • Claude returns only a small JSON plan (structured outputs), never prose or data.
  • The static system prompt is prompt-cached; only the short user message is billed at full rate.
  • Low effort: translation is a simple task, so Claude thinks briefly.
  • Resident names are masked before sending; repeated questions are answered from the plan cache.
"""

import json
import logging
from dataclasses import dataclass
from typing import Protocol

import anthropic

from app.config import Settings
from app.nlu.prompt import OUTPUT_SCHEMA, SYSTEM_PROMPT, build_user_message

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens


@dataclass
class Translation:
    plan: dict
    usage: TokenUsage
    model: str


class TranslatorUnavailable(Exception):
    """The language model could not be reached or returned an unusable response."""


class Translator(Protocol):
    async def translate(
        self,
        masked_question: str,
        today: str,
        wings: list[str],
        previous_plan_json: str | None,
        hint: str | None = None,
    ) -> Translation: ...


class ClaudeTranslator:
    provider = "anthropic"

    def __init__(self, settings: Settings):
        self._settings = settings
        self.model_name = settings.anthropic_model
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key,  # None → SDK falls back to ANTHROPIC_API_KEY / profile
            timeout=settings.anthropic_timeout_seconds,
            max_retries=settings.anthropic_max_retries,
        )

    async def close(self) -> None:
        await self._client.close()

    async def translate(
        self,
        masked_question: str,
        today: str,
        wings: list[str],
        previous_plan_json: str | None,
        hint: str | None = None,
    ) -> Translation:
        s = self._settings
        extra: dict = {}
        if s.anthropic_refusal_fallbacks:
            extra = {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        try:
            response = await self._client.beta.messages.create(
                model=s.anthropic_model,
                max_tokens=s.anthropic_max_tokens,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=[
                    {
                        "role": "user",
                        "content": build_user_message(masked_question, today, wings, previous_plan_json, hint),
                    }
                ],
                output_config={
                    "effort": s.anthropic_effort,
                    "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
                },
                **extra,
            )
        except anthropic.AuthenticationError as exc:
            log.error("Claude authentication failed; check ANTHROPIC_API_KEY")
            raise TranslatorUnavailable("authentication") from exc
        except anthropic.PermissionDeniedError as exc:
            log.error("Claude API key lacks permission for model %s", s.anthropic_model)
            raise TranslatorUnavailable("permission") from exc
        except anthropic.NotFoundError as exc:
            log.error("Claude model %s not found", s.anthropic_model)
            raise TranslatorUnavailable("model_not_found") from exc
        except anthropic.BadRequestError as exc:
            log.error("Claude rejected the request: %s", exc.message)
            raise TranslatorUnavailable("bad_request") from exc
        except anthropic.RateLimitError as exc:
            log.warning("Claude rate limit reached (after SDK retries)")
            raise TranslatorUnavailable("rate_limited") from exc
        except anthropic.APIStatusError as exc:
            log.warning("Claude API error %s (request %s)", exc.status_code, getattr(exc, "request_id", None))
            raise TranslatorUnavailable("api_error") from exc
        except anthropic.APIConnectionError as exc:  # includes timeouts
            log.warning("Could not reach Claude: %s", type(exc).__name__)
            raise TranslatorUnavailable("connection") from exc

        u = response.usage
        usage = TokenUsage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_input_tokens=u.cache_read_input_tokens or 0,
            cache_creation_input_tokens=u.cache_creation_input_tokens or 0,
        )

        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None) if response.stop_details else None
            log.info("Claude declined the request (category=%s)", category)
            return Translation(
                plan={"action": "unsupported", "message": "This request can't be processed by the assistant."},
                usage=usage,
                model=response.model,
            )
        if response.stop_reason == "max_tokens":
            log.error("Claude hit max_tokens before finishing the plan; raise ANTHROPIC_MAX_TOKENS")
            raise TranslatorUnavailable("max_tokens")

        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            plan = json.loads(text)
        except json.JSONDecodeError as exc:
            log.error("Claude returned non-JSON output (request %s)", getattr(response, "_request_id", None))
            raise TranslatorUnavailable("invalid_json") from exc
        return Translation(plan=plan, usage=usage, model=response.model)
