"""Query service: question → masked → (cache | Claude) → resolved & validated plan → database → answer."""

import asyncio
import hashlib
import json
import logging
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.config import Settings
from app.data.repository import ResidentRepository
from app.engine.describe import describe_plan
from app.engine.executor import execute
from app.engine.resolver import resolve_plan
from app.engine.responder import HELP_OPTIONS, build_response
from app.engine.stores import TTLCache
from app.models.plan import QueryPlan
from app.nlu.topic_gate import TopicGate
from app.nlu.translator import TokenUsage, Translator, TranslatorUnavailable
from app.nlu.vocabulary import EntityIndex

log = logging.getLogger(__name__)

_RESIDENT_ID = re.compile(r"\bRES[-_ ]?\d{3,6}\b", re.IGNORECASE)
GATE_MESSAGE = (
    "That doesn't look like a question about residents. I can help with resident counts, status, care type, "
    "wing, room, caregiver, age and admission or discharge dates."
)

_CATEGORY = {
    "count": "RESIDENT_COUNT",
    "list": "RESIDENT_LIST",
    "breakdown": "RESIDENT_BREAKDOWN",
    "aggregate": "RESIDENT_AGGREGATE",
    "lookup": "RESIDENT_LOOKUP",
    "help": "HELP",
    "clarify": "AMBIGUOUS_QUERY",
    "unsupported": "OUT_OF_POLICY",
}


@dataclass
class Stats:
    queries: int = 0
    by_tier: dict[str, int] = field(default_factory=lambda: {"llm": 0, "cache": 0, "gate": 0, "unavailable": 0})
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    llm_errors: int = 0

    def add(self, u: TokenUsage) -> None:
        self.input_tokens += u.input_tokens
        self.output_tokens += u.output_tokens
        self.cache_read_input_tokens += u.cache_read_input_tokens
        self.cache_creation_input_tokens += u.cache_creation_input_tokens


def _cache_key(today: str, masked_question: str, previous_json: str | None) -> str:
    normalised = " ".join(masked_question.lower().split())
    return hashlib.sha256(f"{today}\n{normalised}\n{previous_json or ''}".encode()).hexdigest()


class QueryService:
    def __init__(
        self,
        repo: ResidentRepository,
        translator: Translator | None,
        settings: Settings,
        gate: TopicGate | None = None,
    ):
        self.repo = repo
        self.translator = translator
        self.settings = settings
        self.gate = gate
        self._llm_day = ""
        self._llm_calls_today = 0
        self.cache = TTLCache(settings.cache_max_entries, settings.cache_ttl_seconds)
        self.sessions = TTLCache(settings.session_max_entries, settings.session_ttl_seconds)
        self.stats = Stats()
        self.index: EntityIndex | None = None
        self._refresh_task: asyncio.Task | None = None

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        await self.refresh_vocabulary()
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def stop(self) -> None:
        if self._refresh_task:
            self._refresh_task.cancel()

    async def refresh_vocabulary(self) -> None:
        self.index = EntityIndex(await self.repo.vocabulary())

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.vocab_refresh_seconds)
            try:
                await self.refresh_vocabulary()
            except Exception:  # keep serving with the previous index
                log.exception("vocabulary refresh failed")

    @property
    def ready(self) -> bool:
        return self.index is not None

    # ------------------------------------------------------------------ main entry
    async def handle(self, question: str, session_id: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        assert self.index is not None, "service not started"
        query_id = f"q_{secrets.token_hex(6)}"
        today = self.settings.today()
        mask = self.index.mask(question)

        previous = self.sessions.get(session_id) if session_id else None
        previous_json = json.dumps(previous, sort_keys=True, separators=(",", ":")) if previous else None

        # Two cache keys: standalone questions are cached WITHOUT the conversation, so a repeat hits the cache
        # whatever was asked before; real follow-ups ("of them") are cached together with the previous plan.
        standalone_key = _cache_key(today, mask.text, None)
        followup_key = _cache_key(today, mask.text, previous_json) if previous_json else None

        usage = TokenUsage()
        tier = "cache"
        raw = (self.cache.get(followup_key) if followup_key else None) or self.cache.get(standalone_key)
        uses_previous = False
        if raw is None:
            # Local topic check (free, on this server) before any paid model call. Questions that mention a
            # specific resident are about residents by definition, so they skip the check.
            if self.gate and not mask.placeholders and not _RESIDENT_ID.search(question):
                verdict = await asyncio.to_thread(self.gate.check, question)
                if verdict.blocked:
                    return self._gated(query_id, question, started)
            if self.translator is None:
                return self._unavailable(
                    query_id, question, started, "The assistant's language model is not configured on this server."
                )
            if self._daily_limit_reached(today):
                log.warning(
                    "daily language-model call limit reached", extra={"limit": self.settings.llm_daily_call_limit}
                )
                return self._unavailable(
                    query_id, question, started, "The daily question limit has been reached. Please try again tomorrow."
                )
            try:
                translation = await self.translator.translate(mask.text, today, self.index.wings, previous_json)
            except TranslatorUnavailable as exc:
                self.stats.llm_errors += 1
                log.warning("translation unavailable: %s", exc)
                return self._unavailable(
                    query_id,
                    question,
                    started,
                    "The assistant is temporarily unavailable. Please try again in a moment.",
                )
            raw, usage, tier = translation.plan, translation.usage, "llm"
            uses_previous = bool(raw.pop("uses_previous", False)) if isinstance(raw, dict) else False
            self.stats.add(usage)

        outcome = resolve_plan(raw, self.index, mask, question)
        plan = outcome.plan
        if outcome.problem:
            log.info("plan resolution: %s", outcome.problem)
        elif tier == "llm":  # cache only plans that resolved cleanly
            self.cache.set(followup_key if (uses_previous and followup_key) else standalone_key, raw)

        data = await execute(plan, self.repo)
        response = build_response(plan, data)
        if session_id and plan.is_data_query:
            self.sessions.set(session_id, plan.model_dump(mode="json", exclude_none=True))

        latency = round((time.perf_counter() - started) * 1000)
        self.stats.queries += 1
        self.stats.by_tier[tier] += 1
        log.info(
            "query handled",
            extra={
                "query_id": query_id,
                "tier": tier,
                "action": plan.action,
                "tokens": usage.total,
                "latency_ms": latency,
                "matched": data.matched,
            },
        )
        return self._result(query_id, question, plan, data.matched, data.total, response, tier, usage, latency)

    # ------------------------------------------------------------------ response shaping
    def _result(
        self, query_id, question, plan: QueryPlan, matched, total, response, tier, usage: TokenUsage, latency
    ) -> dict[str, Any]:
        interpretation = describe_plan(plan, lambda i: self.index.display_by_id.get(i, i))
        plan_json = plan.model_dump(mode="json", exclude_none=True)
        is_data = plan.is_data_query
        return {
            "queryId": query_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "userQuery": question,
            "tier": tier,
            "interpretation": interpretation,
            "tokensConsumed": usage.total,
            "tokenUsage": usage.__dict__,
            "tokensSaved": 0,
            "pipelineLatencyMs": latency,
            "policy": {
                "passed": plan.action != "unsupported",
                "policyCategory": "OUT_OF_SCOPE" if plan.action == "unsupported" else "RESIDENT_INTENT",
                "violationReason": plan.message if plan.action == "unsupported" else None,
            },
            "intent": {
                "category": _CATEGORY[plan.action],
                "confidence": 1.0 if tier == "cache" else 0.9,
                "matchedPattern": " · ".join(interpretation) or plan.action,
                "parameters": plan_json,
                "isDeterministic": tier == "cache",
                "tokensConsumed": usage.total,
            },
            "mcpCall": {
                "jsonrpc": "2.0",
                "id": query_id,
                "method": "tools/call",
                "params": {"name": "care_app.query_residents", "arguments": plan_json},
            }
            if is_data
            else None,
            "mcpResponse": {
                "jsonrpc": "2.0",
                "id": query_id,
                "result": {
                    "tool": "care_app.query_residents",
                    "data": {"matched": matched, "total": total},
                    "totalMatched": matched,
                    "filtersApplied": plan_json.get("filters", {}),
                    "executionTimeMs": latency,
                },
            }
            if is_data
            else None,
            "response": {**response, "isLlmInterpreted": tier == "llm"},
        }

    def _daily_limit_reached(self, today: str) -> bool:
        """Counts model calls per calendar day (per process). Returns True when the configured cap is hit."""
        if self._llm_day != today:
            self._llm_day, self._llm_calls_today = today, 0
        limit = self.settings.llm_daily_call_limit
        if limit and self._llm_calls_today >= limit:
            return True
        self._llm_calls_today += 1
        return False

    def _gated(self, query_id: str, question: str, started: float) -> dict[str, Any]:
        self.stats.queries += 1
        self.stats.by_tier["gate"] += 1
        plan = QueryPlan(action="unsupported", message=GATE_MESSAGE)
        response = {
            "headline": "Not available",
            "summaryText": GATE_MESSAGE,
            "count": 0,
            "isPolicyViolation": True,
            "clarificationOptions": HELP_OPTIONS[:3],
        }
        latency = round((time.perf_counter() - started) * 1000)
        log.info("query filtered locally", extra={"query_id": query_id, "tier": "gate", "latency_ms": latency})
        return self._result(query_id, question, plan, 0, 0, response, "gate", TokenUsage(), latency)

    def _unavailable(self, query_id: str, question: str, started: float, message: str) -> dict[str, Any]:
        self.stats.queries += 1
        self.stats.by_tier["unavailable"] += 1
        plan = QueryPlan(action="clarify", message=message)
        response = {"headline": "Assistant unavailable", "summaryText": message, "count": 0, "needsClarification": True}
        latency = round((time.perf_counter() - started) * 1000)
        return self._result(query_id, question, plan, 0, 0, response, "unavailable", TokenUsage(), latency)
