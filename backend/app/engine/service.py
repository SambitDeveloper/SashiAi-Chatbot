"""Query service: question → mask → (plan cache | deterministic planner | local Qwen) → shared validation → MongoDB → answer.

Order of planners: a cached plan, then the deterministic planner (no model), then the local Qwen model only when
the deterministic planner cannot safely resolve the question. Every plan, whichever planner produced it, goes
through the same resolver (exact database values), Pydantic validation and grounding check (engine/grounding.py)
before the read-only executor runs it. No external language model is ever called.
"""

import asyncio
import copy
import dataclasses
import hashlib
import json
import logging
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import yaml

from app.config import Settings
from app.data.repository import ResidentRepository
from app.engine.describe import describe_plan
from app.engine.executor import execute
from app.engine.grounding import DateScan, check_plan, out_of_scope_terms, scan_question
from app.engine.resolver import resolve_plan
from app.engine.responder import HELP_OPTIONS, build_response
from app.engine.steps import StepTrace, reason_code
from app.engine.stores import TTLCache
from app.models.plan import QueryPlan, canonical_field
from app.nlu.direct_query import DirectQueryBuilder
from app.nlu.grammar import skeleton as grammar_skeleton
from app.nlu.intent_router import DEFAULT_ROUTES_FILE, SemanticIntentRouter
from app.nlu.missing_values import MissingValueDetector
from app.nlu.projection import ProjectionDetector
from app.nlu.prompt import filters_to_conditions
from app.nlu.references import ReferenceResolver
from app.nlu.structure import StructureAnalyzer
from app.nlu.topic_gate import TopicGate
from app.nlu.translator import ModelUsage, QueryTranslator, SchemaContext, TranslationRequest, TranslatorUnavailable
from app.nlu.vocabulary import EntityIndex

log = logging.getLogger(__name__)

_RESIDENT_ID = re.compile(r"\bRES[-_ ]?\d{3,6}\b", re.IGNORECASE)
GATE_MESSAGE = (
    "That doesn't look like a question about residents. I can help with resident counts, status, care type, "
    "facility, room, caregiver, age and admission or discharge dates."
)

OR_ACROSS_FIELDS_MESSAGE = (
    "I can't combine conditions on different fields with OR in one question. Please ask them separately, e.g. "
    "'How many residents have no caregiver?' and 'How many residents have no room?'."
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
    by_tier: dict[str, int] = field(
        default_factory=lambda: {"llm": 0, "cache": 0, "gate": 0, "direct": 0, "unavailable": 0}
    )
    model_calls: int = 0  # local Qwen inferences (no per-token cost)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    inference_ms: int = 0
    llm_errors: int = 0
    completeness_retries: int = 0  # model retried after failing the grounding check
    external_llm_calls: int = 0  # always 0: there is no external model client in this application

    def add(self, u: ModelUsage) -> None:
        self.model_calls += u.calls
        self.prompt_tokens += u.prompt_tokens
        self.completion_tokens += u.completion_tokens
        self.inference_ms += u.inference_ms


def _cache_key(today: str, masked_question: str, previous_json: str | None, context: str = "") -> str:
    """`context`: vocabulary/schema version, data scope and model identity — when any changes, nothing is reused."""
    normalised = " ".join(masked_question.lower().split())
    return hashlib.sha256(f"{today}\n{normalised}\n{previous_json or ''}\n{context}".encode()).hexdigest()


def _normalised_plan(raw: Any) -> str:
    """A model plan in canonical form (key order, condition order, list order), to detect an identical retry."""
    if not isinstance(raw, dict):
        return json.dumps(raw)
    plan = {k: v for k, v in raw.items() if k != "uses_previous"}
    if isinstance(plan.get("conditions"), list):
        plan["conditions"] = sorted(
            ({k: sorted(v) if isinstance(v, list) else v for k, v in c.items()} if isinstance(c, dict) else c
             for c in plan["conditions"]),
            key=lambda c: json.dumps(c, sort_keys=True),
        )  # fmt: skip
    return json.dumps(plan, sort_keys=True, default=str)


MODEL_DECLINED_MESSAGE = (
    "I couldn't work out a safe way to answer that from the resident records. Please rephrase it, for example: "
    "'active residents in memory care admitted last month' or 'what is <resident name>'s room?'."
)


def _clarify_message(g) -> str:
    if g.blocking:  # say exactly what cannot be done; a rephrase of the same thing would not help
        return "I can't answer that as asked: " + "; ".join(g.blocking) + "."
    return (
        f"I couldn't map every part of your question safely ({'; '.join(g.problems())}). "
        "Please rephrase it, for example: 'active residents in memory care admitted last month'."
    )


class QueryService:
    def __init__(
        self,
        repo: ResidentRepository,
        translator: QueryTranslator | None,
        settings: Settings,
        gate: TopicGate | None = None,
        intent_router: SemanticIntentRouter | None = None,
    ):
        self.repo = repo
        self.translator = translator
        self.settings = settings
        self.gate = gate
        self.intent_router = intent_router
        self.direct_builder = (
            DirectQueryBuilder(intent_router.config)
            if intent_router is not None and settings.direct_answers_enabled
            else None
        )
        routes = intent_router.config if intent_router is not None else _load_routes(settings)
        # Question structure (subjects, references, comparisons): shared by the pre-check below and the validator.
        self.structure = StructureAnalyzer(routes)
        # Stage 1: comparisons with another resident (or among residents) are not supported yet. They are answered
        # with a clear message before any planner runs, so they never cost a model call and never execute.
        self.precheck_comparisons = True
        direct_cfg = routes.get("direct", {})
        # Date-field anchor words, shared by the deterministic planner and the grounding check (one source of truth).
        self.missing_detector = MissingValueDetector(routes)
        self.reference_resolver = ReferenceResolver(routes)
        self.projection_detector = ProjectionDetector(routes)
        self.date_anchors: dict[str, list[str]] = {
            f: [str(w).lower() for w in words] for f, words in (direct_cfg.get("date_fields") or {}).items()
        }
        self.cache = TTLCache(settings.cache_max_entries, settings.cache_ttl_seconds)
        # Questions whose model plan failed validation: asking the same thing in the same context again costs no
        # model calls. Keyed like the plan cache (incl. conversation context, schema/vocabulary version, data scope,
        # model identity), with the shorter conversation lifetime.
        self.failures = TTLCache(settings.cache_max_entries, settings.session_ttl_seconds)
        self.vocab_version = ""
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
        vocab = await self.repo.vocabulary()
        self.index = EntityIndex(vocab)
        # schema/vocabulary version: changes whenever the values the planners resolve against change
        self.vocab_version = hashlib.sha256(
            json.dumps(dataclasses.asdict(vocab), sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    @property
    def cache_context(self) -> str:
        """Everything besides the question and the conversation that decides how a question is interpreted."""
        s = self.settings
        scope = f"{s.data_source}:{s.mongodb_db}.{s.mongodb_collection}"  # the data scope (tenant scope: not built yet)
        model = f"{s.llm_model_name}:{json.dumps(s.llm_chat_template_kwargs, sort_keys=True)}"
        return f"{self.vocab_version}|{scope}|{model}"

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
        """Answer one question. Every pipeline step is logged as step-01 … step-19 (app/engine/steps.py)."""
        query_id = f"q_{secrets.token_hex(6)}"
        trace = StepTrace(query_id, log)
        trace.record("receive_question", "received", chars=len(question), in_conversation=bool(session_id))
        try:
            return await self._handle(question, session_id, query_id, trace)
        except Exception as exc:
            trace.finish("failed", error=type(exc).__name__)  # the full trace even when a step raises
            raise

    async def _handle(self, question: str, session_id: str | None, query_id: str, trace: StepTrace) -> dict[str, Any]:
        started = time.perf_counter()
        assert self.index is not None, "service not started"
        today = self.settings.today()
        today_d = date.fromisoformat(today)
        trace.start("mask_names")
        mask = self.index.mask(question)
        trace.record("mask_names", "ok", residents_masked=len(mask.placeholders),
                     ambiguous_names=sum(1 for ids in mask.placeholders.values() if len(ids) > 1))  # fmt: skip
        scan = scan_question(mask.text, today_d)
        trace.start("question_structure")
        structure = self.structure.analyse(mask.text)
        trace.record(
            "question_structure", "comparison" if structure.is_comparison else "ok",
            operations=structure.operations, subjects=len(structure.subjects), references=len(structure.references),
            comparisons=[f"{c.field}:{c.op}" for c in structure.comparisons] or None,
            cross_record=bool(structure.cross_record) or None, dates_found=len(scan.dates),
        )  # fmt: skip

        previous = self.sessions.get(session_id) if session_id else None
        previous_filters = dict(previous.get("filters") or {}) if previous else {}
        if previous and "filters" in previous:  # show the model its own (conditions) format
            conditions = filters_to_conditions(previous.pop("filters"))
            previous = {**previous, "conditions": conditions}
        previous_json = json.dumps(previous, sort_keys=True, separators=(",", ":")) if previous else None
        trace.record("conversation_memory", "previous_plan" if previous else "none",
                     previous_action=(previous or {}).get("action"))  # fmt: skip

        # Two cache keys: standalone questions are cached WITHOUT the conversation, so a repeat hits the cache
        # whatever was asked before; real follow-ups ("of them") are cached together with the previous plan.
        standalone_key = _cache_key(today, mask.text, None, self.cache_context)
        followup_key = _cache_key(today, mask.text, previous_json, self.cache_context) if previous_json else None
        # a failed interpretation is tied to the exact conversation context, never reused across contexts
        failure_key = _cache_key(today, mask.text, previous_json, self.cache_context)

        usage = ModelUsage()
        tier = "cache"
        trace.start("plan_cache")
        followup_hit = self.cache.get(followup_key) if followup_key else None
        cached = followup_hit or self.cache.get(standalone_key)
        trace.record("plan_cache", "hit" if cached else "miss", method="exact fingerprint",
                     key=("follow-up" if followup_hit else "standalone") if cached else None, entries=len(self.cache))  # fmt: skip
        raw = cached["plan"] if cached else None
        uses_previous = bool(cached and cached.get("uses_previous"))
        fallback_reason = ""  # why the deterministic planner did not answer this question (observability)
        model_trace: list[dict] = []  # every plan the local model generated for this question, as generated
        if raw is None:
            # Local topic check before any model inference. Questions that mention a specific resident are about
            # residents by definition, so they skip the check.
            if self.gate and not mask.placeholders and not _RESIDENT_ID.search(question):
                trace.start("topic_check")
                verdict = await asyncio.to_thread(self.gate.check, question)
                trace.record("topic_check", "blocked" if verdict.blocked else "passed", method="embedding (bge-small)",
                             resident_score=round(verdict.resident_score, 3), off_topic_score=round(verdict.off_topic_score, 3))  # fmt: skip
                if verdict.blocked:
                    result = self._gated(query_id, question, started)
                    trace.finish("off_topic", tier="gate", model_calls=0)
                    return result
            else:
                trace.skip("topic_check", "topic check disabled" if not self.gate else "names a resident")
            # Policy check: topics the resident records do not hold are refused deterministically (no model call).
            oos = out_of_scope_terms(question)
            trace.record("policy_check", "refused" if oos else "passed", topics=oos or None)
            if oos:
                result = self._out_of_scope(query_id, question, started, oos)
                trace.finish("refused_by_policy", tier="gate", model_calls=0)
                return result
            if self.precheck_comparisons and structure.is_comparison:
                kind = "reference" if structure.references else "cross_record"
                raw, tier = {"action": "clarify", "message": self.structure.messages[kind]}, "direct"
                fallback_reason = f"unsupported:comparison:{kind}"
                trace.record("comparison_check", "not_supported", kind=kind)
            else:
                trace.record("comparison_check", "passed" if not structure.is_comparison else "precheck_off")
            intent = None
            if raw is None and self.intent_router and self.intent_router.ready:
                # Classify the KIND of question on the skeleton (date/age phrases removed): they carry no intent
                # signal but would pull the embedding away from the route examples and drop the router's score.
                route_text = grammar_skeleton(mask.text, today_d)
                trace.start("intent_classifier")
                intent = await asyncio.to_thread(self.intent_router.score, route_text)
                trace.record("intent_classifier", "scored" if intent else "no_score", method="embedding (bge-small)",
                             route=intent.route if intent else None,
                             confidence=round(intent.confidence, 3) if intent else None,
                             margin=round(intent.margin, 3) if intent else None)  # fmt: skip
                if intent:
                    log.info(
                        "semantic router score",
                        extra={
                            "query_id": query_id,
                            "route": intent.route,
                            "confidence": intent.confidence,
                            "margin": intent.margin,
                            "scores": intent.scores,
                        },
                    )
            else:
                trace.skip("intent_classifier", "already decided" if raw is not None else "classifier not ready")
            # Deterministic planner (no model): a confident route + database values → plan. It must pass the same
            # resolver and grounding check as a model plan; anything uncertain goes on to the local model.
            if intent and self.direct_builder:
                trace.start("fast_path")
                direct = self.direct_builder.build(
                    question, mask, intent, self.index, today_d, has_previous=previous is not None
                )
                reason = direct.reason
                if direct.plan:
                    check = resolve_plan(direct.plan, self.index, mask, question)
                    grounded = None if check.problem else self._ground(check.plan, question, mask, scan, previous_filters if previous else None, structure=structure)
                    if grounded is not None and grounded.ok:
                        raw, tier = direct.plan, "direct"
                    elif grounded is not None and grounded.blocking:
                        # the question itself cannot be answered as asked (e.g. emergency contacts for many residents,
                        # an unresolvable "them"): no model can change that, so ask now
                        raw, tier = {"action": "clarify", "message": _clarify_message(grounded)}, "direct"
                        reason = "blocking:" + "; ".join(grounded.blocking)
                    else:
                        reason = (
                            f"resolve:{check.problem}"
                            if check.problem
                            else "grounding:" + "; ".join(grounded.problems())
                        )
                if tier != "direct":
                    fallback_reason = reason
                trace.record(
                    "fast_path",
                    ("answered" if raw.get("action") != "clarify" else "clarified") if tier == "direct" else "fallback_to_model",
                    reason=None if tier == "direct" else reason_code(reason),
                    chosen_by=direct.details.get("chosen_by"),
                )  # fmt: skip
                extra = {
                    "query_id": query_id,
                    "route": direct.details.get("route", intent.route),
                    "confidence": direct.details.get("confidence", intent.confidence),
                    "margin": direct.details.get("margin", intent.margin),
                    "chosen_by": direct.details.get("chosen_by", "router"),
                }
                if tier == "direct":
                    log.info("direct answer via MongoDB MCP (no language model)", extra={**extra, "action": raw["action"]})
                else:
                    words = [w for w in direct.details.get("words", []) if w.islower()]  # never names
                    log.info(
                        "direct answer not possible; asking language model",
                        extra={**extra, "reason": reason, **({"words": words} if words else {})},
                    )
            elif not fallback_reason and raw is None:
                fallback_reason = "planner_not_ready"
            trace.skip("fast_path", "already decided" if raw is not None else "fast path not available")
            if raw is None and fallback_reason == "missing_or_across_fields":
                # "no caregiver OR no room": the plan format has no OR across fields and no model can change that,
                # so ask right away instead of spending model calls on a plan the validator must reject.
                raw, tier = {"action": "clarify", "message": OR_ACROSS_FIELDS_MESSAGE}, "direct"
            if raw is None and (failed := self.failures.get(failure_key)):
                # same question, same conversation, same schema and model: the model already failed on it
                model_trace.append({"cached_failure": True, "problems": failed["problems"]})
                raw, tier = {"action": "clarify", "message": failed["message"]}, "cache"
                trace.record("failure_cache", "hit")
            elif raw is None:
                trace.record("failure_cache", "miss")
            else:
                trace.skip("failure_cache", "already decided")
            if raw is None:
                if self.translator is None:
                    trace.record("local_model", "not_configured")
                    trace.finish("model_unavailable", tier="unavailable", model_calls=0)
                    return self._unavailable(
                        query_id,
                        question,
                        started,
                        "This question needs the local query model, which is not configured on this server. Simpler "
                        "questions still work, e.g. 'How many active residents are in memory care?'.",
                    )
                trace.start("local_model")
                try:
                    translation = await self.translator.translate(self._request(mask.text, today, previous_json, scan))
                except TranslatorUnavailable as exc:
                    self.stats.llm_errors += 1
                    log.warning("local model unavailable: %s", exc)
                    model_trace.append({"attempt": 1, "error": str(exc)})
                    trace.record("local_model", "unavailable", error=str(exc), model=getattr(self.translator, "model_name", None))
                    trace.finish("model_unavailable", tier="unavailable", model_calls=0)
                    return self._unavailable(
                        query_id,
                        question,
                        started,
                        "I couldn't interpret that question right now. Please try again in a moment or rephrase it "
                        "more simply.",
                        model_trace=model_trace,
                        fallback_reason=fallback_reason,
                    )
                self._trace_model(query_id, model_trace, 1, translation.plan, mask)
                raw, usage, tier = translation.plan, translation.usage, "llm"
                uses_previous = bool(raw.pop("uses_previous", False)) if isinstance(raw, dict) else False
                self.stats.add(usage)
                trace.record("local_model", "planned", model=getattr(self.translator, "model_name", None),
                             inference_ms=translation.usage.inference_ms, prompt_tokens=translation.usage.prompt_tokens,
                             completion_tokens=translation.usage.completion_tokens,
                             action=raw.get("action") if isinstance(raw, dict) else None)  # fmt: skip
            else:
                trace.skip("local_model", "already decided")
        else:
            trace.skip_until("resolve_values", "plan cache hit")

        retried = False

        retry_info: dict[str, Any] = {}
        timing = {"resolve_ms": 0, "validate_ms": 0, "retry_ms": 0}

        def resolve():
            t = time.perf_counter()
            out = resolve_plan(raw, self.index, mask, question)
            timing["resolve_ms"] += round((time.perf_counter() - t) * 1000)
            return out

        def validate(p):
            t = time.perf_counter()
            out = self._ground(p, question, mask, scan, previous_filters if previous else None, uses_previous, structure)
            timing["validate_ms"] += round((time.perf_counter() - t) * 1000)
            return out

        async def retry_once(hint: str, feedback: list[str], trigger: str = "validation") -> bool:
            """One more model attempt with feedback (never more than one per question). True when a plan came back."""
            nonlocal usage, raw, uses_previous, retried
            retried = True
            retry_info.update(trigger=trigger, problem_count=len(feedback))
            self.stats.completeness_retries += 1
            # problem counts only: the problem texts can quote resident or staff names
            log.warning("model plan rejected; retrying", extra={"query_id": query_id, "problem_count": len(feedback)})
            t = time.perf_counter()
            try:
                retry = await self.translator.translate(self._request(mask.text, today, previous_json, scan, hint=hint))
            except TranslatorUnavailable as exc:
                self.stats.llm_errors += 1
                model_trace.append({"attempt": 2, "error": str(exc), "feedback": feedback})
                retry_info["error"] = str(exc)
                timing["retry_ms"] += round((time.perf_counter() - t) * 1000)
                return False
            timing["retry_ms"] += round((time.perf_counter() - t) * 1000)
            self._trace_model(query_id, model_trace, 2, retry.plan, mask, feedback=feedback)
            if model_trace and _normalised_plan(retry.plan) == _normalised_plan(model_trace[0].get("plan")):
                model_trace[-1]["identical_to_previous"] = True  # the feedback did not change the plan
                retry_info["identical_plan"] = True
                log.info("retry produced the same plan", extra={"query_id": query_id})
            self.stats.add(retry.usage)
            usage = usage + retry.usage
            raw = retry.plan
            uses_previous = bool(raw.pop("uses_previous", False)) if isinstance(raw, dict) else False
            return True

        outcome = resolve()
        # A resident used as a caregiver/facility/room is a planning error a model can correct (the resident exists;
        # "couldn't find" would be wrong): one retry with that feedback.
        if outcome.problem and outcome.problem.startswith("role:") and tier == "llm" and outcome.hint:
            if await retry_once(outcome.hint, [outcome.problem], trigger="wrong_role"):
                outcome = resolve()
        plan = outcome.plan
        grounding_problems: list[str] = []
        g = None
        if not outcome.problem:
            g = validate(plan)
            # Model plan failed the grounding check: retry once with the problems as feedback — but only when the
            # problems are ones a model can correct (a dropped or invented condition/field). Blocking problems
            # (unsupported structure, conversation context, schema/policy limits) go straight to the user.
            if not g.ok and tier == "llm" and not g.correctable:
                model_trace.append({"retry_skipped": True, "reason": g.blocking})
                log.info("model plan failed with a non-correctable problem; not retrying", extra={"query_id": query_id})
            if not g.ok and tier == "llm" and g.correctable and not retried:
                if await retry_once(g.hint(), g.problems()):
                    outcome = resolve()
                    plan = outcome.plan
                    if not outcome.problem:
                        g = validate(plan)
            if not outcome.problem:
                if g.ok:
                    plan = g.plan
                    if g.fixes:
                        log.info("plan canonicalised", extra={"query_id": query_id, "fixes": g.fixes})
                else:
                    grounding_problems = g.problems()
                    plan = QueryPlan(action="clarify", message=_clarify_message(g))
                    if tier == "llm":  # remember: the same question in the same context must not cost model calls again
                        self.failures.set(failure_key, {"problems": grounding_problems, "message": plan.message})

        # What the user reads is always written by the backend, never by the model:
        #   • a comparison question gets the "not supported yet" explanation, whatever stopped it;
        #   • a model that answered "clarify"/"unsupported" by itself gets a fixed request to rephrase (its own wording
        #     can be wrong — "not a valid resident" for a resident that exists twice — or leak placeholders).
        if structure.is_comparison and not plan.is_data_query:
            kind = "reference" if structure.references else "cross_record"
            plan = QueryPlan(action="clarify", message=self.structure.messages[kind])
            fallback_reason = fallback_reason or f"unsupported:comparison:{kind}"
        elif tier in ("llm", "cache") and not outcome.problem and not grounding_problems and plan.action in ("clarify", "unsupported"):
            model_trace.append({"model_message_replaced": True, "action": plan.action})
            plan = QueryPlan(action="clarify", message=MODEL_DECLINED_MESSAGE)

        trace.record("resolve_values", "ok" if not outcome.problem else "clarify", duration_ms=timing["resolve_ms"],
                     problem=reason_code(outcome.problem), action=plan.action)  # fmt: skip
        if g is None:
            trace.skip("validate_plan", "values did not resolve")
        else:
            trace.record("validate_plan", "passed" if g.ok else "rejected", duration_ms=timing["validate_ms"],
                         missing=len(g.missing) or None, invented=len(g.invented) or None,
                         blocking=len(g.blocking) or None, fixes=len(g.fixes) or None)  # fmt: skip
        if retried:
            trace.record("model_retry", "retried", duration_ms=timing["retry_ms"], **retry_info)
        elif g is not None and not g.ok and tier == "llm" and not g.correctable:
            trace.record("model_retry", "skipped", reason="not correctable by a model")
        else:
            trace.skip("model_retry", "not needed")
        cacheable = tier == "llm" and not outcome.problem and not grounding_problems
        if outcome.problem:
            log.info("plan resolution: %s", outcome.problem)
        # (a plan that resolved and grounded cleanly is cached only AFTER it has executed successfully, below: a plan
        # that fails at execution must not be replayed from the cache for every later ask)

        # Model-written text (a clarifying question or "not available" reason) may contain the resident_N placeholders
        # the model saw; show the user the names they actually typed instead.
        if plan.message and mask.originals:
            plan.message = _unmask(plan.message, mask.originals)

        exec_started = time.perf_counter()
        data = await execute(plan, self.repo)
        db_ms = round((time.perf_counter() - exec_started) * 1000)
        trace.record("execute_query", "ok" if plan.is_data_query else "no_query", duration_ms=db_ms, action=plan.action,
                     matched=data.matched, rows=len(data.rows) + len(data.looked_up or []))  # fmt: skip
        trace.start("build_answer")
        response = build_response(plan, data)
        trace.record("build_answer", "clarification" if response.get("needsClarification") else "answered")
        cached_now = False
        if cacheable and not outcome.problem:  # executed and answered: now it may be reused
            entry = {"plan": raw, "uses_previous": uses_previous}
            self.cache.set(followup_key if (uses_previous and followup_key) else standalone_key, entry)
            cached_now = True
        saved = bool(session_id and plan.is_data_query)
        if saved:
            self.sessions.set(session_id, plan.model_dump(mode="json", exclude_none=True))
        trace.record("remember", "ok", plan_cached=cached_now, conversation_saved=saved)

        latency = round((time.perf_counter() - started) * 1000)
        self.stats.queries += 1
        self.stats.by_tier[tier] += 1
        documents_fetched = len(data.rows) + len(data.looked_up or [])
        log.info(
            "query handled",
            extra={
                "query_id": query_id,
                "selectedTier": "deterministic" if tier in ("cache", "direct") else tier,
                "plannerName": {"direct": "direct_query", "cache": "plan_cache", "llm": "local_qwen"}.get(tier, tier),
                "plannerDecision": "answered" if tier in ("direct", "cache") else "fallback",
                "fallbackReason": fallback_reason or None,
                "modelCalls": usage.calls,
                "externalLlmCalls": 0,
                "inferenceMs": usage.inference_ms,
                "resolvedAction": plan.action,
                "resolvedFilters": plan.model_dump(mode="json", exclude_none=True).get("filters") or {},
                "validationResult": outcome.problem or ("grounding" if grounding_problems else "ok"),
                "databaseExecutionTimeMs": db_ms,
                "documentsFetched": documents_fetched,
                "responseGenerationMethod": "template",
                "matched": data.matched,
                "latency_ms": latency,
            },
        )
        trace.finish(
            "clarified" if response.get("needsClarification") else "answered",
            tier=tier, model_calls=usage.calls, matched=data.matched,
        )
        return self._result(
            query_id, question, plan, data.matched, data.total, response, tier, usage, latency,
            model_trace=model_trace, fallback_reason=fallback_reason,
        )

    def _trace_model(
        self, query_id: str, trace: list[dict], attempt: int, raw: Any, mask, feedback: list[str] | None = None
    ) -> None:
        """Record what the local model generated (before any resolving or correction).

        The API response gets the plan as generated (staff see it in the Inspector). The server log gets the same
        plan with resident references other than the resident_N placeholders replaced, so names never reach log files.
        """
        entry: dict[str, Any] = {"attempt": attempt, "plan": copy.deepcopy(raw)}
        if feedback:
            entry["feedback"] = feedback  # the problems the retry was asked to fix
        trace.append(entry)
        log.info(
            "local model plan generated",
            extra={"query_id": query_id, "attempt": attempt, "plan": _redact_for_log(raw),
                   **({"feedback_count": len(feedback)} if feedback else {})},
        )

    def schema_context(self, relevant_wings: list[str] | None = None) -> SchemaContext:
        """Schema metadata for the model: only the wings relevant to the question, never the whole facility list
        (each name costs ~16 prompt tokens; a real database's full list overflowed the context and made every new
        prompt slow). Any wing the model writes is still resolved and validated against the full live list."""
        idx = self.index
        example_wing = sorted(idx.wings)[0] if idx.wings else None
        return SchemaContext(
            idx.statuses, idx.care_types, list(relevant_wings or []), idx.caregiver_values(), example_wing
        )

    def _relevant_wings(self, masked: str, previous_json: str | None) -> list[str]:
        """Wings named in the question, plus those in the previous plan for a follow-up."""
        wings = [w for values in self.index.mentioned_wings(masked).values() for w in values]
        if previous_json:
            try:
                for cond in json.loads(previous_json).get("conditions", []):
                    if canonical_field(cond.get("field")) == "facility":
                        wings += list(cond.get("include", []) or []) + list(cond.get("exclude", []) or [])
            except (ValueError, AttributeError):
                pass
        return list(dict.fromkeys(wings))

    def _request(
        self, masked: str, today: str, previous_json: str | None, scan: DateScan, hint: str | None = None
    ) -> TranslationRequest:
        context = self.schema_context(self._relevant_wings(masked, previous_json))
        hints = scan.hints() + [
            f"'{masked[n.start : n.end]}' = age {' '.join(f'{k} {v}' for k, v in n.bounds.items())}" for n in scan.numbers
        ]
        return TranslationRequest(masked, today, context, previous_json, hints, hint)

    def _ground(
        self, plan: QueryPlan, question: str, mask, scan: DateScan, previous_filters: dict | None, uses_previous=False,
        structure=None,
    ):
        return check_plan(
            plan, question, mask, self.index, scan, self.date_anchors, previous_filters, uses_previous,
            missing_detector=self.missing_detector, reference_resolver=self.reference_resolver,
            projection_detector=self.projection_detector, structure=structure,
        )

    # ------------------------------------------------------------------ response shaping
    def _result(
        self, query_id, question, plan: QueryPlan, matched, total, response, tier, usage: ModelUsage, latency,
        model_trace: list[dict] | None = None, fallback_reason: str | None = None,
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
            # Local model accounting: tokens are processed on this server (no per-token cost, no external call).
            "tokensConsumed": usage.total,
            "tokenUsage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens},
            "modelCalls": usage.calls,
            "model": getattr(self.translator, "model_name", None) if usage.calls else None,
            "externalLlmCalls": 0,
            # Why the deterministic planner did not answer (empty when it did), and every plan the local model
            # generated for this question, as generated (attempt 2 = the retry, with the feedback it was given).
            "fallbackReason": fallback_reason or None,
            "modelPlans": model_trace or [],
            "tokensSaved": 0,
            "pipelineLatencyMs": latency,
            "policy": {
                "passed": plan.action != "unsupported",
                "policyCategory": "OUT_OF_SCOPE" if plan.action == "unsupported" else "RESIDENT_INTENT",
                "violationReason": plan.message if plan.action == "unsupported" else None,
            },
            "intent": {
                "category": _CATEGORY[plan.action],
                "confidence": 1.0 if tier in ("cache", "direct") else 0.9,
                "matchedPattern": " · ".join(interpretation) or plan.action,
                "parameters": plan_json,
                # Deterministic = resolved with no generative model: a direct planner answer or a cached plan.
                "isDeterministic": tier in ("cache", "direct"),
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
        return self._result(query_id, question, plan, 0, 0, response, "gate", ModelUsage(), latency)

    def _out_of_scope(self, query_id: str, question: str, started: float, terms: list[str]) -> dict[str, Any]:
        self.stats.queries += 1
        self.stats.by_tier["gate"] += 1
        message = (
            f"Resident records here don't include {terms[0]} information. I can help with resident counts, status, "
            "care type, facility, room, caregiver, age and admission or discharge dates."
        )
        plan = QueryPlan(action="unsupported", message=message)
        response = {
            "headline": "Not available",
            "summaryText": message,
            "count": 0,
            "isPolicyViolation": True,
            "clarificationOptions": HELP_OPTIONS[:3],
        }
        latency = round((time.perf_counter() - started) * 1000)
        log.info("query refused by policy", extra={"query_id": query_id, "tier": "gate", "latency_ms": latency})
        return self._result(query_id, question, plan, 0, 0, response, "gate", ModelUsage(), latency)

    def _unavailable(
        self, query_id: str, question: str, started: float, message: str,
        model_trace: list[dict] | None = None, fallback_reason: str | None = None,
    ) -> dict[str, Any]:
        self.stats.queries += 1
        self.stats.by_tier["unavailable"] += 1
        plan = QueryPlan(action="clarify", message=message)
        response = {"headline": "Assistant unavailable", "summaryText": message, "count": 0, "needsClarification": True}
        latency = round((time.perf_counter() - started) * 1000)
        return self._result(
            query_id, question, plan, 0, 0, response, "unavailable", ModelUsage(), latency,
            model_trace=model_trace, fallback_reason=fallback_reason,
        )


def _load_routes(settings: Settings) -> dict:
    path = settings.semantic_router_routes_file or DEFAULT_ROUTES_FILE
    try:
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except OSError:
        log.warning("routes file %s not found; date anchors disabled", path)
        return {}


_PLACEHOLDER = re.compile(r"^resident_\d+$")


def _redact_for_log(raw: Any) -> Any:
    """A model plan safe for log files: resident references that are not resident_N placeholders become
    "<resident name>", and free text the model wrote (message) is reduced to its length."""
    if not isinstance(raw, dict):
        return "<not a JSON object>"
    plan = copy.deepcopy(raw)
    lookup = plan.get("lookup")
    if isinstance(lookup, dict) and isinstance(lookup.get("residents"), list):
        lookup["residents"] = [
            r if isinstance(r, str) and _PLACEHOLDER.match(r) else "<resident name>" for r in lookup["residents"]
        ]
    if isinstance(plan.get("message"), str):
        plan["message"] = f"<{len(plan['message'])} characters>"
    return plan


def _unmask(text: str, originals: dict[str, str]) -> str:
    """resident_1 / Resident_1 → the name the user typed for that placeholder."""
    return re.sub(
        r"\bresident_(\d+)\b",
        lambda m: originals.get(f"resident_{m.group(1)}", m.group(0)),
        text,
        flags=re.IGNORECASE,
    )
