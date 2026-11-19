"""Query service: question → mask → RequestIR → QueryPlan compiler → shared validation → MongoDB → answer.

Order of planners: the deterministic planner (no model), then a stored semantic plan template (engine/
semantic_cache.py: found by meaning, every slot refilled from THIS question, then the same validator), then the local
Qwen model only when neither can safely interpret the question. Model IR and deterministic plans use the same
compiler, resolver (exact database values), Pydantic validation and grounding check (engine/grounding.py) before
the read-only executor runs. No external language model is ever called.
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
from app.engine.phraser import ResponsePhraser
from app.engine.resolver import resolve_plan
from app.engine.responder import HELP_OPTIONS, build_response
from app.engine.semantic_cache import Compatibility, QuestionView, SemanticPlanCache
from app.engine.steps import StepTrace, reason_code
from app.engine.stores import TTLCache
from app.models.plan import QueryPlan, canonical_field
from app.nlu.direct_query import DirectQueryBuilder
from app.nlu.grammar import skeleton as grammar_skeleton
from app.nlu.intent_router import DEFAULT_ROUTES_FILE, SemanticIntentRouter
from app.nlu.missing_values import MissingValueDetector
from app.nlu.projection import ProjectionDetector
from app.nlu.references import ReferenceResolver
from app.nlu.request_ir import IRCompilationError, compile_request_ir, normalize_request_ir, request_ir_from_plan
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
_REQUEST_IR_VERSION = "request-ir-v3"
RETRY_MIN_BUDGET_S = 1.0  # a retry with less of the request's model-time budget left than this is not attempted
# What a stored plan template's meaning depends on besides the vocabulary: the plan model and the resident view mapping.
# Any change here makes every stored template incompatible (semantic_cache.Compatibility).
_RESIDENT_SCHEMA_FILE = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "resident_schema.yaml"
_SCHEMA_HASH = hashlib.sha256(
    (json.dumps(QueryPlan.model_json_schema(), sort_keys=True) + _RESIDENT_SCHEMA_FILE.read_text(encoding="utf-8")).encode()
).hexdigest()[:16]

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
        default_factory=lambda: {"llm": 0, "cache": 0, "template": 0, "gate": 0, "direct": 0, "unavailable": 0}
    )
    model_calls: int = 0  # local Qwen inferences (no per-token cost)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    inference_ms: int = 0
    llm_errors: int = 0
    completeness_retries: int = 0  # model retried after failing the grounding check
    external_llm_calls: int = 0  # requests sent to an off-server model (LLM_PROVIDER=bedrock); 0 with the local model
    model_calls_attempted: int = 0  # every request sent (or queued) to the local model, whatever its outcome
    model_calls_timed_out: int = 0

    def add(self, u: ModelUsage) -> None:
        self.model_calls += u.calls
        self.prompt_tokens += u.prompt_tokens
        self.completion_tokens += u.completion_tokens
        self.inference_ms += u.inference_ms


@dataclass
class ModelCalls:
    """Local-model requests made for ONE question: attempted = succeeded + timed_out + failed.

    `model_calls` / `modelCalls` (ModelUsage.calls) count only completed calls, so a question whose only call timed out
    reported model_calls=0 as if the model was never asked. These counters say what happened to each attempt.
    failed = no free slot within the deadline ("overloaded"), unreachable, HTTP error, invalid or truncated output.
    """

    attempted: int = 0
    succeeded: int = 0
    timed_out: int = 0
    failed: int = 0

    def record(self, error: str | None) -> None:
        self.attempted += 1
        if error is None:
            self.succeeded += 1
        elif error == "timeout":
            self.timed_out += 1
        else:
            self.failed += 1

    def trace(self) -> dict[str, int]:
        return {
            "model_calls_attempted": self.attempted,
            "model_calls_succeeded": self.succeeded,
            "model_calls_timed_out": self.timed_out,
            "model_calls_failed": self.failed,
        }

    def api(self) -> dict[str, int]:
        return {
            "modelCallsAttempted": self.attempted,
            "modelCallsSucceeded": self.succeeded,
            "modelCallsTimedOut": self.timed_out,
            "modelCallsFailed": self.failed,
        }


def _compile_hint(exc: IRCompilationError) -> str:
    """Retry feedback for a RequestIR the strict validator rejected: the exact problems, nothing reworded."""
    lines = ["Your previous RequestIR was rejected by the validator:", *(f"- {d}" for d in exc.details)]
    lines.append(
        "Fix ONLY these problems and keep every other part of the interpretation. Use only the listed fields and "
        "operators. If the question really needs something they cannot express, put that part in \"unsupported\" "
        "with its exact evidence instead of approximating it."
    )
    return "\n".join(lines)


def _compile_failure_message(failure: dict | None, mask: Any) -> str:
    """What the user reads when no executable plan is left. Written by the backend; quotes are the user's own words."""
    if not failure:
        return MODEL_DECLINED_MESSAGE

    def unmask(text: str) -> str:
        originals = getattr(mask, "originals", {}) or {}
        return re.sub(r"resident_\d+", lambda m: originals.get(m.group(0).lower(), m.group(0)), text)

    quotes = [unmask(q) for q in failure.get("quotes") or []][:2]
    shown = " and ".join(f'"{q}"' for q in quotes)
    if failure.get("capability"):
        return (f"I can't answer that yet: {failure['capability']} isn't supported. You can ask the question without "
                "that part, or split it into simpler questions.")  # fmt: skip
    if failure.get("code") == "declared_ambiguous" and shown:
        return f"Could you clarify what you mean by {shown}? I want to be sure I answer the right question."
    if failure.get("code") == "declared_unsupported" and shown:
        return (f"The resident records can't answer {shown}. You can ask the question without that part, or "
                "rephrase it.")  # fmt: skip
    return MODEL_DECLINED_MESSAGE


def _cache_key(today: str, masked_question: str, previous_json: str | None, context: str = "") -> str:
    """`context`: vocabulary/schema version, data scope and model identity — when any changes, nothing is reused."""
    normalised = " ".join(masked_question.lower().split())
    return hashlib.sha256(f"{today}\n{normalised}\n{previous_json or ''}\n{context}".encode()).hexdigest()


def _session_cache_key(session_id: str, context: str) -> str:
    """Bind caller-selected conversation IDs to this service's configured data scope."""
    return hashlib.sha256(f"{context}\n{session_id}".encode()).hexdigest()


def _normalised_plan(raw: Any) -> str:
    """Canonical JSON for comparing model RequestIR candidates across retries."""
    if not isinstance(raw, dict):
        return json.dumps(raw)

    def normalise(value):
        if isinstance(value, dict):
            out = {k: normalise(v) for k, v in value.items() if k != "uses_previous"}
            if value.get("kind") in {"and", "or"} and isinstance(out.get("children"), list):
                out["children"] = sorted(out["children"], key=lambda item: json.dumps(item, sort_keys=True))
            if value.get("operator") in {"in", "not_in"} and isinstance(out.get("value"), list):
                out["value"] = sorted(out["value"], key=str)
            return out
        if isinstance(value, list):
            return [normalise(item) for item in value]
        return value

    return json.dumps(normalise(raw), sort_keys=True, default=str)


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


@dataclass
class _Turn:
    """Everything one question carries through the pipeline stages of QueryService._handle."""

    question: str
    session_id: str | None
    query_id: str
    trace: StepTrace
    started: float
    budget_s: float  # one model-time budget per question, counted from arrival (Settings.llm_request_budget_seconds)
    today: str
    today_d: date
    mask: Any
    scan: DateScan
    structure: Any
    session_key: str | None
    previous: dict | None
    previous_filters: dict
    previous_json: str | None
    failure_key: str  # a failed interpretation is tied to the exact conversation context
    refers_back: bool  # a follow-up ("of them"): plan templates are never used or stored for it
    template_view: QuestionView
    template_compat: Compatibility | None
    template_id: str | None = None
    usage: ModelUsage = field(default_factory=ModelUsage)
    calls: "ModelCalls" = field(default_factory=lambda: ModelCalls())
    tier: str = "llm"
    raw: Any = None
    request_ir: Any = None
    ir_status: str = "not_used"
    uses_previous: bool = False
    fallback_reason: str = ""  # why the deterministic planner did not answer this question (observability)
    brief: str | None = None  # the analysis brief the local model gets (nlu/brief.py)
    model_trace: list = field(default_factory=list)  # each model RequestIR and its deterministic compiled plan
    intent: Any = None
    retried: bool = False
    retry_info: dict = field(default_factory=dict)
    # the last RequestIR compile failure: {code, details, capability, quotes, retryable}; drives the retry feedback and
    # the backend-written message when no executable plan is left (never the model's own wording)
    compile_failure: dict | None = None
    timing: dict = field(default_factory=lambda: {"resolve_ms": 0, "validate_ms": 0, "retry_ms": 0})

    def budget_left(self) -> float:
        return self.budget_s - (time.perf_counter() - self.started)


class QueryService:
    def __init__(
        self,
        repo: ResidentRepository,
        translator: QueryTranslator | None,
        settings: Settings,
        gate: TopicGate | None = None,
        intent_router: SemanticIntentRouter | None = None,
        semantic_cache: SemanticPlanCache | None = None,
    ):
        self.repo = repo
        self.semantic_cache = semantic_cache
        self.translator = translator
        # Off-server model (LLM_PROVIDER=bedrock)? Drives externalLlmCalls and the planner/response labels only.
        self._external = bool(getattr(translator, "external", False))
        self._model_label = "bedrock_gemma" if self._external else "local_qwen"
        self.settings = settings
        self.gate = gate
        self.intent_router = intent_router
        # Optional: the local model rewrites the template summary (fact-checked; template kept on any miss).
        self.phraser = (
            ResponsePhraser(translator, settings.llm_phrasing_timeout_seconds, settings.llm_phrasing_max_tokens)
            if translator is not None and settings.llm_response_phrasing
            else None
        )
        self.direct_builder = (
            DirectQueryBuilder(intent_router.config)
            if intent_router is not None and settings.direct_answers_enabled
            else None
        )
        routes = intent_router.config if intent_router is not None else _load_routes(settings)
        self.routes = routes
        # Question structure (subjects, references, comparisons): shared by the pre-check below and the validator.
        self.structure = StructureAnalyzer(routes)
        # Stage 1: comparisons with another resident (or among residents) are not supported yet. They are answered
        # with a clear message before any planner runs, so they never cost a model call and never execute.
        self.precheck_comparisons = True
        direct_cfg = routes.get("direct", {})
        self.default_number_field = direct_cfg.get("default_number_field")
        self.lookup_attribute_words = direct_cfg.get("lookup_attributes", {})
        # Date-field anchor words, shared by the deterministic planner and the grounding check (one source of truth).
        self.missing_detector = MissingValueDetector(routes)
        self.reference_resolver = ReferenceResolver(routes)
        self.projection_detector = ProjectionDetector(routes)
        self.date_anchors: dict[str, list[str]] = {
            f: [str(w).lower() for w in words] for f, words in (direct_cfg.get("date_fields") or {}).items()
        }
        # Questions whose model plan failed validation: asking the same thing in the same context again costs no
        # model calls. Keyed by an exact fingerprint (incl. conversation context, schema/vocabulary version, data scope,
        # model identity), with the conversation lifetime. (The exact-fingerprint SUCCESSFUL-plan cache was replaced by
        # the semantic template cache on 2026-11-18; this failure cache is a different mechanism and stays.)
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

    def _template_compat(self) -> Compatibility:
        """What a stored plan template must match to be reusable here. The vocabulary part covers only the values a
        template's meaning depends on (statuses, care types); residents, rooms, facilities and caregivers are slots,
        so adding one does not invalidate every template."""
        s = self.settings
        vocab = hashlib.sha256(
            json.dumps([sorted(self.index.statuses), sorted(self.index.care_types)]).encode()
        ).hexdigest()[:16]
        return Compatibility(
            request_ir_version=_REQUEST_IR_VERSION,
            vocabulary_hash=vocab,
            schema_hash=_SCHEMA_HASH,
            model_version=s.model_version_tag,
            embedding_model=s.topic_gate_model,
        )

    @property
    def cache_context(self) -> str:
        """Everything besides the question and the conversation that decides how a question is interpreted."""
        s = self.settings
        scope = f"{s.data_source}:{s.mongodb_db}.{s.mongodb_collection}"  # the data scope (tenant scope: not built yet)
        model = s.model_version_tag
        return f"{_REQUEST_IR_VERSION}|{self.vocab_version}|{scope}|{model}"

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
        received = time.perf_counter()
        trace = StepTrace(query_id, log)
        trace.record("receive_question", "received", chars=len(question), in_conversation=bool(session_id))
        try:
            result = await self._handle(question, session_id, query_id, trace)
        except Exception as exc:
            trace.finish("failed", error=type(exc).__name__, failure_stage=trace.current_step)
            log.info("query completed", extra={"query_id": query_id, "outcome": "failed", "error": type(exc).__name__,
                                               "total_ms": round((time.perf_counter() - received) * 1000)})  # fmt: skip
            raise
        # One summary line per question: total time from receiving the question to the finished response (every stage,
        # including model calls, retries and the database), plus where the time went. No question text or names.
        response = result.get("response") or {}
        log.info(
            "query completed",
            extra={
                "query_id": query_id,
                "total_ms": round((time.perf_counter() - received) * 1000),
                "tier": result.get("tier"),
                "outcome": "clarified" if response.get("needsClarification") else "answered",
                "model_calls": result.get("modelCallsAttempted"),
                "model_timeouts": result.get("modelCallsTimedOut"),
                "fallback_reason": reason_code(result.get("fallbackReason")),  # category only: details can quote names
            },
        )
        return result

    async def _handle(self, question: str, session_id: str | None, query_id: str, trace: StepTrace) -> dict[str, Any]:
        """The pipeline as named stages (the step numbers are app/engine/steps.py)."""
        t = self._prepare(question, session_id, query_id, trace)  # 02 mask · 03 structure · 04 memory
        stopped = await self._screen(t)  # 05 topic · 06 policy
        if stopped is not None:
            return stopped
        await self._plan_deterministic(t)  # 07 comparison · 08 intent · 09 fast path
        await self._plan_from_template(t)  # 10 semantic cache
        self._plan_from_failure_cache(t)  # 11 failure cache
        self._compile_direct(t)
        stopped = await self._plan_with_model(t)  # 12 local model
        if stopped is not None:
            return stopped
        outcome, plan, g, grounding_problems = await self._resolve_and_validate(t)  # 13 resolve · 14 validate · 15 retry
        return await self._answer(t, outcome, plan, g, grounding_problems)  # 16 execute · 17 answer · 18 remember · 19

    # ------------------------------------------------------------------ stages
    def _prepare(self, question: str, session_id: str | None, query_id: str, trace: StepTrace) -> _Turn:
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

        session_key = _session_cache_key(session_id, self.cache_context) if session_id else None
        previous_entry = self.sessions.get(session_key) if session_key else None
        previous = previous_entry.get("request_ir") if previous_entry else None
        previous_plan = previous_entry.get("plan") if previous_entry else None
        previous_filters = dict((previous_plan or {}).get("filters") or {}) if previous_plan else {}
        previous_json = json.dumps(previous, sort_keys=True, separators=(",", ":")) if previous else None
        trace.record("conversation_memory", "previous_plan" if previous else "none",
                     previous_action=(previous or {}).get("action"))  # fmt: skip
        return _Turn(
            question=question, session_id=session_id, query_id=query_id, trace=trace, started=started,
            budget_s=self.settings.llm_request_budget_seconds or self.settings.llm_timeout_seconds,
            today=today, today_d=today_d, mask=mask, scan=scan, structure=structure, session_key=session_key,
            previous=previous, previous_filters=previous_filters, previous_json=previous_json,
            failure_key=_cache_key(today, mask.text, previous_json, self.cache_context),
            refers_back=self.reference_resolver.resolve(mask.text, previous is not None).refers_to_previous,
            template_view=QuestionView(
                mask.text, list(mask.placeholders), scan, self.index, self.date_anchors, dict(mask.placeholders)
            ),
            template_compat=self._template_compat() if self.semantic_cache is not None else None,
        )  # fmt: skip

    async def _screen(self, t: _Turn) -> dict[str, Any] | None:
        """Topic check and policy check, before any planning. A finished result when the question stops here."""
        trace = t.trace
        # Questions that mention a specific resident are about residents by definition, so they skip the topic check.
        if self.gate and not t.mask.placeholders and not _RESIDENT_ID.search(t.question):
            trace.start("topic_check")
            verdict = await asyncio.to_thread(self.gate.check, t.question)
            trace.record("topic_check", "blocked" if verdict.blocked else "passed", method="embedding (bge-small)",
                         resident_score=round(verdict.resident_score, 3), off_topic_score=round(verdict.off_topic_score, 3))  # fmt: skip
            if verdict.blocked:
                result = self._gated(t.query_id, t.question, t.started)
                trace.finish("off_topic", tier="gate", model_calls=0)
                return result
        else:
            trace.skip("topic_check", "topic check disabled" if not self.gate else "names a resident")
        # Policy check: topics the resident records do not hold are refused deterministically (no model call).
        oos = out_of_scope_terms(t.question)
        trace.record("policy_check", "refused" if oos else "passed", topics=oos or None)
        if oos:
            result = self._out_of_scope(t.query_id, t.question, t.started, oos)
            trace.finish("refused_by_policy", tier="gate", model_calls=0)
            return result
        return None

    async def _plan_deterministic(self, t: _Turn) -> None:
        """Comparison pre-check, intent classifier and the deterministic fast path (no model)."""
        trace, mask, question = t.trace, t.mask, t.question
        if self.precheck_comparisons and t.structure.is_comparison:
            kind = "reference" if t.structure.references else "cross_record"
            t.raw, t.tier = {"action": "clarify", "message": self.structure.messages[kind]}, "direct"
            t.fallback_reason = f"unsupported:comparison:{kind}"
            trace.record("comparison_check", "not_supported", kind=kind)
        else:
            trace.record("comparison_check", "passed" if not t.structure.is_comparison else "precheck_off")
        if t.raw is None and self.intent_router and self.intent_router.ready:
            # Classify the KIND of question on the skeleton (date/age phrases removed): they carry no intent
            # signal but would pull the embedding away from the route examples and drop the router's score.
            route_text = grammar_skeleton(mask.text, t.today_d)
            trace.start("intent_classifier")
            t.intent = await asyncio.to_thread(self.intent_router.score, route_text)
            intent = t.intent
            trace.record("intent_classifier", "scored" if intent else "no_score", method="embedding (bge-small)",
                         route=intent.route if intent else None,
                         confidence=round(intent.confidence, 3) if intent else None,
                         margin=round(intent.margin, 3) if intent else None)  # fmt: skip
            if intent:
                log.info(
                    "semantic router score",
                    extra={
                        "query_id": t.query_id,
                        "route": intent.route,
                        "confidence": intent.confidence,
                        "margin": intent.margin,
                        "scores": intent.scores,
                    },
                )
        else:
            trace.skip("intent_classifier", "already decided" if t.raw is not None else "classifier not ready")
        # Deterministic planner (no model): a confident route + database values → plan. It must pass the same
        # resolver and grounding check as a model plan; anything uncertain goes on to the local model.
        intent = t.intent
        if intent and self.direct_builder:
            trace.start("fast_path")
            direct = self.direct_builder.build(
                question, mask, intent, self.index, t.today_d, has_previous=t.previous is not None
            )
            reason = direct.reason
            if direct.plan:
                try:
                    t.request_ir = request_ir_from_plan(direct.plan, mask.text)
                    _, direct_plan = compile_request_ir(
                        t.request_ir, mask.text, self.index.statuses, self.index.care_types
                    )
                    t.uses_previous = bool(direct_plan.pop("uses_previous", False))
                    check = resolve_plan(direct_plan, self.index, mask, question)
                    grounded = None if check.problem else self._ground(
                        check.plan, question, mask, t.scan, t.previous_filters if t.previous else None,
                        t.uses_previous, t.structure,
                    )
                except IRCompilationError as exc:
                    check = None
                    grounded = None
                    reason = f"ir_compile:{exc}"
                t.ir_status = "compiled" if grounded is not None else "rejected"
                if grounded is not None and grounded.ok:
                    t.raw, t.tier = direct_plan, "direct"
                elif grounded is not None and grounded.blocking:
                    # the question itself cannot be answered as asked (e.g. emergency contacts for many residents,
                    # an unresolvable "them"): no model can change that, so ask now
                    t.raw, t.tier = {"action": "clarify", "message": _clarify_message(grounded)}, "direct"
                    reason = "blocking:" + "; ".join(grounded.blocking)
                else:
                    reason = (
                        f"resolve:{check.problem}"
                        if check is not None and check.problem
                        else "grounding:" + "; ".join(grounded.problems())
                        if grounded is not None
                        else reason
                    )
            if t.tier != "direct":
                t.fallback_reason = reason
            trace.record(
                "fast_path",
                ("answered" if t.raw.get("action") != "clarify" else "clarified") if t.tier == "direct" else "fallback_to_model",
                reason=None if t.tier == "direct" else reason_code(reason),
                chosen_by=direct.details.get("chosen_by"),
            )  # fmt: skip
            extra = {
                "query_id": t.query_id,
                "route": direct.details.get("route", intent.route),
                "confidence": direct.details.get("confidence", intent.confidence),
                "margin": direct.details.get("margin", intent.margin),
                "chosen_by": direct.details.get("chosen_by", "router"),
            }
            if t.tier == "direct":
                log.info("direct answer via MongoDB MCP (no language model)", extra={**extra, "action": t.raw["action"]})
            else:
                words = [w for w in direct.details.get("words", []) if w.islower()]  # never names
                log.info(
                    "direct answer not possible; asking language model",
                    extra={**extra, "reason": reason, **({"words": words} if words else {})},
                )
        elif not t.fallback_reason and t.raw is None:
            t.fallback_reason = "no_deterministic_plan"
        trace.skip("fast_path", "already decided" if t.raw is not None else "fast path not available")
        if t.raw is None and t.fallback_reason == "missing_or_across_fields":
            # "no caregiver OR no room": the plan format has no OR across fields and no model can change that,
            # so ask right away instead of spending model calls on a plan the validator must reject.
            t.raw, t.tier = {"action": "clarify", "message": OR_ACROSS_FIELDS_MESSAGE}, "direct"

    async def _plan_from_template(self, t: _Turn) -> None:
        """Semantic plan template: a stored (not revoked) template found by meaning, every slot refilled from THIS
        question, then the same resolve_plan + check_plan as any plan. Similarity only nominates; any doubt → Qwen."""
        trace = t.trace
        if t.raw is None and self.semantic_cache is not None and not t.refers_back:

            async def template_fits(candidate: dict) -> str | None:
                out = resolve_plan(candidate, self.index, t.mask, t.question)
                if out.problem:
                    return f"resolve:{reason_code(out.problem)}"
                checked = self._ground(out.plan, t.question, t.mask, t.scan, None, False, t.structure)
                if not checked.ok:  # counts only: the problem texts can quote names or values
                    return (f"grounding:missing={len(checked.missing)},invented={len(checked.invented)},"
                            f"blocking={len(checked.blocking)}")  # fmt: skip
                return None

            trace.start("semantic_cache")
            hit = await self.semantic_cache.lookup(t.template_view, t.template_compat, template_fits)
            rejected: dict[str, int] = {}
            for c in hit.candidates:
                if c["verdict"] != "passed":
                    key = c["verdict"].split(":")[0]
                    rejected[key] = rejected.get(key, 0) + 1
            trace.record(
                "semantic_cache", "hit" if hit.plan else "miss", method="bge-small + Atlas Vector Search",
                reason=hit.reason or None, template_id=hit.template_id,
                similarity=round(hit.similarity, 4) if hit.similarity is not None else None,
                best_similarity=max((c["similarity"] for c in hit.candidates), default=None),
                candidates=len(hit.candidates), rejected=rejected or None,
                embed_ms=hit.embed_ms, search_ms=hit.search_ms, check_ms=hit.validate_ms,
            )  # fmt: skip
            if hit.plan:
                t.raw, t.tier, t.template_id = hit.plan, "template", hit.template_id
                t.request_ir = request_ir_from_plan(t.raw, t.mask.text)
                t.ir_status = "template"
        else:
            trace.skip("semantic_cache", "already decided" if t.raw is not None
                       else "follow-up question" if t.refers_back else "semantic cache disabled")  # fmt: skip

    def _plan_from_failure_cache(self, t: _Turn) -> None:
        if t.raw is None and (failed := self.failures.get(t.failure_key)):
            # same question, same conversation, same schema and model: the model already failed on it
            t.model_trace.append({"cached_failure": True, "problems": failed["problems"]})
            t.raw, t.tier = {"action": "clarify", "message": failed["message"]}, "cache"
            t.trace.record("failure_cache", "hit")
        elif t.raw is None:
            t.trace.record("failure_cache", "miss")
        else:
            t.trace.skip("failure_cache", "already decided")

    def _compile_direct(self, t: _Turn) -> None:
        """A deterministic plan without a RequestIR yet (comparison/OR-across messages) goes through the compiler too."""
        if t.raw is not None and t.tier == "direct" and t.request_ir is None:
            try:
                t.request_ir = request_ir_from_plan(t.raw, t.mask.text)
                _, t.raw = compile_request_ir(t.request_ir, t.mask.text, self.index.statuses, self.index.care_types)
                t.uses_previous = bool(t.raw.pop("uses_previous", False))
                t.ir_status = "compiled"
            except IRCompilationError:
                t.request_ir = None
                t.ir_status = "rejected"
                t.raw = {"action": "clarify", "message": "I couldn't safely represent that request. Please rephrase it."}

    async def _plan_with_model(self, t: _Turn) -> dict[str, Any] | None:
        """The local model writes a RequestIR when nothing earlier planned the question. A result when it stops here."""
        trace = t.trace
        if t.raw is not None:
            trace.skip("local_model", "already decided")
            return None
        if self.translator is None:
            trace.record("local_model", "not_configured")
            trace.finish("model_unavailable", tier="unavailable", model_calls=0)
            return self._unavailable(
                t.query_id,
                t.question,
                t.started,
                "This question needs the local query model, which is not configured on this server. Simpler "
                "questions still work, e.g. 'How many active residents are in memory care?'.",
                request_ir_status=t.ir_status,
            )
        t.brief = self._question_brief(t.question, t.mask, t.intent, t.today_d)
        trace.start("local_model")
        try:
            translation = await self.translator.translate(
                self._request(t.mask.text, t.today, t.previous_json, t.scan, brief=t.brief, timeout_s=t.budget_left())
            )
        except TranslatorUnavailable as exc:
            self.stats.llm_errors += 1
            self._count_call(t.calls, str(exc))
            log.warning("local model unavailable: %s", exc)
            t.model_trace.append({"attempt": 1, "error": str(exc)})
            trace.record("local_model", "timeout" if str(exc) == "timeout" else "unavailable", error=str(exc),
                         model=getattr(self.translator, "model_name", None),
                         timeout_s=self.settings.llm_timeout_seconds)  # fmt: skip
            trace.finish("model_unavailable", tier="unavailable", model_calls=t.calls.succeeded, **t.calls.trace())
            return self._unavailable(
                t.query_id,
                t.question,
                t.started,
                "The query model is still starting up (this takes a few minutes after a restart). Please ask again "
                "shortly; simple questions are answered meanwhile."
                if str(exc) == "warming_up"
                else "I couldn't interpret that question right now. Please try again in a moment or rephrase it "
                "more simply.",
                model_trace=t.model_trace,
                fallback_reason=t.fallback_reason,
                calls=t.calls,
                request_ir_status=t.ir_status,
            )
        self._count_call(t.calls, None)
        t.request_ir = translation.request_ir
        t.usage, t.tier = translation.usage, "llm"
        self.stats.add(translation.usage)  # a retry below adds its own usage (_retry_once)
        failure = self._compile_model_ir(t, t.request_ir, attempt=1)
        if failure is not None and failure.retryable and self.translator is not None:
            # Recoverable: the validator says exactly what is wrong. One retry with that feedback (same budget and
            # identical-retry protections as every retry) before anything is shown to the user.
            await self._retry_once(t, _compile_hint(failure), failure.details, trigger=f"compile:{failure.code}")
        t.uses_previous = bool(t.raw.pop("uses_previous", False)) if isinstance(t.raw, dict) else False
        trace.record("local_model", "interpreted" if t.ir_status == "compiled" else "rejected",
                     model=getattr(self.translator, "model_name", None),
                     inference_ms=translation.usage.inference_ms, prompt_tokens=translation.usage.prompt_tokens,
                     completion_tokens=translation.usage.completion_tokens,
                     action=t.raw.get("action") if isinstance(t.raw, dict) else None)  # fmt: skip
        return None

    def _resolve(self, t: _Turn):
        started = time.perf_counter()
        out = resolve_plan(t.raw, self.index, t.mask, t.question)
        t.timing["resolve_ms"] += round((time.perf_counter() - started) * 1000)
        return out

    def _validate(self, t: _Turn, plan: QueryPlan):
        started = time.perf_counter()
        out = self._ground(
            plan, t.question, t.mask, t.scan, t.previous_filters if t.previous else None, t.uses_previous, t.structure,
        )
        t.timing["validate_ms"] += round((time.perf_counter() - started) * 1000)
        return out

    async def _retry_once(self, t: _Turn, hint: str, feedback: list[str], trigger: str = "validation") -> bool:
        """One more model attempt with feedback (never more than one per question). True when a plan came back."""
        if t.retried:  # the one retry is already spent (compile, role or grounding feedback): never a second one
            t.model_trace.append({"retry_skipped": True, "reason": "retry already used"})
            return False
        t.retried = True
        t.retry_info.update(trigger=trigger, problem_count=len(feedback))
        self.stats.completeness_retries += 1
        # problem counts only: the problem texts can quote resident or staff names
        log.warning("model plan rejected; retrying", extra={"query_id": t.query_id, "problem_count": len(feedback)})
        started = time.perf_counter()
        left = t.budget_left()
        if left < RETRY_MIN_BUDGET_S:  # the request's budget is spent: a doomed retry would only load the CPU
            t.model_trace.append({"attempt": 2, "error": "deadline", "feedback": feedback})
            t.retry_info.update(error="deadline", skipped=True)
            log.info("retry skipped: request budget spent", extra={"query_id": t.query_id, "left_s": round(left, 2)})
            return False
        try:
            retry_ir = (
                json.dumps(t.request_ir, sort_keys=True, separators=(",", ":")) if isinstance(t.request_ir, dict) else None
            )
            retry = await self.translator.translate(
                self._request(
                    t.mask.text,
                    t.today,
                    t.previous_json,
                    t.scan,
                    hint=hint,
                    brief=t.brief,
                    retry_ir_json=retry_ir,
                    timeout_s=left,
                )
            )
        except TranslatorUnavailable as exc:
            self.stats.llm_errors += 1
            self._count_call(t.calls, str(exc))
            t.model_trace.append({"attempt": 2, "error": str(exc), "feedback": feedback})
            t.retry_info["error"] = str(exc)
            t.timing["retry_ms"] += round((time.perf_counter() - started) * 1000)
            return False
        t.timing["retry_ms"] += round((time.perf_counter() - started) * 1000)
        self._count_call(t.calls, None)
        retry_ir_value = retry.request_ir
        failure = self._compile_model_ir(t, retry_ir_value, attempt=2, feedback=feedback)
        if t.model_trace and _normalised_plan(retry_ir_value) == _normalised_plan(t.model_trace[0].get("requestIR")):
            t.model_trace[-1]["identical_to_previous"] = True
            t.retry_info["identical_plan"] = True
            log.info("retry produced the same RequestIR", extra={"query_id": t.query_id})
        if failure is not None:
            t.retry_info["compile_error"] = failure.code
        self.stats.add(retry.usage)
        t.usage = t.usage + retry.usage
        if t.ir_status != "compiled":  # compiled: _compile_model_ir already stored the normalised IR
            t.request_ir = retry_ir_value
        t.uses_previous = bool(t.raw.pop("uses_previous", False)) if isinstance(t.raw, dict) else False
        return True

    async def _resolve_and_validate(self, t: _Turn):
        """Resolve exact database values, run the grounding check, retry the model once when its mistake is fixable.
        Returns (resolve outcome, final plan, grounding result or None, grounding problems)."""
        trace = t.trace
        outcome = self._resolve(t)
        # A resident used as a caregiver/facility/room is a planning error a model can correct (the resident exists;
        # "couldn't find" would be wrong): one retry with that feedback.
        if outcome.problem and outcome.problem.startswith(("role:", "invented:")) and t.tier == "llm" and outcome.hint:
            trigger = "wrong_role" if outcome.problem.startswith("role:") else "invented_reference"
            if await self._retry_once(t, outcome.hint, [outcome.problem], trigger=trigger):
                outcome = self._resolve(t)
        plan = outcome.plan
        grounding_problems: list[str] = []
        g = None
        if not outcome.problem:
            g = self._validate(t, plan)
            # Model plan failed the grounding check: retry once with the problems as feedback — but only when the
            # problems are ones a model can correct (a dropped or invented condition/field). Blocking problems
            # (unsupported structure, conversation context, schema/policy limits) go straight to the user.
            if not g.ok and t.tier == "llm" and not g.correctable:
                t.model_trace.append({"retry_skipped": True, "reason": g.blocking})
                log.info("model plan failed with a non-correctable problem; not retrying", extra={"query_id": t.query_id})
            if not g.ok and t.tier == "llm" and g.correctable and not t.retried:
                if await self._retry_once(t, g.hint(), g.problems()):
                    outcome = self._resolve(t)
                    plan = outcome.plan
                    if not outcome.problem:
                        g = self._validate(t, plan)
            if not outcome.problem:
                if g.ok:
                    plan = g.plan
                    if g.fixes:
                        log.info("plan canonicalised", extra={"query_id": t.query_id, "fixes": g.fixes})
                else:
                    grounding_problems = g.problems()
                    plan = QueryPlan(action="clarify", message=_clarify_message(g))
                    if t.tier == "llm":  # remember: the same question in the same context must not cost model calls again
                        self.failures.set(t.failure_key, {"problems": grounding_problems, "message": plan.message})

        # What the user reads is always written by the backend, never by the model:
        #   • a comparison question gets the "not supported yet" explanation, whatever stopped it;
        #   • a model that answered "clarify"/"unsupported" by itself gets a fixed request to rephrase (its own wording
        #     can be wrong — "not a valid resident" for a resident that exists twice — or leak placeholders).
        if t.structure.is_comparison and not plan.is_data_query:
            kind = "reference" if t.structure.references else "cross_record"
            plan = QueryPlan(action="clarify", message=self.structure.messages[kind])
            t.fallback_reason = t.fallback_reason or f"unsupported:comparison:{kind}"
        elif (
            t.tier in ("llm", "cache")
            and not outcome.problem
            and not grounding_problems
            and plan.action in ("clarify", "unsupported")
        ):
            t.model_trace.append({"model_message_replaced": True, "action": plan.action})
            plan = QueryPlan(action="clarify", message=_compile_failure_message(t.compile_failure, t.mask))

        trace.record("resolve_values", "ok" if not outcome.problem else "clarify", duration_ms=t.timing["resolve_ms"],
                     problem=reason_code(outcome.problem), action=plan.action)  # fmt: skip
        if g is None:
            trace.skip("validate_plan", "values did not resolve")
        else:
            trace.record("validate_plan", "passed" if g.ok else "rejected", duration_ms=t.timing["validate_ms"],
                         missing=len(g.missing) or None, invented=len(g.invented) or None,
                         blocking=len(g.blocking) or None, fixes=len(g.fixes) or None)  # fmt: skip
        if t.retried:
            trace.record("model_retry", "retried", duration_ms=t.timing["retry_ms"], **t.retry_info)
        elif g is not None and not g.ok and t.tier == "llm" and not g.correctable:
            trace.record("model_retry", "skipped", reason="not correctable by a model")
        else:
            trace.skip("model_retry", "not needed")
        if outcome.problem:
            log.info("plan resolution: %s", outcome.problem)
        return outcome, plan, g, grounding_problems

    async def _answer(self, t: _Turn, outcome, plan: QueryPlan, g, grounding_problems: list[str]) -> dict[str, Any]:
        """Execute the read-only query, build the answer, remember (template + conversation) and log the result."""
        trace = t.trace
        # A plan becomes a reusable template only when it came from the fast path or the model (not from a template),
        # compiled, resolved and passed validation unchanged, is a standalone data question — and only AFTER it
        # executed and answered (below). There is no review step (decided 2026-11-18): every later reuse is re-validated
        # against the new question, and a wrong template can be switched off with `revoked: true`.
        template_candidate = (
            self.semantic_cache is not None
            and t.tier in ("direct", "llm")
            and t.ir_status == "compiled"
            # a text match ("starts with A") is not a slot the template cache can refill: storing it would keep "A"
            # literally and could serve it for "starts with B". Such plans are never stored as templates.
            and not plan.filters.has_text_match
            and not outcome.problem
            and not grounding_problems
            and g is not None
            and g.ok
            and plan.is_data_query
            and not t.uses_previous
            and not t.refers_back
        )
        # Model-written text (a clarifying question or "not available" reason) may contain the resident_N placeholders
        # the model saw; show the user the names they actually typed instead.
        if plan.message and t.mask.originals:
            plan.message = _unmask(plan.message, t.mask.originals)

        trace.start("execute_query")
        exec_started = time.perf_counter()
        data = await execute(plan, self.repo)
        db_ms = round((time.perf_counter() - exec_started) * 1000)
        trace.record("execute_query", "ok" if plan.is_data_query else "no_query", duration_ms=db_ms, action=plan.action,
                     matched=data.matched, rows=len(data.rows) + len(data.looked_up or []))  # fmt: skip
        trace.start("build_answer")
        response = build_response(plan, data)
        phrasing = None
        if self.phraser is not None:
            response, phrasing = await self.phraser.phrase(t.question, plan, response, total=data.total)
        trace.record("build_answer", "clarification" if response.get("needsClarification") else "answered",
                     phrasing=phrasing.status if phrasing else None, phrasing_reason=phrasing.reason if phrasing else None,
                     phrasing_ms=phrasing.ms if phrasing else None)  # fmt: skip
        template_recorded: bool | None = None
        template_skip: str | None = None
        if template_candidate and not response.get("needsClarification"):  # executed and answered
            # the RESOLVED plan (exact database values; resident ids are mapped back to the question's placeholders):
            # validation passed it unchanged (_ground blocks any rewrite), so it is exactly what was executed
            template_skip = self.semantic_cache.record(
                plan.model_dump(mode="json", exclude_none=True), t.template_view, t.template_compat,
                source="fast_path" if t.tier == "direct" else "local_model",
            )  # fmt: skip
            template_recorded = template_skip is None
        saved = bool(t.session_id and plan.is_data_query)
        if saved and t.session_key:
            self.sessions.set(
                t.session_key,
                {
                    "plan": plan.model_dump(mode="json", exclude_none=True),
                    "request_ir": t.request_ir,
                },
            )
        trace.record("remember", "ok", template_recorded=template_recorded,
                     template_skip=template_skip.split(":")[0] if template_skip else None,
                     template_used=t.template_id, conversation_saved=saved)  # fmt: skip

        latency = round((time.perf_counter() - t.started) * 1000)
        self.stats.queries += 1
        self.stats.by_tier[t.tier] += 1
        documents_fetched = len(data.rows) + len(data.looked_up or [])
        log.info(
            "query handled",
            extra={
                "query_id": t.query_id,
                "selectedTier": "deterministic" if t.tier in ("cache", "direct", "template") else t.tier,
                "plannerName": {"direct": "direct_query", "cache": "failure_cache", "template": "semantic_template",
                                "llm": self._model_label}.get(t.tier, t.tier),
                "plannerDecision": "answered" if t.tier in ("direct", "cache", "template") else "fallback",
                "fallbackReason": t.fallback_reason or None,
                "modelCalls": t.usage.calls,
                **t.calls.api(),
                "externalLlmCalls": t.calls.attempted if self._external else 0,
                "inferenceMs": t.usage.inference_ms,
                "resolvedAction": plan.action,
                "resolvedFilters": plan.model_dump(mode="json", exclude_none=True).get("filters") or {},
                "validationResult": outcome.problem or ("grounding" if grounding_problems else "ok"),
                "databaseExecutionTimeMs": db_ms,
                "documentsFetched": documents_fetched,
                "responseGenerationMethod": f"template+{self._model_label}" if phrasing and phrasing.status == "phrased" else "template",
                "matched": data.matched,
                "latency_ms": latency,
            },
        )  # fmt: skip
        trace.finish(
            "clarified" if response.get("needsClarification") else "answered",
            tier=t.tier, model_calls=t.usage.calls, matched=data.matched, **t.calls.trace(),
        )
        return self._result(
            t.query_id, t.question, plan, data.matched, data.total, response, t.tier, t.usage, latency,
            model_trace=t.model_trace, fallback_reason=t.fallback_reason, calls=t.calls,
            request_ir_status=t.ir_status,
            validation_result="passed" if g and g.ok else "rejected" if g else "not_run",
            response_generation="phrased" if phrasing and phrasing.status == "phrased" else "template",
        )

    def _count_call(self, calls: ModelCalls, error: str | None) -> None:
        calls.record(error)
        self.stats.model_calls_attempted += 1
        if self._external:
            self.stats.external_llm_calls += 1
        if error == "timeout":
            self.stats.model_calls_timed_out += 1

    def _compile_model_ir(
        self, t: _Turn, ir: Any, attempt: int, feedback: list[str] | None = None
    ) -> IRCompilationError | None:
        """Normalise (meaning-preserving rewrites only), then compile one model RequestIR with the strict validator.

        Success: t.raw / t.request_ir / t.ir_status = "compiled", t.compile_failure cleared. Failure: t.raw becomes a
        clarification, t.compile_failure keeps the exact error, and the trace records code, details and capability.
        """
        normalized, rewrites = normalize_request_ir(ir) if isinstance(ir, dict) else (ir, [])
        try:
            _, plan = compile_request_ir(normalized, t.mask.text, self.index.statuses, self.index.care_types)
        except IRCompilationError as exc:
            self._trace_model(t.query_id, t.model_trace, attempt, ir, t.mask, feedback=feedback,
                              error="unexecutable RequestIR")  # fmt: skip
            t.model_trace[-1]["compileError"] = {
                "code": exc.code, "details": exc.details, "retryable": exc.retryable, "capability": exc.capability,
            }  # fmt: skip
            if rewrites:
                t.model_trace[-1]["normalized"] = rewrites
            t.compile_failure = {"code": exc.code, "details": exc.details, "capability": exc.capability,
                                 "quotes": exc.quotes, "retryable": exc.retryable}  # fmt: skip
            t.raw = {"action": "clarify", "message": "I couldn't safely represent every part of that request."}
            t.ir_status = "rejected"
            log.info("model RequestIR rejected", extra={"query_id": t.query_id, "attempt": attempt,
                                                         "code": exc.code, "retryable": exc.retryable})  # fmt: skip
            return exc
        self._trace_model(t.query_id, t.model_trace, attempt, ir, t.mask, feedback=feedback, compiled_plan=plan)
        if rewrites:
            t.model_trace[-1]["normalized"] = rewrites
        t.raw, t.request_ir, t.ir_status, t.compile_failure = plan, normalized, "compiled", None
        return None

    def _trace_model(
        self, query_id: str, trace: list[dict], attempt: int, raw: Any, mask,
        feedback: list[str] | None = None, compiled_plan: dict | None = None, error: str | None = None,
    ) -> None:
        """Record the model's RequestIR and, when possible, its deterministic QueryPlan compilation.

        Resident references are redacted from server logs; the structured trace retains the masked model output.
        """
        entry: dict[str, Any] = {"attempt": attempt, "requestIR": copy.deepcopy(raw)}
        if compiled_plan is not None:
            entry["plan"] = copy.deepcopy(compiled_plan)
        if error:
            entry["error"] = error
        if feedback:
            entry["feedback"] = feedback  # the problems the retry was asked to fix
        trace.append(entry)
        log.info(
            "local model RequestIR generated",
            extra={
                "query_id": query_id,
                "attempt": attempt,
                "request_ir": _redact_for_log(raw),
                "compiled": compiled_plan is not None,
                **({"feedback_count": len(feedback)} if feedback else {}),
            },
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
        """Facilities named in the question, plus those in a previous RequestIR."""
        wings = [w for values in self.index.mentioned_wings(masked).values() for w in values]
        if previous_json:
            try:
                ir = json.loads(previous_json)

                def walk(node):
                    if not isinstance(node, dict):
                        return
                    if canonical_field(node.get("field")) == "facility":
                        value = node.get("value")
                        wings.extend(value if isinstance(value, list) else [value] if value else [])
                    for child in node.get("children") or []:
                        walk(child)
                    walk(node.get("child"))

                walk(ir.get("where"))
            except (ValueError, AttributeError):
                pass
        return list(dict.fromkeys(wings))

    def _request(
        self, masked: str, today: str, previous_json: str | None, scan: DateScan, hint: str | None = None,
        brief: str | None = None, retry_ir_json: str | None = None, timeout_s: float | None = None,
    ) -> TranslationRequest:
        context = self.schema_context(self._relevant_wings(masked, previous_json))
        hints = scan.hints() + [
            f"'{masked[n.start : n.end]}' = age {' '.join(f'{k} {v}' for k, v in n.bounds.items())}" for n in scan.numbers
        ]
        return TranslationRequest(
            masked, today, context, previous_json, hints, hint, brief, retry_ir_json, timeout_s=timeout_s
        )

    def _question_brief(self, question: str, mask, intent, today_d: date) -> str | None:
        """The analysis brief for the local model (nlu/brief.py); None when disabled or nothing was understood."""
        if not self.settings.llm_question_brief or self.direct_builder is None:
            return None
        try:
            return self.direct_builder.describe(question, mask, intent, self.index, today_d).to_text()
        except Exception as exc:  # the brief is a help, never a reason to fail the question
            log.warning("question brief failed: %s", type(exc).__name__)
            return None

    def _ground(
        self, plan: QueryPlan, question: str, mask, scan: DateScan, previous_filters: dict | None, uses_previous=False,
        structure=None,
    ):
        compiled = plan.model_dump(mode="json", exclude_none=True)
        result = check_plan(
            plan, question, mask, self.index, scan, self.date_anchors, previous_filters, uses_previous,
            missing_detector=self.missing_detector, reference_resolver=self.reference_resolver,
            projection_detector=self.projection_detector, structure=structure,
        )
        if result.plan.model_dump(mode="json", exclude_none=True) != compiled:
            result.blocking.append("backend validation would change the compiled RequestIR plan")
        return result

    # ------------------------------------------------------------------ response shaping
    def _result(
        self, query_id, question, plan: QueryPlan, matched, total, response, tier, usage: ModelUsage, latency,
        model_trace: list[dict] | None = None, fallback_reason: str | None = None, calls: ModelCalls | None = None,
        request_ir_status: str = "not_run", validation_result: str = "not_run",
        response_generation: str = "template",
    ) -> dict[str, Any]:
        calls = calls or ModelCalls()
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
            "modelCalls": usage.calls,  # completed calls; attempts and timeouts are counted separately below
            **calls.api(),
            "model": getattr(self.translator, "model_name", None) if usage.calls or calls.attempted else None,
            "externalLlmCalls": calls.attempted if self._external else 0,
            # Why the deterministic planner did not answer (empty when it did), and every plan the local model
            # generated for this question, as generated (attempt 2 = the retry, with the feedback it was given).
            "fallbackReason": fallback_reason or None,
            "modelPlans": model_trace or [],
            "failureAttribution": {
                "model": (
                    "timeout" if calls.timed_out else "failed" if calls.failed else
                    "interpreted" if calls.succeeded else "not_used"
                ),
                "requestIR": request_ir_status,
                "validation": validation_result,
                "execution": "completed" if is_data else "not_run",
                "responseGeneration": response_generation,
            },
            "tokensSaved": 0,
            "pipelineLatencyMs": latency,
            "policy": {
                "passed": plan.action != "unsupported",
                "policyCategory": "OUT_OF_SCOPE" if plan.action == "unsupported" else "RESIDENT_INTENT",
                "violationReason": plan.message if plan.action == "unsupported" else None,
            },
            "intent": {
                "category": _CATEGORY[plan.action],
                "confidence": 1.0 if tier in ("cache", "direct", "template") else 0.9,
                "matchedPattern": " · ".join(interpretation) or plan.action,
                "parameters": plan_json,
                # Deterministic = resolved with no generative model: a direct planner answer or a cached plan.
                "isDeterministic": tier in ("cache", "direct", "template"),
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
        model_trace: list[dict] | None = None, fallback_reason: str | None = None, calls: ModelCalls | None = None,
        request_ir_status: str = "not_run",
    ) -> dict[str, Any]:
        self.stats.queries += 1
        self.stats.by_tier["unavailable"] += 1
        plan = QueryPlan(action="clarify", message=message)
        response = {"headline": "Assistant unavailable", "summaryText": message, "count": 0, "needsClarification": True}
        latency = round((time.perf_counter() - started) * 1000)
        return self._result(
            query_id, question, plan, 0, 0, response, "unavailable", ModelUsage(), latency,
            model_trace=model_trace, fallback_reason=fallback_reason, calls=calls,
            request_ir_status=request_ir_status,
            validation_result="not_run",
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
    """Keep structure for diagnostics while redacting evidence, condition values and entity names."""
    if not isinstance(raw, dict):
        return "<not a JSON object>"

    def scrub(value, key: str | None = None):
        if key in {"evidence", "message", "reason"} and isinstance(value, str):
            return f"<{len(value)} characters>"
        if key in {"value", "include", "exclude"}:
            if isinstance(value, list):
                return ["<value>" for _ in value]
            return "<value>" if value is not None else None
        if key == "residents" and isinstance(value, list):
            return [r if isinstance(r, str) and _PLACEHOLDER.match(r) else "<resident>" for r in value]
        if isinstance(value, dict):
            return {k: scrub(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        return value

    return scrub(raw)


def _unmask(text: str, originals: dict[str, str]) -> str:
    """resident_1 / Resident_1 → the name the user typed for that placeholder."""
    return re.sub(
        r"\bresident_(\d+)\b",
        lambda m: originals.get(f"resident_{m.group(1)}", m.group(0)),
        text,
        flags=re.IGNORECASE,
    )
