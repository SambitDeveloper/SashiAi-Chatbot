"""Benchmark the query planners on the gold set (tests/data/gold_questions.py: 50 questions × 3 phrasings).

    uv run python -m scripts.benchmark --mode deterministic            # A: deterministic planner only (no model)
    uv run python -m scripts.benchmark --mode qwen --llama-pid <pid>   # B: deterministic + local Qwen (llama-server)

Runs the real QueryService (real BGE topic gate + semantic router, memory repository with tests/data/residents.json,
Today pinned to 2026-11-09, plan cache disabled so every question is planned cold). For every question both the
EXPECTED plan and the ACTUAL plan are executed, and the answer signatures (counts, ordered ids, groups, aggregates)
are compared — accuracy is measured on database results, not on JSON validity.
"""

import argparse
import asyncio
import json
import os
import statistics
import threading
import time
from collections import Counter
from pathlib import Path

from app.config import Settings
from app.data.memory_repository import MemoryResidentRepository
from app.engine.executor import execute
from app.engine.service import QueryService
from app.models.plan import QueryPlan
from app.nlu.intent_router import SemanticIntentRouter
from app.nlu.topic_gate import TopicGate
from app.nlu.translator import TranslationRequest, TranslatorUnavailable
from tests.data.eval_50 import EVAL
from tests.data.gold_questions import GOLD, TODAY
from tests.data.paraphrases import PARA

DATASETS = {"gold": GOLD, "eval50": EVAL, "para100": PARA}

MODELS = str(Path(__file__).resolve().parents[1] / ".models")
DATA_ACTIONS = {"count", "list", "breakdown", "aggregate", "lookup"}


class Settings_(Settings):
    def today(self) -> str:
        return TODAY


class CountingTranslator:
    """Wraps the real translator to count calls, failures by kind and inference time."""

    def __init__(self, inner):
        self.inner = inner
        self.provider = inner.provider
        self.model_name = inner.model_name
        self.calls = 0
        self.errors: Counter = Counter()
        self.inference_ms: list[int] = []
        self.max_prompt_tokens = 0
        self.max_completion_tokens = 0
        self.max_total_tokens = 0

    async def health(self):
        return await self.inner.health()

    async def translate(self, request: TranslationRequest):
        self.calls += 1
        try:
            t = await self.inner.translate(request)
        except TranslatorUnavailable as exc:
            self.errors[str(exc)] += 1
            raise
        self.inference_ms.append(t.usage.inference_ms)
        u = t.usage
        self.max_prompt_tokens = max(self.max_prompt_tokens, u.prompt_tokens)
        self.max_completion_tokens = max(self.max_completion_tokens, u.completion_tokens)
        self.max_total_tokens = max(self.max_total_tokens, u.prompt_tokens + u.completion_tokens)
        return t


# ---------------------------------------------------------------------------------------------- scoring
def atoms(plan: dict) -> set[tuple]:
    out = set()
    for field, body in (plan.get("filters") or {}).items():
        for key, val in body.items():
            if key in ("include", "exclude"):
                out |= {(field, key, v) for v in val}
            else:
                out.add((field, key, str(val)))
    return out


def canon(plan: dict) -> dict:
    return {
        "action": plan.get("action"),
        "filters": atoms(plan),
        "group_by": frozenset(plan.get("group_by") or []),
        "aggregate": (plan.get("aggregate") or {}).get("op"),
        "sort": tuple((plan.get("sort") or {}).values()) or None,
        "limit": plan.get("limit"),
        "residents": frozenset(r.upper() for r in (plan.get("lookup") or {}).get("residents", [])),
    }


async def signature(plan: dict, repo) -> tuple | None:
    if plan.get("action") not in DATA_ACTIONS:
        return None
    qp = QueryPlan.model_validate(plan)
    d = await execute(qp, repo)
    if qp.action == "count":
        return ("count", d.matched)
    if qp.action == "list":
        ids = [r["residentId"] for r in d.rows]
        return ("list", d.matched, tuple(ids) if qp.sort else tuple(sorted(ids)))
    if qp.action == "breakdown":
        return ("breakdown", tuple(sorted((g.key, g.count) for g in d.groups)))
    if qp.action == "aggregate":
        if d.group_values is not None:
            return ("agg", tuple(sorted((g.key, None if g.value is None else round(g.value, 2)) for g in d.group_values)))
        return ("agg", None if d.aggregate_value is None else round(d.aggregate_value, 2))
    return ("lookup", tuple(r["residentId"] for r in d.looked_up), tuple(d.checks or []))


# ---------------------------------------------------------------------------------------------- resources
class ProcSampler:
    """Samples a process's RSS and CPU time from /proc while the benchmark runs."""

    def __init__(self, pid: int | None):
        self.pid = pid
        self.rss_kb: list[int] = []
        self.cpu_samples: list[tuple[float, float]] = []  # (wall time, process CPU seconds) for peak CPU
        self._stop = threading.Event()
        self.cpu_start = self._cpu()
        self.t_start = time.perf_counter()

    def _cpu(self) -> float:
        if not self.pid:
            return 0.0
        fields = Path(f"/proc/{self.pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")

    def _status(self, key: str) -> int:
        for line in Path(f"/proc/{self.pid}/status").read_text().splitlines():
            if line.startswith(key):
                return int(line.split()[1])
        return 0

    def start(self):
        if not self.pid:
            return
        def loop():
            while not self._stop.is_set():
                self.rss_kb.append(self._status("VmRSS:"))
                self.cpu_samples.append((time.perf_counter(), self._cpu()))
                time.sleep(0.1)
        threading.Thread(target=loop, daemon=True).start()

    def stop(self) -> dict:
        self._stop.set()
        if not self.pid:
            return {}
        wall = time.perf_counter() - self.t_start
        cpu = self._cpu() - self.cpu_start
        return {
            "rss_mb_avg": round(statistics.mean(self.rss_kb) / 1024, 1),
            "rss_mb_peak_sampled": round(max(self.rss_kb) / 1024, 1),
            "vm_hwm_mb": round(self._status("VmHWM:") / 1024, 1),
            "cpu_seconds": round(cpu, 1),
            "wall_seconds": round(wall, 1),
            "avg_cores_busy": round(cpu / wall, 2),
            "peak_cores_busy_1s": self._peak_cores(),
        }

    def _peak_cores(self) -> float | None:
        """Highest CPU use over any 1-second window (cores busy; 4.0 = four cores fully used)."""
        s, best, j = self.cpu_samples, 0.0, 0
        for i in range(len(s)):
            while j < i and s[i][0] - s[j][0] > 1.0:
                j += 1
            if s[i][0] - s[j][0] >= 0.9:
                best = max(best, (s[i][1] - s[j][1]) / (s[i][0] - s[j][0]))
        return round(best, 2) if s else None


def self_rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return round(int(line.split()[1]) / 1024, 1)
    return 0.0


# ---------------------------------------------------------------------------------------------- run
async def run(
    mode: str, llama_pid: int | None, out: Path | None, limit: int | None, label: str = "", dataset: str = "gold",
    model_only: bool = False,
) -> dict:
    data = DATASETS[dataset]
    settings = Settings_(
        _env_file=None,
        environment="test",
        cache_max_entries=0,
        topic_gate_cache_dir=MODELS,
        llm_timeout_seconds=float(os.environ.get("LLM_TIMEOUT_SECONDS", "30")),
    )
    repo = MemoryResidentRepository()
    gate = TopicGate(settings.topic_gate_margin, model_name=settings.topic_gate_model, cache_dir=MODELS)
    router = SemanticIntentRouter(cache_dir=MODELS)
    await gate.start()
    await router.start()
    assert gate.ready and router.ready, "embedding models did not load"

    translator = None
    if mode == "qwen":
        from app.nlu.qwen_translator import QwenTranslator

        translator = CountingTranslator(QwenTranslator(settings))
        assert await translator.health(), f"llama-server not ready at {settings.llm_server_url}"
    service = QueryService(repo, translator, settings, gate=gate, intent_router=router)
    await service.refresh_vocabulary()
    if model_only:  # every question reaches the model: no deterministic fast path, no comparison pre-check.
        service.direct_builder = None  # The validator, resolver, retry and caching rules are unchanged.
        service.precheck_comparisons = False

    warm_up_ms = None
    if translator is not None:  # identical warm-up for every configuration (same as the API's startup warm-up)
        t = time.perf_counter()
        await translator.inner.warm_up(service.schema_context(), TODAY)
        warm_up_ms = round((time.perf_counter() - t) * 1000)
        translator.calls, translator.errors, translator.inference_ms = 0, Counter(), []  # exclude warm-up from stats
        translator.max_prompt_tokens = translator.max_completion_tokens = translator.max_total_tokens = 0

    sampler = ProcSampler(llama_pid)
    sampler.start()
    rows = []
    items = [(g, i, q) for g in data for i, q in enumerate(g["q"])][:limit]
    for n, (g, variant, question) in enumerate(items):
        session = f"bench-{n}"
        if g.get("followup_of"):
            await service.handle(g["followup_of"], session_id=session)
        calls_before = translator.calls if translator else 0
        inf_before = len(translator.inference_ms) if translator else 0
        result = await service.handle(question, session_id=session)
        model_calls = (translator.calls - calls_before) if translator else 0
        actual = result["intent"]["parameters"]
        exp, act = canon(g["plan"]), canon(actual)
        exp_sig = await signature(g["plan"], repo)
        act_sig = await signature(actual, repo) if result["tier"] != "unavailable" else None
        exp_data = g["plan"]["action"] in DATA_ACTIONS
        act_action = actual.get("action")
        if exp_data:
            outcome = (
                "correct" if act_sig == exp_sig
                else "clarify" if act_action == "clarify" or result["tier"] == "unavailable"
                else "wrong_execution" if act_action in DATA_ACTIONS
                else "wrongly_refused"
            )
        else:
            outcome = (
                "correct" if act_action == g["plan"]["action"]
                else "wrong_execution" if act_action in DATA_ACTIONS
                else "clarify" if act_action == "clarify" or result["tier"] == "unavailable"
                else "other"
            )  # fmt: skip
        date_atoms = {a for a in exp["filters"] if a[0].endswith("_date")}
        entity_atoms = {a for a in exp["filters"] if a[0] in ("primary_caregiver", "wing")}
        rows.append(
            {
                "id": g["id"], "variant": variant, "kind": g["kind"], "question": question,
                "tier": result["tier"], "model_calls": model_calls, "external_calls": result.get("externalLlmCalls", 0),
                "latency_ms": result["pipelineLatencyMs"], "tokens": result["tokensConsumed"],
                "inference_ms": sum(translator.inference_ms[inf_before:]) if translator else 0,
                "prompt_tokens": (result.get("tokenUsage") or {}).get("prompt_tokens", 0),
                "completion_tokens": (result.get("tokenUsage") or {}).get("completion_tokens", 0),
                "expected": g["plan"], "actual": actual, "outcome": outcome,
                "exact_plan": exp == act,
                "action_ok": exp["action"] == act["action"],
                "tp": len(exp["filters"] & act["filters"]), "fp": len(act["filters"] - exp["filters"]),
                "fn": len(exp["filters"] - act["filters"]),
                "has_negation": any(a[1] == "exclude" for a in exp["filters"]),
                "has_date": bool(date_atoms),
                "has_age": any(a[0] == "age" for a in exp["filters"]),
                # joined data: caregiver names come from `staffs`, wing names from `facilities` (resident_schema.yaml)
                "has_relationship": bool(entity_atoms) or bool(set(g["plan"].get("group_by") or []) & {"primary_caregiver", "wing"}),
                "is_followup": bool(g.get("followup_of")),
                "date_ok": (date_atoms <= act["filters"] and not {a for a in act["filters"] if a[0].endswith("_date")} - date_atoms) if date_atoms else None,
                "entity_ok": (entity_atoms == {a for a in act["filters"] if a[0] in ("primary_caregiver", "wing")} and exp["residents"] == act["residents"]) if (entity_atoms or exp["residents"]) else None,
            }
        )  # fmt: skip
        print(f"{n + 1:3}/{len(items)} [{result['tier']:11}] {outcome:15} {question[:70]}", flush=True)
    resources = sampler.stop()

    def rate(xs):
        xs = [x for x in xs if x is not None]
        return round(100 * sum(xs) / len(xs), 1) if xs else None

    def pct(xs, n):
        return round(100 * xs / n, 1) if n else None

    def summary(sub: list[dict]) -> dict:
        if not sub:
            return {"n": 0}
        tp, fp, fn = (sum(r[k] for r in sub) for k in ("tp", "fp", "fn"))
        lat = [r["latency_ms"] for r in sub]
        data_rows = [r for r in sub if r["expected"]["action"] in DATA_ACTIONS]
        not_correct = [r for r in sub if r["outcome"] != "correct"]

        def acc(flag):
            rows_ = [r for r in sub if r[flag]]
            return {"n": len(rows_), "pct": rate([r["outcome"] == "correct" for r in rows_])}

        exp_clarify = [r for r in sub if r["expected"]["action"] == "clarify"]
        exp_refuse = [r for r in sub if r["expected"]["action"] == "unsupported"]
        return {
            "latency_ms_mean": round(statistics.mean(lat)),
            "deterministic_success_pct": rate([r["tier"] == "direct" and r["outcome"] == "correct" for r in sub]),
            "correct_clarification": {"n": len(exp_clarify), "pct": rate([r["outcome"] == "correct" for r in exp_clarify])},
            "correct_refusal": {"n": len(exp_refuse), "pct": rate([r["outcome"] == "correct" for r in exp_refuse])},
            "incorrect_answer_pct": rate([r["outcome"] == "wrong_execution" for r in sub]),
            "wrongly_refused_pct": rate([r["outcome"] == "wrongly_refused" for r in sub]),
            "input_tokens": sum(r["prompt_tokens"] for r in sub),
            "output_tokens": sum(r["completion_tokens"] for r in sub),
            "missing_filter_rate_pct": pct(sum(r["fn"] > 0 for r in data_rows), len(data_rows)),
            "invented_filter_rate_pct": pct(sum(r["fp"] > 0 for r in data_rows), len(data_rows)),
            "negation_accuracy": acc("has_negation"),
            "date_range_accuracy": acc("has_date"),
            "age_comparison_accuracy": acc("has_age"),
            "relationship_accuracy": acc("has_relationship"),
            "followup_accuracy": acc("is_followup"),
            # of the answers that were not correct, how many were safe (clarified / refused, nothing wrong executed)
            "safe_clarification_rate_pct": pct(sum(r["outcome"] != "wrong_execution" for r in not_correct), len(not_correct)),
            "incorrect_executions": sum(r["outcome"] == "wrong_execution" for r in sub),
            "n": len(sub),
            "answer_accuracy_pct": rate([r["outcome"] == "correct" for r in sub]),
            "exact_plan_pct": rate([r["exact_plan"] for r in sub]),
            "action_accuracy_pct": rate([r["action_ok"] for r in sub]),
            "filter_precision_pct": round(100 * tp / (tp + fp), 1) if tp + fp else None,
            "filter_recall_pct": round(100 * tp / (tp + fn), 1) if tp + fn else None,
            "date_accuracy_pct": rate([r["date_ok"] for r in sub]),
            "entity_accuracy_pct": rate([r["entity_ok"] for r in sub]),
            "incorrect_execution_pct": rate([r["outcome"] == "wrong_execution" for r in sub]),
            "clarification_pct": rate([r["outcome"] == "clarify" for r in sub]),
            "zero_model_call_pct": rate([r["model_calls"] == 0 for r in sub]),
            "model_calls": sum(r["model_calls"] for r in sub),
            "external_llm_calls": sum(r["external_calls"] for r in sub),
            "latency_ms_p50": statistics.median(lat),
            "latency_ms_p95": sorted(lat)[max(0, int(len(lat) * 0.95) - 1)],
            "latency_ms_p99": sorted(lat)[max(0, int(len(lat) * 0.99) - 1)],
            "tiers": dict(Counter(r["tier"] for r in sub)),
            "outcomes": dict(Counter(r["outcome"] for r in sub)),
        }

    report = {
        "mode": mode,
        "model_only": model_only,
        "overall": summary(rows),
        "label": label,
        "warm_up_ms": warm_up_ms,
        "settings": {k: getattr(settings, k) for k in (
            "llm_model_name", "llm_chat_template_kwargs", "llm_temperature", "llm_top_p", "llm_top_k", "llm_seed",
            "llm_max_tokens", "llm_timeout_seconds", "llm_max_concurrency")},
        "dataset": dataset,
        "wall_seconds_total": round(time.perf_counter() - sampler.t_start, 1),
        "by_kind": {k: summary([r for r in rows if r["kind"] == k]) for k in ("simple", "complex", "oos")},
        # the 50 base questions (first phrasing of each gold item) and the questions the model actually planned
        "base_50": summary([r for r in rows if r["variant"] == 0]),
        "model_routed": summary([r for r in rows if r["model_calls"] > 0]),
        "simple_with_model_calls": [r["question"] for r in rows if r["kind"] == "simple" and r["model_calls"]],
        "wrong_executions": [
            {"q": r["question"], "expected": r["expected"], "actual": r["actual"]}
            for r in rows if r["outcome"] == "wrong_execution"
        ],  # fmt: skip
        "model": {
            "calls": translator.calls if translator else 0,
            "errors": dict(translator.errors) if translator else {},
            # context fit: the largest request actually processed, and any truncation/overflow failures
            # (finish_reason=length → "max_tokens"; context overflow → HTTP 400 "api_error")
            "max_prompt_tokens": translator.max_prompt_tokens if translator else 0,
            "max_completion_tokens": translator.max_completion_tokens if translator else 0,
            "max_prompt_plus_output_tokens": translator.max_total_tokens if translator else 0,
            "truncation_or_overflow_errors": (translator.errors.get("max_tokens", 0) + translator.errors.get("api_error", 0))
            if translator
            else 0,
            "invalid_json_pct": round(100 * translator.errors.get("invalid_json", 0) / translator.calls, 1)
            if translator and translator.calls
            else 0.0,
            "inference_ms_mean": round(statistics.mean(translator.inference_ms)) if translator and translator.inference_ms else None,
            "timeouts": translator.errors.get("timeout", 0) if translator else 0,
            "context_overflow_or_truncation": (translator.errors.get("max_tokens", 0) + translator.errors.get("api_error", 0)) if translator else 0,
            "inference_ms_p50": statistics.median(translator.inference_ms) if translator and translator.inference_ms else None,
            "inference_ms_p95": sorted(translator.inference_ms)[int(len(translator.inference_ms) * 0.95) - 1]
            if translator and translator.inference_ms
            else None,
            "inference_ms_p99": sorted(translator.inference_ms)[max(0, int(len(translator.inference_ms) * 0.99) - 1)]
            if translator and translator.inference_ms
            else None,
        },
        "llama_server_process": resources,
        "benchmark_process_peak_rss_mb": self_rss_mb(),
        "rows": rows,
    }
    if out:
        out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1, default=str))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["deterministic", "qwen"], required=True)
    p.add_argument("--llama-pid", type=int)
    p.add_argument("--out", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--dataset", choices=list(DATASETS), default="gold")
    p.add_argument("--label", default="", help="e.g. the model name; the model itself is chosen by LLM_SERVER_URL, "
                   "LLM_MODEL_NAME and LLM_CHAT_TEMPLATE_KWARGS (same prompts, schema and validation for every model)")
    p.add_argument("--model-only", action="store_true",
                   help="send every question to the model (fast path and comparison pre-check off; validation unchanged)")
    a = p.parse_args()
    asyncio.run(run(a.mode, a.llama_pid, a.out, a.limit, a.label, a.dataset, a.model_only))


if __name__ == "__main__":
    main()
