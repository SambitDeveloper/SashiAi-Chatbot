"""Numbered step logging: one log line per pipeline step, for every question, in a fixed order.

    step-01 receive_question     step-08 comparison_check     step-15 model_retry
    step-02 mask_names           step-09 intent_classifier    step-16 execute_query
    step-03 question_structure   step-10 fast_path            step-17 build_answer
    step-04 conversation_memory  step-11 failure_cache        step-18 remember
    step-05 plan_cache           step-12 local_model          step-19 finished
    step-06 topic_check          step-13 resolve_values
    step-07 policy_check         step-14 validate_plan

Each step is logged exactly once per question: with its outcome (and how long it took when it did work), or as
`skipped` with the reason. A question that stops early (off-topic, refused, model unavailable, an error) still gets
the remaining steps logged as skipped, so every query_id has the complete 01–19 trace.

Privacy: the step lines carry only counts, categories, scores, codes and timings — never the question text, resident
or staff names, or condition values (same rule as the rest of the logs: logging_setup.py).
"""

import logging
import time
from typing import Any

STEPS: tuple[str, ...] = (
    "receive_question",  # 01 the API hands the question to the pipeline
    "mask_names",  # 02 resident names → resident_N placeholders (live vocabulary)
    "question_structure",  # 03 operation, subjects, references, comparisons (grammar only)
    "conversation_memory",  # 04 the previous plan of this chat, for follow-ups
    "plan_cache",  # 05 exact fingerprint lookup of an earlier checked plan (no embedding)
    "topic_check",  # 06 embedding model (bge-small): is this about residents at all?
    "policy_check",  # 07 topics the records do not hold (medications, billing …)
    "comparison_check",  # 08 comparisons with another resident: not supported yet
    "intent_classifier",  # 09 embedding model (bge-small): count / list / breakdown … route scores
    "fast_path",  # 10 deterministic planner, no model
    "failure_cache",  # 11 the model already failed on this exact question in this context
    "local_model",  # 12 Qwen3-4B-Instruct via llama.cpp writes a plan
    "resolve_values",  # 13 names/rooms/facilities in the plan → real database values
    "validate_plan",  # 14 semantic validation: does the plan answer THIS question?
    "model_retry",  # 15 one retry with feedback when the model's mistake is fixable
    "execute_query",  # 16 read-only MongoDB query
    "build_answer",  # 17 template answer
    "remember",  # 18 plan cache + conversation memory
    "finished",  # 19 outcome and total time
)
NUMBER = {name: i + 1 for i, name in enumerate(STEPS)}


class StepTrace:
    def __init__(self, query_id: str, logger: logging.Logger):
        self.query_id = query_id
        self.log = logger
        self.started = time.perf_counter()
        self.logged: set[str] = set()
        self._timers: dict[str, float] = {}

    def start(self, name: str) -> None:
        self._timers[name] = time.perf_counter()

    def record(self, name: str, status: str, duration_ms: int | None = None, **details: Any) -> None:
        """Log one step (once). The duration comes from start(name) unless given explicitly."""
        if name in self.logged:
            return
        if duration_ms is None and name in self._timers:
            duration_ms = round((time.perf_counter() - self._timers.pop(name)) * 1000)
        n = NUMBER[name]
        extra = {"query_id": self.query_id, "step": n, "step_name": name, "status": status}
        if duration_ms is not None:
            extra["duration_ms"] = duration_ms
        extra.update({k: v for k, v in details.items() if v is not None})
        self.logged.add(name)
        self.log.info(f"step-{n:02d} {name}: {status}", extra=extra)

    def skip(self, name: str, reason: str) -> None:
        self.record(name, "skipped", reason=reason)

    def skip_until(self, before: str, reason: str) -> None:
        """Mark every step not yet logged, up to (not including) `before`, as skipped — keeps the trace in order."""
        for name in STEPS[: NUMBER[before] - 1]:
            if name not in self.logged:
                self.skip(name, reason)

    def finish(self, outcome: str, **details: Any) -> None:
        """Log every step not reached as skipped, then step-19 with the outcome and the total time."""
        self.skip_until("finished", f"not reached ({outcome})")
        self.record("finished", outcome, total_ms=round((time.perf_counter() - self.started) * 1000), **details)


def reason_code(reason: str | None) -> str | None:
    """A fallback/problem reason reduced to its category: the part after ':' can quote names or values."""
    if not reason:
        return None
    return reason.split(":", 1)[0].strip() or None
