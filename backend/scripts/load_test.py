"""Concurrent-request scenario against the full QueryService and a running llama-server.

    LLM_SERVER_URL=... LLM_MAX_CONCURRENCY=<slots> LLM_CHAT_TEMPLATE_KWARGS=... \
        uv run python -m scripts.load_test --users 6 --per-user 5 --llama-pid <pid>

N simulated staff members each ask M questions back-to-back, all users at the same time. Questions are the gold
"complex" phrasings (the ones that need the model), so every request exercises local inference. Production settings
are used (LLM_TIMEOUT_SECONDS default 15 s, concurrency capped at LLM_MAX_CONCURRENCY): a request that cannot get a
slot in time, or times out, is reported as unavailable — the user sees a controlled message, not a hang.
"""

import argparse
import asyncio
import json
import statistics
import time

from app.data.memory_repository import MemoryResidentRepository
from app.engine.service import QueryService
from app.nlu.intent_router import SemanticIntentRouter
from app.nlu.qwen_translator import QwenTranslator
from app.nlu.topic_gate import TopicGate
from scripts.benchmark import MODELS, ProcSampler, Settings_
from tests.data.gold_questions import GOLD


async def main(users: int, per_user: int, llama_pid: int | None, label: str) -> dict:
    settings = Settings_(_env_file=None, environment="test", cache_max_entries=0, topic_gate_cache_dir=MODELS)
    gate = TopicGate(settings.topic_gate_margin, model_name=settings.topic_gate_model, cache_dir=MODELS)
    router = SemanticIntentRouter(cache_dir=MODELS)
    await gate.start()
    await router.start()
    translator = QwenTranslator(settings)
    service = QueryService(MemoryResidentRepository(), translator, settings, gate=gate, intent_router=router)
    await service.refresh_vocabulary()
    pool = [q for g in GOLD if g["kind"] == "complex" and not g.get("followup_of") for q in g["q"]]

    results: list[dict] = []

    async def user(u: int):
        for i in range(per_user):
            q = pool[(u * per_user + i) % len(pool)]
            t = time.perf_counter()
            r = await service.handle(q, session_id=f"load-{u}")
            results.append({"ms": (time.perf_counter() - t) * 1000, "tier": r["tier"], "calls": r["modelCalls"]})

    sampler = ProcSampler(llama_pid)
    sampler.start()
    t0 = time.perf_counter()
    await asyncio.gather(*(user(u) for u in range(users)))
    wall = time.perf_counter() - t0
    proc = sampler.stop()
    lat = sorted(x["ms"] for x in results)
    model_lat = sorted(x["ms"] for x in results if x["calls"])
    report = {
        "label": label,
        "users": users,
        "requests": len(results),
        "slots": settings.llm_max_concurrency,
        "timeout_s": settings.llm_timeout_seconds,
        "wall_s": round(wall, 1),
        "throughput_req_per_s": round(len(results) / wall, 2),
        "latency_ms_p50": round(statistics.median(lat)),
        "latency_ms_p95": round(lat[max(0, int(len(lat) * 0.95) - 1)]),
        "latency_ms_max": round(lat[-1]),
        "model_routed": len(model_lat),
        "model_routed_p95_ms": round(model_lat[max(0, int(len(model_lat) * 0.95) - 1)]) if model_lat else None,
        "unavailable_timeouts_or_overload": sum(x["tier"] == "unavailable" for x in results),
        "model_errors": service.stats.llm_errors,
        "llama_server_process": proc,
    }
    await translator.close()
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--users", type=int, default=6)
    p.add_argument("--per-user", type=int, default=5)
    p.add_argument("--llama-pid", type=int)
    p.add_argument("--label", default="")
    a = p.parse_args()
    asyncio.run(main(a.users, a.per_user, a.llama_pid, a.label))
