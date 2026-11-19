#!/usr/bin/env bash
# Development launcher for the local model (same flags as deploy/). Binds to 127.0.0.1 only.
#   scripts/fetch_model.sh ~/.local/share/sashi-llm/models ~/.local/share/sashi-llm/bin   # once
#   scripts/run_llm_server.sh                                   # Qwen3-4B-Instruct-2507, Q8_0 KV cache (default)
#
# Defaults tuned on the 4-core dev CPU (i5-1335U, 2026-11-18/19):
#   • --parallel 1: one request gets the whole CPU. Each slot gets LLM_CTX_PER_SLOT tokens (--ctx-size is split between
#     slots): the prompt is ~1,820 tokens + up to 512 generated, which did NOT fit the 2,048 a 4096/2-slot setup gave.
#   • --threads 4 (generating: 4.1 tokens/s vs 3.4 at 8) and --threads-batch 8 (reading the prompt: 15 vs 12 tokens/s).
#   • --timeout 300: a cold or long request must not be cut by the server's own HTTP timeout.
#   Match the backend: LLM_MAX_CONCURRENCY = LLM_PARALLEL.
#   LLM_MODEL_FILE=qwen2.5-0.5b-instruct-q4_k_m.gguf LLM_CACHE_TYPE=f16 scripts/run_llm_server.sh   # previous model
set -euo pipefail
ROOT="${SASHI_LLM_HOME:-$HOME/.local/share/sashi-llm}"
export LD_LIBRARY_PATH="$ROOT/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$ROOT/bin/llama-server" \
  --model "$ROOT/models/${LLM_MODEL_FILE:-Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf}" \
  --host 127.0.0.1 --port "${LLM_PORT:-8081}" \
  --ctx-size "${LLM_CTX_SIZE:-$(( ${LLM_CTX_PER_SLOT:-4096} * ${LLM_PARALLEL:-1} ))}" --parallel "${LLM_PARALLEL:-1}" \
  --threads "${LLM_THREADS:-4}" --threads-batch "${LLM_THREADS_BATCH:-8}" \
  --cache-type-k "${LLM_CACHE_TYPE:-q8_0}" --cache-type-v "${LLM_CACHE_TYPE:-q8_0}" \
  --n-predict "${LLM_N_PREDICT:-512}" --timeout "${LLM_SERVER_TIMEOUT:-300}" \
  --reasoning "${LLM_REASONING:-off}" --reasoning-format deepseek \
  --no-webui --no-slots --metrics --offline \
  ${LLM_EXTRA_ARGS:-}   # e.g. "-ctk q8_0 -ctv q8_0" (quantized KV cache; enables flash attention)
