#!/usr/bin/env bash
# Development launcher for the local model (same flags as deploy/). Binds to 127.0.0.1 only.
#   scripts/fetch_model.sh ~/.local/share/sashi-llm/models ~/.local/share/sashi-llm/bin   # once
#   scripts/run_llm_server.sh                                   # Qwen3-0.6B, Q8_0 KV cache (default)
#   LLM_MODEL_FILE=qwen2.5-0.5b-instruct-q4_k_m.gguf LLM_CACHE_TYPE=f16 scripts/run_llm_server.sh   # previous model
set -euo pipefail
ROOT="${SASHI_LLM_HOME:-$HOME/.local/share/sashi-llm}"
export LD_LIBRARY_PATH="$ROOT/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$ROOT/bin/llama-server" \
  --model "$ROOT/models/${LLM_MODEL_FILE:-Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf}" \
  --host 127.0.0.1 --port "${LLM_PORT:-8081}" \
  --ctx-size "${LLM_CTX_SIZE:-4096}" --parallel "${LLM_PARALLEL:-2}" \
  --threads "${LLM_THREADS:-4}" --threads-batch "${LLM_THREADS:-4}" \
  --cache-type-k "${LLM_CACHE_TYPE:-q8_0}" --cache-type-v "${LLM_CACHE_TYPE:-q8_0}" \
  --n-predict "${LLM_N_PREDICT:-256}" --timeout "${LLM_SERVER_TIMEOUT:-30}" \
  --reasoning "${LLM_REASONING:-off}" --reasoning-format deepseek \
  --no-webui --no-slots --metrics --offline \
  ${LLM_EXTRA_ARGS:-}   # e.g. "-ctk q8_0 -ctv q8_0" (quantized KV cache; enables flash attention)
