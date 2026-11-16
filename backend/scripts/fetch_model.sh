#!/usr/bin/env bash
# Controlled artifact fetch for the local query-translation model and (optionally) the llama.cpp server binary.
#
# Run ONCE per host / image build — never at application startup. Every artifact is pinned to an exact revision and
# verified against its SHA-256; a mismatch deletes the file and fails the script.
#
#   scripts/fetch_model.sh <model_dir>                 # model only (Docker deployment uses the official server image)
#   scripts/fetch_model.sh <model_dir> <bin_dir>       # model + llama-server Linux x64 CPU build (bare-metal / dev)
#   MODEL=qwen2.5-0.5b scripts/fetch_model.sh <model_dir> # the previous model (Qwen2.5-0.5B-Instruct)
set -euo pipefail

MODEL_DIR="${1:?usage: fetch_model.sh <model_dir> [bin_dir]}"
BIN_DIR="${2:-}"

# --- Models (select with MODEL=...; default qwen3-4b-2507, previous: qwen3-0.6b) -------------------------------------------------------
MODEL="${MODEL:-qwen3-4b-2507}"
case "$MODEL" in
  qwen2.5-0.5b)  # Qwen2.5-0.5B-Instruct, GGUF Q4_K_M, official Qwen repository, Apache-2.0
    MODEL_REPO="Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    MODEL_REVISION="9217f5db79a29953eb74d5343926648285ec7e67"
    MODEL_FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"
    MODEL_SHA256="74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
    MODEL_BYTES=491400032 ;;
  qwen3-0.6b)    # Qwen3-0.6B, GGUF Q4_K_M (lmstudio-community conversion of Qwen/Qwen3-0.6B; Qwen publishes only Q8_0), Apache-2.0
    MODEL_REPO="lmstudio-community/Qwen3-0.6B-GGUF"
    MODEL_REVISION="3334d820ab76652cf6e242d7c6302b10f0951f23"
    MODEL_FILE="Qwen3-0.6B-Q4_K_M.gguf"
    MODEL_SHA256="cd47557a67d7e8f2891d98b5e1dbf2988544569fdf4f1bdb30e92b71aa61b548"
    MODEL_BYTES=484219808 ;;
  qwen3-1.7b)    # Qwen3-1.7B, GGUF Q4_K_M (lmstudio-community conversion of Qwen/Qwen3-1.7B), Apache-2.0
    MODEL_REPO="lmstudio-community/Qwen3-1.7B-GGUF"
    MODEL_REVISION="e5e31bf4d96de5da2ce124fa86673f0be7c82346"
    MODEL_FILE="Qwen3-1.7B-Q4_K_M.gguf"
    MODEL_SHA256="e0801cbda7e2f3fd00bea4d73b53b422b14b13aa130e778f6414b6b641920b7e"
    MODEL_BYTES=1282439328 ;;
  lfm2.5-1.2b)   # EXPERIMENTAL: Liquid AI LFM2.5-1.2B-Instruct, GGUF Q4_K_M (official LiquidAI repo), LFM Open License v1.0
    MODEL_REPO="LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
    MODEL_REVISION="8ed288026e23958ad9dfa92d53ed773a8eee7125"
    MODEL_FILE="LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
    MODEL_SHA256="b1b3de114215d9507409a662a501a631095a479a419584e8a2ded6304b19b4f5"
    MODEL_BYTES=730895168 ;;
  qwen3-4b-2507) # DEFAULT: Qwen3-4B-Instruct-2507 (non-thinking only), GGUF Q4_K_M imatrix by bartowski, Apache-2.0
    MODEL_REPO="bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF"
    MODEL_REVISION="ae44f08e1392f39c0e474af10c3ff8355c8b6688"
    MODEL_FILE="Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    MODEL_SHA256="2fde00ce69dd4899c70d020845e2638353015bba0fdf161b3eb965f2bca4464e"
    MODEL_BYTES=2497280736 ;;
  qwen3-4b-thinking-2507) # EXPERIMENTAL: Qwen3-4B-Thinking-2507 (thinking-only), GGUF Q4_K_M imatrix by bartowski, Apache-2.0
    MODEL_REPO="bartowski/Qwen_Qwen3-4B-Thinking-2507-GGUF"
    MODEL_REVISION="ba7f9bc071caf4788e3d7a5963543cff0149e483"
    MODEL_FILE="Qwen_Qwen3-4B-Thinking-2507-Q4_K_M.gguf"
    MODEL_SHA256="64ce5838a76b4d63071289c17ed04525265a47abf51546cd747a30e41b7687fa"
    MODEL_BYTES=2497280736 ;;
  *) echo "unknown MODEL=$MODEL (qwen2.5-0.5b | qwen3-0.6b | qwen3-1.7b | lfm2.5-1.2b | qwen3-4b-2507 | qwen3-4b-thinking-2507)" >&2; exit 2 ;;
esac

# --- llama.cpp server (CPU, AVX2, Ubuntu x64 build) --------------------------------------------------------------
LLAMA_TAG="b12017"
LLAMA_ASSET="llama-${LLAMA_TAG}-bin-ubuntu-x64.tar.gz"
LLAMA_SHA256="16fe348b1bc2e943bc4c220938ce46af789e88291138bca5bd68c0cc017769a5"

verify() {  # file sha256
  local got
  got="$(sha256sum "$1" | cut -d' ' -f1)"
  if [[ "$got" != "$2" ]]; then
    echo "CHECKSUM MISMATCH for $1: expected $2, got $got" >&2
    rm -f "$1"
    exit 1
  fi
}

mkdir -p "$MODEL_DIR"
target="$MODEL_DIR/$MODEL_FILE"
if [[ -f "$target" && "$(stat -c %s "$target")" == "$MODEL_BYTES" ]]; then
  echo "model present: $target (verifying)"
else
  curl -fL --retry 3 -o "$target.part" \
    "https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}/${MODEL_FILE}"
  mv "$target.part" "$target"
fi
verify "$target" "$MODEL_SHA256"
echo "model OK: $target ($MODEL_BYTES bytes, sha256 $MODEL_SHA256)"

if [[ -n "$BIN_DIR" ]]; then
  mkdir -p "$BIN_DIR"
  archive="$BIN_DIR/$LLAMA_ASSET"
  if [[ ! -x "$BIN_DIR/llama-server" ]]; then
    curl -fL --retry 3 -o "$archive" \
      "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/${LLAMA_ASSET}"
    verify "$archive" "$LLAMA_SHA256"
    tar -xzf "$archive" -C "$BIN_DIR" --strip-components=1
    rm -f "$archive"
  fi
  echo "llama-server OK: $BIN_DIR/llama-server ($LLAMA_TAG)"
fi
