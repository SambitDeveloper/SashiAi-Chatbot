# Local inference server: llama.cpp `llama-server` (CPU), pinned release, checksum-verified.
# The model weights are NOT baked in: they are fetched once by backend/scripts/fetch_model.sh into a persistent host
# directory (e.g. an EBS volume at /opt/sashi/models) and mounted read-only at /models.
FROM ubuntu:24.04

ARG LLAMA_TAG=b12017
ARG LLAMA_SHA256=16fe348b1bc2e943bc4c220938ce46af789e88291138bca5bd68c0cc017769a5

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL -o /tmp/llama.tgz \
      "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-bin-ubuntu-x64.tar.gz" \
 && echo "${LLAMA_SHA256}  /tmp/llama.tgz" | sha256sum -c - \
 && mkdir -p /opt/llama && tar -xzf /tmp/llama.tgz -C /opt/llama --strip-components=1 && rm /tmp/llama.tgz \
 && useradd --system --uid 10002 llm

USER llm
ENV LD_LIBRARY_PATH=/opt/llama
EXPOSE 8081
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s CMD curl -fsS http://127.0.0.1:8081/health || exit 1
ENTRYPOINT ["/opt/llama/llama-server"]
