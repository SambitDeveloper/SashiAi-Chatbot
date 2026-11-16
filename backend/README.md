# Resident Query API (Python / FastAPI)

Staff ask questions in plain language. Each question becomes a JSON **query plan**; the backend validates the plan,
runs it against the resident database and builds the answer from the results with templates. Simple questions are
planned **deterministically, with no model call**. Only questions the deterministic planner cannot resolve safely go to
a small model running **on our own server**: Qwen2.5-0.5B-Instruct (GGUF Q4_K_M) on llama.cpp. No external AI API is
used anywhere; there are no per-token costs. The model never sees resident records and never produces a number.

```
question → mask resident names → plan cache?
         → topic check (BGE embeddings, local) / out-of-scope policy (medications, billing, …) → "not available"
         → deterministic planner: semantic router + grammar.py dates/ages + database values   (no model call)
              └─ declined → local Qwen2.5-0.5B via llama-server (grammar-constrained JSON plan)
         → SHARED VALIDATION for every plan, whichever planner produced it:
              resolver (exact DB values) → Pydantic QueryPlan → grounding check (app/engine/grounding.py):
              every named value kept, nothing invented, dates/ages/sort/limit/aggregate follow the question;
              model plans failing it are retried once, then the user is asked to clarify
         → MongoDB MCP tools (read-only) → deterministic answer template + "Interpreted as" chips
```

## Run locally

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
cd backend
uv sync                                    # install dependencies into .venv
# local model, once (pinned + SHA-256 verified; ~491 MB model + llama.cpp b12017 CPU build):
scripts/fetch_model.sh ~/.local/share/sashi-llm/models ~/.local/share/sashi-llm/bin
scripts/run_llm_server.sh                  # llama-server on 127.0.0.1:8081 (separate terminal)
uv run uvicorn app.main:app --reload --port 8000
```

In a second terminal, from the repository root: `npm run dev`, then open http://localhost:3000.
Vite forwards `/api/*` to the backend on port 8000.

If llama-server is not running, simple questions are still answered; questions that need the model get a controlled
"couldn't interpret that right now" reply (there is no fallback to any external model). `LLM_ENABLED=false` turns the
model off entirely. `curl localhost:8000/api/ready` shows `llmReady`; `/api/meta` shows `provider` and `model`.

## Data: the Care App MongoDB (DATA_SOURCE=mcp)

Every answer comes from the Care App's MongoDB. The backend starts the official MongoDB MCP Server as a child process
and runs every query through its `aggregate` tool, always with `--readOnly` (no insert/update/delete tools exist) and
telemetry disabled (the server's default is enabled).

```bash
npm install -g mongodb-mcp-server@3.3.0      # needs Node.js 20.19+, 22.13+ or 24+ installed (e.g. via nvm)
# backend/.env:  DATA_SOURCE=mcp  MONGODB_URI=mongodb+srv://...  MONGODB_DB=sal_staging  MONGODB_COLLECTION=residents
```
The Docker image already contains Node.js and `mongodb-mcp-server@3.3.0`. `DATA_SOURCE=mongodb` runs the same
queries through the PyMongo driver instead. There is no built-in data: startup fails without `MONGODB_URI`.

### Schema mapping — `app/data/resident_schema.yaml`

The assistant works with a small set of fields (status, care type, gender, age, facility, caregiver, room, dates).
The YAML file says how each one is computed from the real collection, and every query starts with it inside MongoDB:

| App field | From the Care App collection |
|---|---|
| (all) | only documents with `deletedAt: null` (soft-deleted residents never count) |
| `status` | `status` lower-cased: Active → active, Away → away, Discharged → discharged |
| `gender` | Female/F → Female, Male/M → Male, anything else (Unknown, empty) → Other |
| `age` | stored `age`, otherwise calculated from `birthDate`; values outside 0–120 count as unknown |
| `wing` | the facility: `facilities.facilityName` joined on `facilityId`, else the facility code |
| `primaryCaregiver` | names of all `assignedStaff` (joined to `staffs.cName` → `staffs.name`); a caregiver question matches if any assigned staff member matches |
| `roomNumber` | `unitNo` |
| `admissionDate` / `createdDate` / `dischargeDate` | `admissionDate` text, `createdAt` and `dischargeDate` dates, as YYYY-MM-DD |
| `residentId` | the document `_id` |

Only these fields leave the database (MongoDB's `$project` is the whitelist); medications, insurance, phone, email,
address and PCC details are never read by the app. When the database changes, edit the YAML — no code changes.
Startup stops with a clear message when the collection lacks `required_source_fields` or the mapping cannot run.

Testing the mapping locally (fixed test data written in the real Care App shape into a local MongoDB):
```bash
docker run -d --name care-mongo -p 27017:27017 mongo:7.0
uv run python -m scripts.seed_mongo mongodb://localhost:27017     # refuses non-local hosts
MCP_TEST_MONGODB_URI=mongodb://localhost:27017 uv run pytest -k mcp
```

## Direct answers (semantic router → MongoDB MCP, no language model)

`app/data/intent_routes.yaml` defines each kind of question (route) with example questions and `metadata`
(action, allowed fields, grouping/sort). A question is answered directly — 0 tokens — only when the router is
confident (`min_confidence`, `min_margin`), it contains no digits or blocker words (dates, negation, comparisons,
follow-ups), every word is explained, and every value found (status, care type, gender, wing, caregiver, resident)
comes from the database and is allowed for that route. Everything else goes to the local model. Each decision is
logged (`direct answer via MongoDB MCP` / `direct answer not possible; asking language model` with the reason).
Tune by editing the YAML — no code changes. Direct plans go through the same grounding check as model plans.

## Local model (Qwen3-0.6B on llama.cpp, non-thinking, Q8_0 KV cache)

| | Default: Qwen3-0.6B | Previous (still selectable): Qwen2.5-0.5B-Instruct |
|---|---|---|
| Artifact | `lmstudio-community/Qwen3-0.6B-GGUF` @ `3334d820ab76652cf6e242d7c6302b10f0951f23`, `Qwen3-0.6B-Q4_K_M.gguf` (Qwen publishes only Q8_0) | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` @ `9217f5db79a29953eb74d5343926648285ec7e67`, `qwen2.5-0.5b-instruct-q4_k_m.gguf` |
| Size / SHA-256 | 484,219,808 B (461.8 MiB) / `cd47557a67d7e8f2891d98b5e1dbf2988544569fdf4f1bdb30e92b71aa61b548` | 491,400,032 B (468.6 MiB) / `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` |
| License | Apache-2.0 (base `Qwen/Qwen3-0.6B` @ `c1899de2…`) | Apache-2.0 |
| GGUF checks | arch `qwen3`, 28 blocks, file_type 15 (Q4_K_M); embedded "LM Studio" chat template renders byte-identical to Qwen's official template for our prompts | arch `qwen2`, 24 blocks, Q4_K_M |
| Chat mode | non-thinking: `LLM_CHAT_TEMPLATE_KWARGS={"enable_thinking": false}` + llama-server `--reasoning off` | – |
| KV cache @ ctx 4096 | 448 MiB f16 → **238 MiB Q8_0** (default; same accuracy) | 48 MiB f16 |

Select the previous model: `LLM_MODEL_FILE=qwen2.5-0.5b-instruct-q4_k_m.gguf LLM_CACHE_TYPE=f16` for the server and
`LLM_MODEL_NAME=qwen2.5-0.5b-instruct-q4_k_m` for the API. Runtime: llama.cpp `llama-server` b12017 (CPU,
`llama-b12017-bin-ubuntu-x64.tar.gz`, SHA-256 `16fe348b…69a5`). Client: `app/nlu/qwen_translator.py` (OpenAI-compatible
`/v1/chat/completions`, JSON-Schema → GBNF constrained output, temperature 0, `cache_prompt`). On startup the API warms
every inference slot (a cold first inference measured >15 s, the request timeout).

### Thinking vs non-thinking (evaluated 2026-11-13, non-thinking kept)

Same model, Q8_0 KV, ctx 4096, same 50 questions, validator, 15 s timeout and retry policy. Thinking verified on: rendered
prompt (non-thinking injects an empty `<think></think>`), and `reasoning_content` in the output (llama.cpp
`--reasoning-format deepseek`; the JSON grammar applies only after `</think>`, thoughts are never parsed or returned).

| | Non-thinking (default) | Thinking (T 0.6, top-p 0.95, top-k 20) |
|---|---:|---:|
| Correct answers (simple / complex) | 84 % / 64–68 % | 84 % / 56 % |
| Wrong queries executed | 0 | 1 before the validator fix (exclusion without negation) |
| Model calls timed out (15 s) | 0 | 10 of 18 |
| Model inference mean / p95 | 2.2 s / 2.8 s | 12.2 s / 14.6 s (19–30 s untimed) |
| Output tokens (50 questions) | ~700 | ~1,800 |
| Peak RAM | ~1.17 GB | ~1.20 GB |

Thinking is off: `LLM_CHAT_TEMPLATE_KWARGS={"enable_thinking": false}` (default) and llama-server `--reasoning off`.

Weights are fetched once by `scripts/fetch_model.sh` (pinned revision, checksum-verified) into a persistent directory and
mounted read-only; nothing is downloaded at application start. `deploy/docker-compose.yml` runs the model in its own
container on an internal-only network with no published port; `deploy/llama-server.service` is the systemd alternative
bound to 127.0.0.1.

## Benchmark (gold set, 2026-11-13)

`uv run python -m scripts.benchmark --mode deterministic|qwen [--llama-pid PID]` runs `tests/data/gold_questions.py`
(50 questions × 3 phrasings = 150; newly authored for this migration, seeded from the test questions) through the real
service with the real BGE gate/router. Both the expected and the actual plan are **executed** and their results compared.
Measured on a 4-thread Intel i5-1335U laptop (WSL2), llama-server `--parallel 2 --threads 4`.

| Metric (150 items) | A: deterministic only | B: deterministic + local Qwen |
|---|---|---|
| Correct answers (DB result matches) | 74.7 % | 80.0 % |
| — simple questions (69) | 92.8 % | 95.7 % |
| — complex questions (69) | 52.2 % | 60.9 % |
| — out of scope (12) | 100 % | 100 % |
| **Wrong answers executed** | **0** | **0** |
| Clarification / controlled refusal | 25.3 % | 11.3 % clarify + 8.7 % wrongly refused |
| Filter precision / recall | 100 % / 72.7 % | 100 % / 72.7 % |
| Date-boundary accuracy | 90.5 % | 90.5 % |
| Simple questions with 0 model calls | 100 % | 92.8 % (5 of 69 phrasings fell through to Qwen) |
| Qwen calls / invalid JSON | 0 | 57 / 0 % |
| Qwen inference p50 / p95 | – | 1.9 s / 4.0 s |
| Total latency p50 / p95 | 47 ms / 67 ms | 54 ms / 4.5 s |
| External LLM calls / tokens | 0 | 0 |

Local model resources: model file 491 MB; llama-server RSS 627 MB idle, **675 MB peak** under load (KV cache for 2×2048
context included); ~3.8 of 4 cores busy while generating; load time ~1.2 s from local disk. API process peak RSS
~475 MB (BGE topic gate + router). Gemini was not re-run for comparison (removed; it would send question text to an
external API).

**Limitations.** Qwen2.5-0.5B is weak at this task: of 57 inferences, only 8 produced a plan that survived validation
and gave the right answer; most of its gain comes from questions it gets right after the deterministic canonicalisation
(sort direction, limit, aggregate op, date bounds). Its frequent mistakes — invented conditions, dropped negation,
wrong grouping, refusing valid questions — are caught by the grounding check and turned into a retry and then a
clarifying question, never executed. Not yet handled well: negation ("not active"), age ranges, multi-field breakdowns,
two-value includes, follow-ups ("of them"). Raising this needs a larger local model (e.g. Qwen2.5-1.5B/3B Q4, ~1–2 GB)
or extending the deterministic grammar; measure on a held-out question set, not this one.

## Tests and lint

```bash
uv run pytest          # model calls are faked; set MCP_TEST_MONGODB_URI to also test the real MCP server
uv run ruff check app tests
```

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `LLM_ENABLED` | `true` | Use the local model for questions the deterministic planner cannot resolve |
| `LLM_SERVER_URL` | `http://127.0.0.1:8081` | llama-server address (private network only; compose sets `http://llm:8081`) |
| `LLM_MODEL_NAME` | `qwen2.5-0.5b-instruct-q4_k_m` | Reported in responses/logs |
| `LLM_MAX_TOKENS` | `256` | Output cap per plan |
| `LLM_TIMEOUT_SECONDS` | `15` | Per-inference timeout (also the wait for a free slot) |
| `LLM_MAX_CONCURRENCY` | `2` | Concurrent inferences; keep equal to llama-server `--parallel` |
| llama-server flags | see `deploy/` | `--ctx-size 4096 --parallel 2 --threads 4 --n-predict 256 --no-webui --no-slots --offline` |
| `APP_TIMEZONE` | `UTC` | Facility timezone for "today", "last week", … (e.g. `Asia/Kolkata`) |
| `DATA_SOURCE` | `mcp` | `mcp` (MongoDB MCP Server tools, read-only) or `mongodb` (driver); `memory` is test-only |
| `MCP_SERVER_COMMAND` | `mongodb-mcp-server` | How to start the MCP server (Docker image sets the bundled path) |
| `MCP_MAX_DOCUMENTS` / `MCP_TIMEOUT_SECONDS` | `5000` / `20` | Result cap per MCP query (server default is 100) and per-call timeout |
| `MCP_NODE_PATH` | auto | Node.js binary for the MCP server. By default the first suitable `node` on PATH, else the newest nvm/volta/`/usr/local` install (an old system Node 18 is skipped) |
| `SEMANTIC_ROUTER_ENABLED` / `SEMANTIC_ROUTER_ROUTES_FILE` | `true` / `app/data/intent_routes.yaml` | Intent scoring and routes file |
| `DIRECT_ANSWERS_ENABLED` | `true` | Answer confident simple questions via MongoDB MCP without any model call |
| `MONGODB_URI` / `MONGODB_DB` / `MONGODB_COLLECTION` | – / `care_app` / `residents` | MongoDB connection |
| `VOCAB_REFRESH_SECONDS` | `300` | How often names/wings/rooms are reloaded from the database |
| `TOPIC_GATE_ENABLED` | `true` | Local topic check: a small embedding model on this server stops clearly off-topic questions before any planning |
| `TOPIC_GATE_MODEL` / `TOPIC_GATE_CACHE_DIR` | `BAAI/bge-small-en-v1.5` / `.models` | Embedding model (~65 MB, downloaded once) and where it is stored |
| `TOPIC_GATE_MARGIN` | `0.02` | How clearly off-topic a question must be to block it; higher blocks less. Tuned so no resident test question is blocked. Reference examples: `app/nlu/topic_examples.py` |
| `CACHE_MAX_ENTRIES` / `CACHE_TTL_SECONDS` | `5000` / `86400` | Plan cache (repeat questions skip planning) |
| `SESSION_TTL_SECONDS` | `1800` | How long follow-up context is kept per conversation |
| `RATE_LIMIT_PER_MINUTE` / `RATE_LIMIT_PER_DAY` | `60` / `2000` | Per-IP limits on `POST /api/query` |
| `API_TOKEN` | – | If set, `/api/*` (except health/ready) requires `Authorization: Bearer <token>` |
| `CORS_ORIGINS` | – | Comma-separated allowed origins when the UI is served from another domain |
| `ENABLE_DATA_BROWSER` | `false` | Exposes `GET /api/residents` for the in-app record browser |
| `ENVIRONMENT` | `development` | `production` disables the interactive API docs |

## API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/api/query` | `{"query": "...", "session_id": "optional-uuid"}` → answer, plan, interpretation, `tier`, `modelCalls`, `externalLlmCalls` (always 0) |
| `GET` | `/api/health` | Liveness |
| `GET` | `/api/ready` | Readiness (data source reachable, index loaded) |
| `GET` | `/api/meta` | Record count, model, feature flags |
| `GET` | `/api/stats` | Questions by tier, local model calls/tokens/inference time, cache hits |
| `GET` | `/api/residents` | Only when `ENABLE_DATA_BROWSER=true`; never includes emergency contacts |
| `GET` | `/api/docs` | Interactive docs (non-production only) |

## Production notes

- Build and run: `scripts/fetch_model.sh /opt/sashi/models` once, then `docker compose -f deploy/docker-compose.yml up -d --build`
  (the `app` container serves the API and the React app; the `llm` container is internal only).
- **Authentication:** `API_TOKEN` is a shared secret for service-to-service use. End-user authentication should come from
  the Care App's existing login/SSO in front of this service.
- **MongoDB:** the mapping lives in `app/data/resident_schema.yaml`. Use a read-only database user. Useful indexes:
  `residents {deletedAt: 1}`, `staffs {cName: 1}`, `facilities {facilityId: 1}`.
- **Multiple replicas:** the plan cache, conversation memory and rate limiter are per process. Use Redis for shared state.
- **Data handling:** no question text leaves the server: the model runs locally. Resident names are still masked before
  the model sees the question, and llama-server runs with `--no-slots` (no prompt inspection endpoint) and `--offline`.
- **Sizing:** see "Benchmark" below for measured RAM/CPU. Size the host for the model's ~0.7 GB RSS plus the API
  process (~0.5 GB with the BGE models), and budget ~4 vCPUs for inference.
- **Data quality (sal_staging):** 68 residents have no admission date, ~18 have no usable birth date, 213 have no
  assigned staff, and 24 have a facility code with no `facilities` record (shown by code). Those residents are
  "unknown" for that field.
