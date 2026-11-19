# Resident Query API (Python / FastAPI)

Staff ask questions in plain language. The interpreter emits a compositional **RequestIR**; a deterministic compiler
lowers supported operations to the restricted **QueryPlan**, which the backend validates before querying the resident
database. Simple questions are interpreted **deterministically, with no model call**. Only questions the fast path
cannot safely interpret go to the configured local Qwen model on llama.cpp. No external AI API is used; the model
never sees resident records and never produces a number.

```
question → mask resident names → plan cache?
         → topic check (BGE embeddings, local) / out-of-scope policy (medications, billing, …) → "not available"
         → deterministic planner: semantic router + grammar.py dates/ages + database values   (no model call)
              └─ declined → local Qwen via llama-server (grammar-constrained RequestIR)
         → deterministic RequestIR compiler → resolver (exact DB values) → Pydantic QueryPlan
         → grounding check (app/engine/grounding.py) → read-only database query
         → model plans failing validation are retried once, then the user is asked to clarify
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
scripts/run_api.sh                         # API on :8000 with auto-reload (polling on WSL /mnt/c — plain
                                           # `uvicorn --reload` there silently keeps serving old code)
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

## AWS Bedrock model (Gemma 4 31B, `LLM_PROVIDER=bedrock`)

`LLM_PROVIDER` picks the model client: `local` (default in code: Qwen on llama-server, above) or `bedrock`
(`app/nlu/bedrock_translator.py`). Exactly one is used; **a Bedrock failure is never retried on Qwen** — it gets the
normal "couldn't interpret that question" answer. The Qwen GGUF, `qwen_translator.py` and `run_llm_server.sh` are kept
for switching back (`LLM_PROVIDER=local`, start llama-server, restart the API).

API facts (AWS Bedrock user guide, checked 2026-11-19): Gemma 4 is served **only** on `bedrock-mantle`, under the
**`/openai/v1`** prefix — `POST https://bedrock-mantle.us-west-2.api.aws/openai/v1/chat/completions`, model
`google.gemma-4-31b`, In-Region only. `/v1/chat/completions`, bedrock-runtime, Converse and InvokeModel reject it.
Auth is an Amazon Bedrock API key sent as `Authorization: Bearer <key>`.

```bash
# backend/.env (git-ignored) — never commit the key
LLM_PROVIDER=bedrock
BEDROCK_API_KEY=<your Bedrock API key>     # or AWS_BEARER_TOKEN_BEDROCK
BEDROCK_REGION=us-west-2                   # default
BEDROCK_MODEL_ID=google.gemma-4-31b    # default
LLM_TIMEOUT_SECONDS=30                     # the 120 s default is sized for the local CPU
LLM_MAX_CONCURRENCY=4                      # 1 is sized for one llama-server slot
```

Differences from the local client (same prompt, RequestIR schema, validation, grounding check and retry):

- **No constrained decoding by default.** Structured outputs are not listed for Gemma 4 on bedrock-mantle, and Bedrock's
  JSON-Schema subset rejects parts of the RequestIR schema (numeric bounds, >12 optional properties, objects without
  `additionalProperties: false`). The reply is parsed as JSON (a ```` ```json ```` fence is tolerated) and validated as
  usual; malformed or off-schema output becomes `invalid_json` or a validator retry. `BEDROCK_RESPONSE_FORMAT=json_object|json_schema`
  is an opt-in experiment.
- `top_k`, `seed`, `cache_prompt` and `LLM_CHAT_TEMPLATE_KWARGS` are local-only and not sent. No warm-up is needed.
- `BEDROCK_REASONING_EFFORT=low|medium|high` (unset = model default). Reasoning tokens are not returned on Chat Completions.
- Failures: 401/403 → `auth`, 429 → `overloaded` (mantle throughput ramps up over time), 5xx → `unavailable`, other 4xx →
  `api_error`. `/api/ready` `llmReady` = `GET https://bedrock-mantle.{region}.api.aws/v1/models` returns 200 (endpoint +
  key; cached 30 s).
- **Data leaves the server.** The masked question (resident names replaced by placeholders), the wing list and the
  status/care-type vocabulary are sent to AWS. Model calls are counted as `externalLlmCalls`, `/api/meta` reports
  `externalLlm: true`, and logs show `plannerName: bedrock_gemma`.

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
| `LLM_ENABLED` | `true` | Use a model for questions the deterministic planner cannot resolve |
| `LLM_PROVIDER` | `local` | `local` = Qwen on llama-server; `bedrock` = Gemma 4 on AWS Bedrock (no fallback between them) |
| `BEDROCK_API_KEY` / `AWS_BEARER_TOKEN_BEDROCK` | – | Bedrock API key (required with `LLM_PROVIDER=bedrock`; `.env` only, never logged) |
| `BEDROCK_REGION` | `us-west-2` | bedrock-mantle region |
| `BEDROCK_MODEL_ID` | `google.gemma-4-31b` | Bedrock model id |
| `BEDROCK_BASE_URL` | `https://bedrock-mantle.{region}.api.aws/openai/v1` | Override only if AWS changes the path |
| `BEDROCK_RESPONSE_FORMAT` | `none` | `json_object` / `json_schema`: opt-in output constraint (experimental) |
| `BEDROCK_REASONING_EFFORT` | unset | `low` / `medium` / `high` |
| `LLM_SERVER_URL` | `http://127.0.0.1:8081` | llama-server address (private network only; compose sets `http://llm:8081`) |
| `LLM_MODEL_NAME` | `qwen2.5-0.5b-instruct-q4_k_m` | Reported in responses/logs |
| `LLM_MAX_TOKENS` | `512` | Output cap per plan |
| `LLM_TIMEOUT_SECONDS` | `60` | Per-inference deadline shared by waiting for a free slot and inference |
| `LLM_MAX_CONCURRENCY` | `2` | Concurrent inferences; keep equal to llama-server `--parallel` |
| llama-server flags | see `deploy/` | `--ctx-size 4096 --parallel 2 --threads 4 --n-predict 512 --no-webui --no-slots --offline` |
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
| `CACHE_MAX_ENTRIES` | `5000` | Failure cache: a question whose model plan failed validation is not re-planned in the same context. (`CACHE_TTL_SECONDS` and the exact successful-plan cache were removed 2026-11-18.) |
| `PLAN_CACHE_ENABLED` | `false` | Semantic plan-template cache (see "Semantic plan cache"). Off = no successful-plan caching at all |
| `PLAN_CACHE_MONGODB_URI` | – | **Dedicated** credential: `readWrite` on `PLAN_CACHE_DATABASE` only. Must differ from `MONGODB_URI` (startup fails otherwise) |
| `PLAN_CACHE_DATABASE` / `PLAN_CACHE_COLLECTION` | `sal_staging_vector` / `residents_ai` | Where templates live (never the resident database; startup fails if equal to `MONGODB_DB`) |
| `PLAN_CACHE_VECTOR_INDEX` | `query_plan_embedding_index` | Atlas Vector Search index name |
| `PLAN_CACHE_SIMILARITY_THRESHOLD` | `0.88` | **Experimental** cosine threshold for nominating candidates — not a safety boundary |
| `PLAN_CACHE_TOP_K` / `PLAN_CACHE_MAX_CANDIDATES` | `5` / `50` | `$vectorSearch` `limit` / `numCandidates` |
| `PLAN_CACHE_TIMEOUT_SECONDS` | `2.0` | Embedding or Atlas call slower than this → cache miss (the question goes to Qwen) |
| `PLAN_CACHE_TTL_HOURS` | `12` | Each template is deleted automatically this long after it was first stored (`expires_at` + MongoDB TTL index `expires_at_ttl`, created by the app at startup); reuse does not extend it |
| `PLAN_CACHE_STORE_TEMPLATES` | `true` | Store checked, executed plans as reusable templates (re-validated on every reuse) |
| `SESSION_TTL_SECONDS` | `1800` | How long follow-up context is kept per conversation |
| `RATE_LIMIT_PER_MINUTE` / `RATE_LIMIT_PER_DAY` | `60` / `2000` | Per-IP limits on `POST /api/query` |
| `API_TOKEN` | – | If set, `/api/*` (except health/ready) requires `Authorization: Bearer <token>` |
| `CORS_ORIGINS` | – | Comma-separated allowed origins when the UI is served from another domain |
| `ENABLE_DATA_BROWSER` | `false` | Exposes `GET /api/residents` for the in-app record browser |
| `ENVIRONMENT` | `development` | `production` disables the interactive API docs |

## Semantic plan cache (`app/engine/semantic_cache.py`, since 2026-11-18)

Replaces the exact-fingerprint successful-plan cache. A **template** is a checked plan whose variable values are typed
slots, stored with the BGE-small embedding of the slot-replaced question in a dedicated Atlas collection.

```
question → fast path (unchanged, first) → if not answered: slot-replaced text → BGE embedding (the topic gate's
model) → $vectorSearch (revoked=false, same compatibility key) → per candidate ≥ threshold:
  logic words equal (and/or/not/each/total/oldest …) → every slot refilled from THIS question (residents, dates,
  age bounds, statuses, care types, genders, facilities, caregivers, limit — same shape: "after" never fills "before",
  "older" never fills "younger") → existing resolve_plan + check_plan (must pass unchanged)
→ exactly one distinct plan passes → executed;  otherwise → failure cache → Qwen (unchanged)
```

Similarity only nominates. Measured with BGE-small: "older than 75"/"younger than 75" 0.984, "Active or Discharged"/
"Active and Discharged" 0.988 — higher than true paraphrases (0.915–0.927). `tests/test_semantic_cache.py` runs every
safety test with an embedder that makes every template a perfect match.

**No review step (decided 2026-11-18).** A fast-path or Qwen plan that passed validation unchanged and executed is
stored with `revoked: false` and is reusable immediately; every reuse is re-validated against the new question. A wrong
template is switched off with `revoke` (`revoked: true`, effective on the next query; storing it again never re-enables
it). Only standalone questions: follow-ups ("of them") are neither stored nor served.

**Compatibility key** (any change → old templates are ignored): RequestIR version, statuses + care types of the
database (adding a resident/room/caregiver does NOT invalidate templates), QueryPlan schema + `resident_schema.yaml`,
LLM model + chat-template settings, embedding model, template format version.

**Stored document** (no raw question, no names, ids, rooms, dates or results):
`_id/template_id`, `question_text` (slot-replaced, e.g. `How many {status} residents were admitted after {date}?`),
`embedding` (384 floats), `logic_signature`, `plan_template` (QueryPlan with `{"$slot": …}` markers), `slots`,
`mention_counts`, `action`, `source` (`fast_path`/`local_model`), `revoked`, `compat_key` + its parts,
`created_at`, `expires_at` (TTL), `last_seen_at`, `seen_count`, `usage_count`, `last_used_at`, `revoked_by/at/reason`.

**Atlas setup (once, by an Atlas admin — not the app's credential):**
1. Database user with `readWrite` on `PLAN_CACHE_DATABASE` only → `PLAN_CACHE_MONGODB_URI`.
2. Vector index (`uv run python -m scripts.plan_templates create-index --admin-uri-env ATLAS_ADMIN_URI`, or the Atlas UI
   with this definition on `PLAN_CACHE_COLLECTION`, type *Vector Search*, name `PLAN_CACHE_VECTOR_INDEX`):
   `{"fields": [{"type": "vector", "path": "embedding", "numDimensions": 384, "similarity": "cosine"},
   {"type": "filter", "path": "revoked"}, {"type": "filter", "path": "compat_key"}]}` (an existing index: `update-index`)
3. `uv run python -m scripts.plan_templates verify` until the index is `queryable`, then `PLAN_CACHE_ENABLED=true`.

**Admin:** `scripts.plan_templates list` (active) / `list --revoked`, `show <id>`; a wrong template:
`revoke <id> --by <name> --reason <why>` (effective on the next query).

**Trade-offs (deliberate):** a repeat question no longer hits an in-process dictionary — it pays one embedding (~10 ms
CPU) plus one Atlas round trip, and is only served once a stored template covers it; follow-ups are not cached;
a plan the validator wrongly accepts is reused for similar
questions until it is revoked. The step log (`step-10 semantic_cache`) records hit/miss, the reason,
similarity, candidates and per-reason rejections, and embedding/search/check times.

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
