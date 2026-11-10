# Resident Query API (Python / FastAPI)

Staff ask questions in plain language. A language model (Google Gemini by default, or Anthropic Claude) translates each question into a JSON **query plan**; the backend
validates the plan, runs it against the resident database and builds the answer from the results.
The model never sees resident records and never produces a number.

```
question → mask resident names → plan cache? → local topic check (off-topic → "not available", 0 tokens)
         → Gemini / Claude (structured output: plan only)
         → resolve names/wings/rooms against the database → validate → query database → answer + "Interpreted as" chips
```

## Run locally

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
cd backend
uv sync                                    # install dependencies into .venv
# put ONE of these in backend/.env (or export it):
#   GEMINI_API_KEY=...           → uses Google Gemini (key from aistudio.google.com)
#   ANTHROPIC_API_KEY=sk-ant-... → uses Anthropic Claude
uv run uvicorn app.main:app --reload --port 8000
```

In a second terminal, from the repository root: `npm run dev`, then open http://localhost:3000.
Vite forwards `/api/*` to the backend on port 8000.

Without a model key the server still starts; questions return "the language model is not configured".
Check with `curl localhost:8000/api/meta`, which shows the active `provider` and `model`.

## Tests and lint

```bash
uv run pytest          # 105 tests; the model is replaced by fakes/mocks, so no API calls or cost
uv run ruff check app tests
```

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `auto` | `auto` (Gemini if `GEMINI_API_KEY` is set, else Anthropic), `gemini`, or `anthropic` |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | – | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Fast, low-cost stable model. `gemini-3.8-flash` is the most capable Flash |
| `GEMINI_THINKING_LEVEL` | `minimal` | `minimal` / `low` / `medium` / `high`, or `default` to let the model decide |
| `GEMINI_MAX_TOKENS` / `GEMINI_TIMEOUT_SECONDS` | `1000` / `20` | Output cap and timeout (SDK retries 2× on transient errors) |
| `ANTHROPIC_API_KEY` | – | Claude API key (alternative provider) |
| `ANTHROPIC_MODEL` | `claude-opus-5-5` | Model used for translation |
| `ANTHROPIC_EFFORT` | `low` | Thinking effort (`low` / `medium` / `high`); low keeps tokens down |
| `ANTHROPIC_MAX_TOKENS` | `4000` | Output cap per translation |
| `ANTHROPIC_TIMEOUT_SECONDS` | `20` | Per-request timeout (SDK retries 2× on transient errors) |
| `ANTHROPIC_REFUSAL_FALLBACKS` | `true` | Server-side fallback model if a request is declined |
| `APP_TIMEZONE` | `UTC` | Facility timezone for "today", "last week", … (e.g. `Asia/Kolkata`) |
| `DATA_SOURCE` | `memory` | `memory` (built-in seed data) or `mongodb` |
| `MONGODB_URI` / `MONGODB_DB` / `MONGODB_COLLECTION` | – / `care_app` / `residents` | MongoDB connection |
| `VOCAB_REFRESH_SECONDS` | `300` | How often names/wings/rooms are reloaded from the database |
| `TOPIC_GATE_ENABLED` | `true` | Local topic check: a free embedding model on this server stops clearly off-topic questions before the language model |
| `TOPIC_GATE_MODEL` / `TOPIC_GATE_CACHE_DIR` | `BAAI/bge-small-en-v1.5` / `.models` | Embedding model (~65 MB, downloaded once) and where it is stored |
| `TOPIC_GATE_MARGIN` | `0.02` | How clearly off-topic a question must be to block it; higher blocks less. Tuned so no resident test question is blocked. Reference examples: `app/nlu/topic_examples.py` |
| `LLM_DAILY_CALL_LIMIT` | `0` | Cost cap: maximum language-model calls per day per server (0 = unlimited). Cached and filtered questions don't count |
| `CACHE_MAX_ENTRIES` / `CACHE_TTL_SECONDS` | `5000` / `86400` | Plan cache (repeat questions cost 0 tokens) |
| `SESSION_TTL_SECONDS` | `1800` | How long follow-up context is kept per conversation |
| `RATE_LIMIT_PER_MINUTE` / `RATE_LIMIT_PER_DAY` | `60` / `2000` | Per-IP limits on `POST /api/query` |
| `API_TOKEN` | – | If set, `/api/*` (except health/ready) requires `Authorization: Bearer <token>` |
| `CORS_ORIGINS` | – | Comma-separated allowed origins when the UI is served from another domain |
| `ENABLE_DATA_BROWSER` | `false` | Exposes `GET /api/residents` for the in-app record browser |
| `ENVIRONMENT` | `development` | `production` disables the interactive API docs |

## API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/api/query` | `{"query": "...", "session_id": "optional-uuid"}` → answer, plan, interpretation, token usage |
| `GET` | `/api/health` | Liveness |
| `GET` | `/api/ready` | Readiness (data source reachable, index loaded) |
| `GET` | `/api/meta` | Record count, model, feature flags |
| `GET` | `/api/stats` | Questions by tier, token totals, cache hits |
| `GET` | `/api/residents` | Only when `ENABLE_DATA_BROWSER=true`; never includes emergency contacts |
| `GET` | `/api/docs` | Interactive docs (non-production only) |

## Production notes

- Build and run: `docker build -t resident-query .` then `docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... resident-query`
  (one container serves both the API and the built React app).
- **Authentication:** `API_TOKEN` is a shared secret for service-to-service use. End-user authentication should come from
  the Care App's existing login/SSO in front of this service.
- **MongoDB schema:** `app/data/mongo_repository.py` assumes the same field names and formats as `app/data/residents.json`.
  Verify against the real collection and add the indexes listed there.
- **Multiple replicas:** the plan cache, conversation memory and rate limiter are per process. Use Redis for shared state.
- **Data handling:** resident names are masked before questions are sent to the model provider (Google or Anthropic), but the
  rest of the question text is sent. Confirm this matches your data-processing agreement with that provider.
- **Data quality:** six pending residents in the seed data have admission dates in the future (scheduled admissions).
  Decide whether "admitted" questions should include them.
