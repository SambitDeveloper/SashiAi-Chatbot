"""FastAPI application factory and entry point.

Run locally:   uv run uvicorn app.main:app --reload --port 8000
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import limiter, protected, public
from app.config import Settings, get_settings
from app.data.repository import ResidentRepository
from app.engine.service import QueryService
from app.logging_setup import configure_logging
from app.nlu.intent_router import DEFAULT_ROUTES_FILE, SemanticIntentRouter
from app.nlu.topic_gate import TopicGate
from app.nlu.translator import QueryTranslator

log = logging.getLogger(__name__)

SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


def build_repository(settings: Settings) -> ResidentRepository:
    """Answers always come from the Care App's MongoDB; the in-memory test data is refused outside tests."""
    if settings.data_source == "memory":
        if settings.environment != "test":
            raise RuntimeError("DATA_SOURCE=memory is for tests only. Set DATA_SOURCE=mcp and MONGODB_URI.")
        from app.data.memory_repository import MemoryResidentRepository

        return MemoryResidentRepository()
    if not settings.mongodb_uri:
        raise RuntimeError(f"MONGODB_URI is required (DATA_SOURCE={settings.data_source}). Set it in backend/.env.")
    if settings.data_source == "mcp":
        from app.data.mcp_repository import McpResidentRepository

        return McpResidentRepository(
            settings.mcp_server_command,
            settings.mongodb_uri,
            settings.mongodb_db,
            settings.mongodb_collection,
            max_documents=settings.mcp_max_documents,
            timeout_seconds=settings.mcp_timeout_seconds,
            node_path=settings.mcp_node_path,
        )
    from app.data.mongo_repository import MongoResidentRepository

    return MongoResidentRepository(settings.mongodb_uri, settings.mongodb_db, settings.mongodb_collection)


def build_translator(settings: Settings) -> QueryTranslator | None:
    """The model client for LLM_PROVIDER: "local" = Qwen on llama-server (private network), "bedrock" = Gemma 4 on AWS
    Bedrock (bedrock-mantle). Exactly one is built; there is no fallback from one to the other."""
    if not settings.llm_enabled:
        log.warning("LLM_ENABLED=false: only questions the deterministic planner resolves will be answered")
        return None
    if settings.llm_provider == "bedrock":
        from app.nlu.bedrock_translator import BedrockTranslator

        return BedrockTranslator(settings)
    from app.nlu.qwen_translator import QwenTranslator

    return QwenTranslator(settings)


def build_semantic_cache(settings: Settings, topic_gate: TopicGate | None):
    """The semantic plan template cache (off unless PLAN_CACHE_ENABLED). Uses its own Atlas connection string and the
    topic gate's already-loaded BGE model (a private instance of the same model only when the gate is disabled)."""
    if not settings.plan_cache_enabled:
        return None
    from app.engine.semantic_cache import AtlasTemplateStore, SemanticPlanCache
    from app.nlu.topic_gate import FastEmbedEmbedder

    own: list = []

    def embedder():
        if topic_gate is not None and topic_gate.embedder is not None:
            return topic_gate.embedder
        if not own:
            own.append(FastEmbedEmbedder(settings.topic_gate_model, settings.topic_gate_cache_dir))
        return own[0]

    store = AtlasTemplateStore(
        settings.plan_cache_mongodb_uri,
        settings.plan_cache_database,
        settings.plan_cache_collection,
        settings.plan_cache_vector_index,
        settings.plan_cache_timeout_seconds,
        ttl_seconds=settings.plan_cache_ttl_hours * 3600,
    )
    log.info(
        "semantic plan cache enabled",
        extra={"database": settings.plan_cache_database, "collection": settings.plan_cache_collection,
               "index": settings.plan_cache_vector_index, "threshold": settings.plan_cache_similarity_threshold},
    )  # fmt: skip
    return SemanticPlanCache(
        store,
        embedder,
        threshold=settings.plan_cache_similarity_threshold,
        top_k=settings.plan_cache_top_k,
        max_candidates=settings.plan_cache_max_candidates,
        timeout_s=settings.plan_cache_timeout_seconds,
        embedding_model=settings.topic_gate_model,
        store_templates=settings.plan_cache_store_templates,
    )


def create_app(
    settings: Settings | None = None,
    repo: ResidentRepository | None = None,
    translator: QueryTranslator | None = None,
    gate: TopicGate | None = None,
    use_default_translator: bool = True,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        repository = repo or build_repository(settings)
        await repository.init()
        model = (
            translator if translator is not None else (build_translator(settings) if use_default_translator else None)
        )
        topic_gate = (
            gate
            if gate is not None
            else (
                TopicGate(
                    settings.topic_gate_margin,
                    model_name=settings.topic_gate_model,
                    cache_dir=settings.topic_gate_cache_dir,
                )
                if settings.topic_gate_enabled
                else None
            )
        )
        intent_router = None
        if settings.semantic_router_enabled:
            intent_router = SemanticIntentRouter(
                settings.semantic_router_routes_file or DEFAULT_ROUTES_FILE, cache_dir=settings.topic_gate_cache_dir
            )
        semantic_cache = build_semantic_cache(settings, topic_gate)
        cache_connect_task = None
        if semantic_cache is not None:  # a cold Atlas connection takes seconds: open it before the first question

            async def connect_cache():
                try:
                    await semantic_cache.store.connect()
                    log.info("semantic plan cache connected")
                except Exception as exc:  # questions still work: every lookup is a miss until Atlas is reachable
                    log.warning("semantic plan cache not reachable at startup: %s", type(exc).__name__)

            cache_connect_task = asyncio.create_task(connect_cache())
        service = QueryService(
            repository, model, settings, gate=topic_gate, intent_router=intent_router, semantic_cache=semantic_cache
        )
        await service.start()
        # Warm the local model's prompt cache in the background (a cold first read of the prompt takes minutes on a
        # CPU), then keep watching: if llama-server restarts, its cache is empty again and is re-warmed automatically.
        async def warm_and_watch():
            await model.warm_up(service.schema_context(), settings.today())
            await model.keep_warm(service.schema_context, settings.today)

        warm_task = asyncio.create_task(warm_and_watch()) if hasattr(model, "keep_warm") else (
            asyncio.create_task(model.warm_up(service.schema_context(), settings.today())) if hasattr(model, "warm_up") else None
        )  # fmt: skip
        # Load the embedding model in the background: the API is usable at once; until it is ready, questions skip
        # the topic check and the deterministic planner and go to the local model.
        gate_task = asyncio.create_task(topic_gate.start()) if topic_gate and not topic_gate.ready else None
        router_task = asyncio.create_task(intent_router.start()) if intent_router else None
        app.state.settings = settings
        app.state.service = service
        log.info(
            "resident query API started",
            extra={
                "data_source": repository.name,
                "llm": model is not None,
                "provider": getattr(model, "provider", None),
                "model": getattr(model, "model_name", None),
                "llm_server": (getattr(model, "endpoint", None) or settings.llm_server_url) if model is not None else None,
                "topic_check": topic_gate is not None,
                "semantic_router": intent_router is not None,
            },
        )
        yield
        for task in (gate_task, router_task, warm_task, cache_connect_task):
            if task and not task.done():
                task.cancel()
        await service.stop()
        if semantic_cache is not None:
            await semantic_cache.drain()
            await semantic_cache.store.close()
        if hasattr(model, "close"):
            await model.close()
        await repository.close()

    app = FastAPI(
        title="Care App Resident Query API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs" if settings.environment != "production" else None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.limiter = limiter

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "Authorization"],
        )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        if request.url.path == "/api/query":
            # end-to-end time as the client sees it: request received → response returned (pipeline + HTTP handling)
            total_ms = round((time.perf_counter() - started) * 1000)
            response.headers["Server-Timing"] = f"total;dur={total_ms}"
            log.info("http request completed", extra={"method": request.method, "path": request.url.path,
                                                      "status": response.status_code, "total_ms": total_ms})  # fmt: skip
        return response

    @app.exception_handler(RateLimitExceeded)
    async def rate_limited(_: Request, __: RateLimitExceeded):
        return JSONResponse({"error": "Too many requests. Please wait a moment and try again."}, status_code=429)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        field_name = ".".join(str(p) for p in first.get("loc", [])[1:]) or "request"
        return JSONResponse({"error": f"{field_name}: {first.get('msg', 'invalid')}"}, status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse({"error": "Something went wrong. Please try again."}, status_code=500)

    app.include_router(public)
    app.include_router(protected)

    static_dir = Path(settings.static_dir) if settings.static_dir else Path(__file__).resolve().parents[2] / "dist"
    if (static_dir / "index.html").exists():
        if (static_dir / "assets").exists():
            app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            if path.startswith("api/"):
                return JSONResponse({"error": "Not found"}, status_code=404)
            candidate = (static_dir / path).resolve()
            if path and candidate.is_file() and static_dir.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(static_dir / "index.html")

    return app


app = create_app()
