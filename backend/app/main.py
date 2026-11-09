"""FastAPI application factory and entry point.

Run locally:   uv run uvicorn app.main:app --reload --port 8000
"""

import logging
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
from app.data.memory_repository import MemoryResidentRepository
from app.data.repository import ResidentRepository
from app.engine.service import QueryService
from app.logging_setup import configure_logging
from app.nlu.translator import ClaudeTranslator, Translator

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
    if settings.data_source == "mongodb":
        from app.data.mongo_repository import MongoResidentRepository

        return MongoResidentRepository(settings.mongodb_uri, settings.mongodb_db, settings.mongodb_collection)
    if settings.environment == "production":
        log.warning("DATA_SOURCE=memory in production: serving built-in seed data, not the live Care App database")
    return MemoryResidentRepository()


def build_translator(settings: Settings) -> Translator | None:
    """Pick the language model provider. LLM_PROVIDER=auto prefers Gemini when GEMINI_API_KEY is set."""
    import os

    has_gemini = bool(settings.gemini_api_key)
    has_anthropic = bool(
        settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )
    provider = settings.llm_provider
    if provider == "auto":
        provider = "gemini" if has_gemini else "anthropic" if has_anthropic else "none"

    if provider == "gemini" and has_gemini:
        from app.nlu.gemini_translator import GeminiTranslator

        return GeminiTranslator(settings)
    if provider == "anthropic" and has_anthropic:
        return ClaudeTranslator(settings)
    log.warning(
        "No language model credentials for provider %r (set GEMINI_API_KEY or ANTHROPIC_API_KEY): "
        "questions will return 'assistant unavailable'",
        settings.llm_provider,
    )
    return None


def create_app(
    settings: Settings | None = None,
    repo: ResidentRepository | None = None,
    translator: Translator | None = None,
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
        service = QueryService(repository, model, settings)
        await service.start()
        app.state.settings = settings
        app.state.service = service
        log.info(
            "resident query API started",
            extra={
                "data_source": repository.name,
                "llm": model is not None,
                "provider": getattr(model, "provider", None),
                "model": getattr(model, "model_name", None),
            },
        )
        yield
        await service.stop()
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
        response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
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
