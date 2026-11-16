"""HTTP routes."""

import hmac

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import Settings
from app.data.repository import list_view
from app.engine.service import QueryService
from app.models.plan import Filters, Sort

limiter = Limiter(key_func=get_remote_address)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=500)
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


def get_service(request: Request) -> QueryService:
    return request.app.state.service


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def require_token(request: Request, settings: Settings = Depends(get_settings_dep)) -> None:
    """Optional shared-secret auth for service-to-service calls (Authorization: Bearer <API_TOKEN>)."""
    if not settings.api_token:
        return
    header = request.headers.get("authorization", "")
    given = header[7:] if header.startswith("Bearer ") else ""
    if not hmac.compare_digest(given.encode(), settings.api_token.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized")


public = APIRouter(prefix="/api")
protected = APIRouter(prefix="/api", dependencies=[Depends(require_token)])


@public.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@public.get("/ready")
async def ready(service: QueryService = Depends(get_service)) -> dict:
    try:
        await service.repo.total()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="data source unavailable") from exc
    if not service.ready:
        raise HTTPException(status_code=503, detail="index not ready")
    # The local model is optional for readiness: simple questions are answered without it.
    model_ready = await service.translator.health() if service.translator is not None else False
    return {"status": "ready", "llmConfigured": service.translator is not None, "llmReady": model_ready}


@protected.get("/meta")
async def meta(service: QueryService = Depends(get_service), settings: Settings = Depends(get_settings_dep)) -> dict:
    return {
        "totalResidents": await service.repo.total(),
        "llmEnabled": service.translator is not None,
        "provider": getattr(service.translator, "provider", None),
        "model": getattr(service.translator, "model_name", None),
        "externalLlm": False,
        "dataBrowserEnabled": settings.enable_data_browser,
    }


@protected.get("/stats")
async def stats(service: QueryService = Depends(get_service)) -> dict:
    s = service.stats
    gate = service.gate
    return {
        **s.__dict__,
        "cacheEntries": len(service.cache),
        "cacheHits": service.cache.hits,
        "topicCheck": {
            "enabled": gate is not None,
            "ready": bool(gate and gate.ready),
            "blocked": gate.blocked_count if gate else 0,
        },
    }


@protected.post("/query")
@limiter.limit(lambda: _rate_limit())
async def query(request: Request, body: QueryRequest, service: QueryService = Depends(get_service)) -> dict:
    return await service.handle(body.query, body.session_id)


@protected.get("/residents")
async def residents(
    service: QueryService = Depends(get_service), settings: Settings = Depends(get_settings_dep)
) -> dict:
    if not settings.enable_data_browser:
        raise HTTPException(status_code=403, detail="The data browser is disabled on this deployment.")
    rows = await service.repo.find(Filters(), sort=Sort(field="last_name", direction="asc"), limit=500)
    return {"residents": [list_view(r) for r in rows]}


def _rate_limit() -> str:
    from app.config import get_settings

    s = get_settings()
    return f"{s.rate_limit_per_minute}/minute;{s.rate_limit_per_day}/day"
