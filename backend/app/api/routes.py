"""HTTP routes."""

import hmac
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.auth.guard import AccessDenied
from app.auth.identity import AuthError, principal_from_caller
from app.auth.principal import SYSTEM, Principal
from app.config import Settings
from app.data.repository import list_view
from app.data.snf_mcp_repository import use_credentials
from app.engine.service import QueryService
from app.models.plan import Filters, Sort

limiter = Limiter(key_func=get_remote_address)
log = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=500)
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


def get_service(request: Request) -> QueryService:
    return request.app.state.service


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def require_token(request: Request, settings: Settings = Depends(get_settings_dep)) -> None:
    """Optional shared secret for service-to-service calls (x-api-key: <API_TOKEN>). Authorization carries the user's
    own access token (forward_credentials)."""
    if not settings.api_token:
        return
    given = request.headers.get("x-api-key", "")
    if not hmac.compare_digest(given.encode(), settings.api_token.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized")


async def forward_credentials(request: Request, service: QueryService = Depends(get_service)) -> None:
    """The signed-in user's Cognito access token (Authorization: Bearer …) and facility (x-facility-id), sent by the UI
    with every request, for every snf_mcp call of this request. They are only passed on: snf_mcp verifies the token
    and checks the facility; the role and permissions come from snf_mcp, never from the browser."""
    if not getattr(service.repo, "per_caller", False):
        return  # the in-memory test data has no users
    auth = request.headers.get("authorization", "")
    token = auth[7:].strip() if auth[:7].lower() == "bearer " else ""
    facility = request.headers.get("x-facility-id", "").strip()
    if not token or not facility:
        raise HTTPException(status_code=401, detail="Sign in to use the assistant: the request has no access token "
                                                    "(Authorization: Bearer …) or facility (x-facility-id).")  # fmt: skip
    use_credentials(token, facility)


async def get_principal(service: QueryService = Depends(get_service)) -> Principal:
    """Who is asking: the SNF_MCP_TOKEN user (app/auth/identity.py). Refused or inactive callers stop here, before any
    query. The in-memory test data has no caller: unrestricted."""
    caller = getattr(service.repo, "caller", None)
    if caller is None:
        return SYSTEM
    try:
        return principal_from_caller(await caller())
    except AuthError as exc:  # DataUnavailable propagates: 503 (app/main.py), never an unrestricted caller
        raise HTTPException(status_code=exc.status, detail=exc.message) from None


public = APIRouter(prefix="/api")
protected = APIRouter(prefix="/api", dependencies=[Depends(require_token), Depends(forward_credentials)])


@public.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@public.get("/ready")
async def ready(service: QueryService = Depends(get_service)) -> dict:
    if not getattr(service.repo, "per_caller", False):  # per-user data (snf_mcp) cannot be read without a user
        try:
            await service.repo.total()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="data source unavailable") from exc
    if not service.ready:
        raise HTTPException(status_code=503, detail="index not ready")
    # The local model is optional for readiness: simple questions are answered without it.
    model_ready = await service.translator.health() if service.translator is not None else False
    warm = getattr(service.translator, "warm", None) if service.translator is not None else None
    # llmWarm: the model's prompt is cached (startup warm-up finished); until then model questions are declined quickly
    return {"status": "ready", "llmConfigured": service.translator is not None, "llmReady": model_ready, "llmWarm": warm}


@protected.get("/meta")
async def meta(service: QueryService = Depends(get_service), settings: Settings = Depends(get_settings_dep)) -> dict:
    return {
        "totalResidents": await service.repo.total(),
        "llmEnabled": service.translator is not None,
        "provider": getattr(service.translator, "provider", None),
        "model": getattr(service.translator, "model_name", None),
        "externalLlm": bool(getattr(service.translator, "external", False)),
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
async def query(
    request: Request, body: QueryRequest, service: QueryService = Depends(get_service),
    principal: Principal = Depends(get_principal), settings: Settings = Depends(get_settings_dep),
) -> dict:
    result = await service.handle(body.query, body.session_id, principal=principal)
    if settings.log_api_responses:  # the exact JSON the UI renders (src/types/resident.ts QueryExecutionResult)
        log.info("api response", extra={"path": "/api/query", "response": result})
    return result


@protected.get("/residents")
async def residents(
    service: QueryService = Depends(get_service), settings: Settings = Depends(get_settings_dep),
    principal: Principal = Depends(get_principal),
) -> dict:
    if not settings.enable_data_browser:
        raise HTTPException(status_code=403, detail="The data browser is disabled on this deployment.")
    try:  # the same guardrails as a question: the Residents module, then only the caller's residents
        rows = await service.browse(Filters(), Sort(field="last_name", direction="asc"), 500, principal)
    except AccessDenied as exc:
        raise HTTPException(status_code=403, detail=exc.message) from None
    return {"residents": [list_view(r) for r in rows]}


def _rate_limit() -> str:
    from app.config import get_settings

    s = get_settings()
    return f"{s.rate_limit_per_minute}/minute;{s.rate_limit_per_day}/day"
