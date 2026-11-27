"""Care App data through our snf_mcp server: the ONLY way this app reads residents and staff.

The UI sends the signed-in user's Cognito access token and facility with every request (Authorization: Bearer …,
x-facility-id); api/routes.py hands them to this module for the request (use_credentials) and every snf_mcp call
carries them. Nothing comes from settings, and caches are kept per user and facility.

Who the token's user is comes from snf_mcp's `get_caller` (role from Cognito groups, own staff record and permissions);
app/auth/identity.py turns it into the principal whose question gate and plan guard apply here. Then, by role:

  ADMIN / SUPER_ADMIN  every query is the aggregation app/data/resident_view.py builds from the validated plan, run by
                       snf_mcp's `run_query` (PyMongo, read-only, scoped by snf_mcp to SNF_MCP_FACILITY_ID).
  STAFF                only the tools in app/data/staff_tools.yaml, which call the Care App REST API as the user
                       (e.g. list_my_residents: their assigned residents). Before every question the tool must exist in
                       snf_mcp and the plan may use only that tool's fields; the plan then runs in memory over the
                       tool's rows. Anything else: "That isn't supported yet" (DataUnavailable).
"""

import datetime as dt
import hashlib
import logging
from contextvars import ContextVar
from pathlib import Path

import yaml
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

from app.auth.guard import plan_fields
from app.data.memory_repository import MemoryResidentRepository
from app.data.repository import FIELD_MAP, DataUnavailable, NotSupported, Vocabulary
from app.data.resident_view import ViewRepository
from app.data.staff import staff_config
from app.engine.stores import TTLCache
from app.models.plan import Filters

log = logging.getLogger(__name__)
STAFF_TOOLS_FILE = Path(__file__).with_name("staff_tools.yaml")
ADMIN_GROUPS = {"ADMIN", "SUPER_ADMIN"}
_CREDENTIALS: ContextVar[tuple[str, str] | None] = ContextVar("snf_mcp_credentials", default=None)


def use_credentials(token: str, facility_id: str) -> None:
    """The current request's access token and facility (api/routes.py); every snf_mcp call of the request uses them."""
    _CREDENTIALS.set((token, facility_id))


class NoCredentials(DataUnavailable):
    """No access token and facility for this call (outside a request, e.g. at startup)."""
MAX_PAGES = 50


class _AdminQueries(ViewRepository):
    """The resident view's aggregations, run by snf_mcp's run_query (admins only)."""

    name = "snf-mcp-admin"

    def __init__(self, call):
        super().__init__()
        self._call = call

    async def _aggregate(self, pipeline: list[dict], collection: str | None = None) -> list[dict]:
        args = {"collection": collection or self.resident_collection, "pipeline": pipeline}
        return (await self._call("run_query", args)).get("documents") or []


def _populated(rows: list[dict], field: str, which) -> bool:
    """Every resident that must have `field` has it: all of them, or those matching {plan field: value}."""
    need = rows if which in (None, "all") else [r for r in rows if all(r.get(FIELD_MAP[k]) == v for k, v in which.items())]
    return bool(need) and all(r.get(FIELD_MAP[field]) not in (None, "") for r in need)


def _age(r: dict) -> int | None:
    age, born = r.get("age"), r.get("birth_date")
    if not isinstance(age, int) and isinstance(born, str) and len(born) >= 10 and born[4] == "-":
        today = dt.date.today()
        age = today.year - int(born[:4]) - (today.strftime("%m-%d") < born[5:10])
    return age if isinstance(age, int) and 0 <= age <= 120 else None


def _key(*parts) -> str:
    return "R-" + hashlib.sha1("|".join(str(p or "") for p in parts).encode()).hexdigest()[:10].upper()


def _caseload(rows: list[dict], facility_id: str, caller: dict) -> MemoryResidentRepository:
    """list_my_residents rows → resident view records + family and staff documents (relations.py, staff.py rules).
    Every row is assigned to the caller (that is the endpoint), so their key goes into each resident's staff list."""
    me, profile = caller.get("c_name") or "", caller.get("profile") or {}
    staff = {me: {"cName": me, "name": profile.get("name"), "designation": profile.get("designation"), "active": True}}
    records, family = [], []
    for r in rows:
        name = (r.get("name") or "").strip()
        first, _, last = name.partition(" ")
        rid = _key(name, r.get("unit"))
        team = [p for p in r.get("assigned_staff") or [] if p.get("name") and p.get("name") != profile.get("name")]
        for p in team:
            staff.setdefault(p["name"], {"cName": p["name"], "name": p["name"], "designation": p.get("role"), "active": True})
        doctor = r.get("primary_physician")
        if doctor:
            staff.setdefault(doctor, {"cName": doctor, "name": doctor, "designation": None, "active": True})
        records.append({
            "id": rid, "residentId": rid, "firstName": first, "lastName": last, "facilityId": facility_id,
            "wing": facility_id, "status": str(r.get("status") or "unknown").lower(), "careType": r.get("care_type") or None,
            "gender": None,
            "age": _age(r), "roomNumber": r.get("unit") or "", "admissionDate": r.get("admission_date"),
            "createdDate": r.get("created_date"), "dischargeDate": r.get("discharge_date"), "primaryCaregiver": [p["name"] for p in team],
            "assignedStaff": [me, *(p["name"] for p in team)], "primaryPhysicianCName": doctor,
            "emergencyContact": {"name": "", "relationship": "", "phone": ""},
        })  # fmt: skip
        family += [{"_id": f"{rid}-{i}", "residentId": rid, "name": m.get("name"), "relation": m.get("role"),
                    "type": "Family"} for i, m in enumerate(r.get("family_members") or [])]  # fmt: skip
    return MemoryResidentRepository(records, family, list(staff.values()))


class SnfMcpRepository:
    """The repository interface (app/data/repository.py), routed by the token user's role (see module docstring)."""

    name = "snf-mcp"

    per_caller = True  # data depends on the request's user and facility (api/routes.py, engine/service.py)

    def __init__(
        self, url: str, timeout_seconds: float, log_responses: bool = False, tools_file: Path = STAFF_TOOLS_FILE,
        cache_seconds: float = 60.0,
    ):  # fmt: skip
        self._url, self._timeout, self._log_responses = url, timeout_seconds, log_responses
        self._tools_cfg = yaml.safe_load(tools_file.read_text(encoding="utf-8"))
        self._admin = _AdminQueries(lambda tool, args: self._call(tool, args))
        self._cache = TTLCache(1024, cache_seconds)  # per user + facility (the token is never a key itself)

    @staticmethod
    def _credentials() -> tuple[str, str]:
        creds = _CREDENTIALS.get()
        if not creds:
            raise NoCredentials("no access token and facility for this snf_mcp call")
        return creds

    @property
    def _facility(self) -> str:
        return self._credentials()[1]

    async def init(self) -> None:
        return None  # nothing to load without a user: every request brings its own token and facility

    async def close(self) -> None:
        return None

    # ------------------------------------------------------------------ who is asking, and which path
    async def _cached(self, key: str, load):
        token, facility = self._credentials()
        user = hashlib.sha256(f"{token}|{facility}".encode()).hexdigest()[:24]
        hit = self._cache.get(f"{user}|{key}")
        if hit is None:
            hit = await load()
            self._cache.set(f"{user}|{key}", hit)
        return hit

    async def caller(self) -> dict:
        """The token's user (c_name, groups, facility_id, profile), from get_caller."""
        return await self._cached("caller", lambda: self._call("get_caller", {}))

    async def _is_admin(self) -> bool:
        return bool(ADMIN_GROUPS & set((await self.caller()).get("groups") or []))

    async def for_plan(self, plan):
        """The repository that runs this plan: run_query for admins; for staff, the configured API tool, after
        checking that it exists and provides every field the plan uses."""
        if await self._is_admin():
            return self._admin
        subject = plan.entity or "resident"
        cfg = self._tools_cfg.get(subject) or {}
        if cfg.get("tool") not in await self._tool_names():
            raise NotSupported(f"no snf_mcp tool for staff questions about {subject}")
        allowed, data = set(cfg.get("fields") or []), None
        if subject != "staff":
            data = await self._staff_residents()
            rows = await data.find(Filters())
            allowed |= {f for f, which in (cfg.get("fields_when_populated") or {}).items() if _populated(rows, f, which)}
        missing = plan_fields(plan) - allowed
        for field, values in (cfg.get("only_values") or {}).items():  # values the tool never returns
            f = getattr(plan.filters, field, None)
            if f is not None and {*(f.include or []), *(f.exclude or [])} - set(values):
                missing.add(field)
        if missing:
            raise NotSupported(f"not available to staff through {cfg['tool']}: {', '.join(sorted(missing))}",
                               tuple(sorted(missing)))
        return data if data is not None else await self._staff_directory(plan, cfg)

    async def bed_rooms(self) -> list[dict]:
        """The facility's `beds` documents (one per room: roomId, beds, updatedAt), through run_query. Admins only:
        staff have no Care App API tool for beds, so their bed questions are "not supported"."""
        if not await self._is_admin():
            raise NotSupported("no snf_mcp tool for staff questions about beds", ("beds",))
        pipeline = [{"$project": {"_id": 0, "roomId": 1, "beds": 1, "updatedAt": 1}}]
        return (await self._call("run_query", {"collection": "beds", "pipeline": pipeline})).get("documents") or []

    @property
    def staff_suggestions(self) -> list[dict]:
        """Follow-up questions offered to staff (staff_tools.yaml `suggestions`): ones their tools can answer."""
        return list(self._tools_cfg.get("suggestions") or [])

    async def _active(self):
        return self._admin if await self._is_admin() else await self._staff_residents()

    # ------------------------------------------------------------------ staff: API tools only
    async def _tool_names(self) -> set[str]:
        return await self._cached("tools", self._list_tools)

    async def _staff_residents(self) -> MemoryResidentRepository:
        cfg = self._tools_cfg["resident"]

        async def load():
            if cfg["tool"] not in await self._tool_names():
                raise NotSupported(f"snf_mcp has no '{cfg['tool']}' tool")
            rows = await self._pages(cfg["tool"], cfg.get("args") or {}, "residents")
            return _caseload(rows, self._facility, await self.caller())

        return await self._cached("staff-residents", load)

    async def _staff_directory(self, plan, cfg: dict) -> MemoryResidentRepository:
        """find_staff per designation the plan names (the API lists only supervisors without one)."""
        f = plan.filters
        designations = set(f.staff_designation.include if f.staff_designation else [])
        roles = staff_config().roles
        for role in f.staff_role.include if f.staff_role else []:
            designations.update(roles.get(role, []))
        if not designations:
            raise NotSupported("staff directory questions from staff need a role or designation")
        people: dict[str, dict] = {}
        for d in sorted(designations):
            for s in await self._pages(cfg["tool"], {"designation": d, **(cfg.get("args") or {})}, "staff"):
                people.setdefault(s["name"], {"cName": s["name"], "name": s["name"], "designation": s.get("designation"),
                                              "active": True})  # fmt: skip
        return MemoryResidentRepository([], [], list(people.values()))

    async def _pages(self, tool: str, args: dict, key: str) -> list[dict]:
        rows: list[dict] = []
        for page in range(1, MAX_PAGES + 1):
            data = await self._call(tool, {**args, "page": page})
            rows += data.get(key) or []
            if not data.get("has_more"):
                break
        return rows

    # ------------------------------------------------------------------ the repository interface
    async def vocabulary(self) -> Vocabulary:
        """Names and values the caller may see. Admins also get every status and care type of the facility
        (get_field_values); staff the configured statuses, so a value none of their residents has is still understood."""
        if _CREDENTIALS.get() is None:
            return Vocabulary()  # outside a request (startup): each caller's names are loaded with their first question
        try:
            if await self._is_admin():
                vocab = await self._admin.vocabulary()
                values = (await self._call("get_field_values", {})).get("values") or {}
            else:
                vocab = await (await self._staff_residents()).vocabulary()
                values = self._tools_cfg.get("values") or {}
        except DataUnavailable as exc:  # no names to learn; the questions themselves get "not supported"
            log.warning("vocabulary not loaded: %s", exc)
            return Vocabulary()
        vocab.statuses = sorted({*vocab.statuses, *(v.lower() for v in values.get("status") or [])})
        vocab.care_types = sorted({*vocab.care_types, *(values.get("careType") or [])})
        return vocab

    def __getattr__(self, name: str):
        """Every other query method (count, find, group_count, staff_*, ...) runs on the caller's path."""
        if name.startswith("_") or not callable(getattr(MemoryResidentRepository, name, None)):
            raise AttributeError(name)

        async def call(*args, **kwargs):
            return await getattr(await self._active(), name)(*args, **kwargs)

        return call

    # ------------------------------------------------------------------ MCP over HTTP
    async def _call(self, tool: str, args: dict) -> dict:
        result = await self._session(lambda client: client.call_tool(tool, args))
        if result.is_error:  # an unknown tool also comes back as an error result
            raise DataUnavailable(" ".join(getattr(c, "text", "") for c in result.content)[:300] or f"'{tool}' failed")
        data = result.structured_content or {}
        if self._log_responses:  # development only (SNF_MCP_LOG_RESPONSES): resident data
            log.info("snf_mcp response", extra={"tool": tool, "tool_args": args, "response": data})
        return data

    def _headers(self) -> dict[str, str]:
        token, facility = self._credentials()
        return {"Authorization": f"Bearer {token}", "x-facility-id": facility}

    async def _list_tools(self) -> set[str]:
        return {t.name for t in (await self._session(lambda client: client.list_tools())).tools}

    async def _session(self, action):
        result = None  # raised only after the client closed: inside it, errors come back wrapped in an ExceptionGroup
        try:
            async with (
                create_mcp_http_client(headers=self._headers()) as http,
                Client(streamable_http_client(self._url, http_client=http), read_timeout_seconds=self._timeout) as client,
            ):
                result = await action(client)
        except Exception as exc:  # server down, wrong URL (404), token rejected or expired (401), protocol error
            root = exc
            while isinstance(root, BaseExceptionGroup) and root.exceptions:
                root = root.exceptions[0]
            raise DataUnavailable(
                f"snf_mcp call to {self._url} failed: {type(root).__name__}: {root} (check SNF_MCP_URL, SNF_MCP_TOKEN)"
            ) from exc
        if result is None:
            raise DataUnavailable("snf_mcp returned nothing")
        return result
