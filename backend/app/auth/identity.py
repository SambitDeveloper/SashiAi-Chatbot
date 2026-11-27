"""Who is asking: the user of the SNF_MCP_TOKEN access token, as our snf_mcp server reports them → a Principal.

Every `run_query` result returns `caller`: the token's username (cName) and Cognito groups, and the user's own staff
record from the database (active, name, accessPermissions; for contact staff, their membership at this facility). The
role comes from the groups, the facility is SNF_MCP_FACILITY_ID, which snf_mcp has already checked. Field names and role values are access_policy.yaml `principal.context`. Nothing is taken from what
a browser says about itself.
"""

from app.auth.policy import AccessPolicy, access_policy
from app.auth.principal import Principal, readable_modules


class AuthError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def principal_from_context(data: dict, policy: AccessPolicy | None = None) -> Principal:
    """The user's data → Principal. An admin needs no permissions; a staff member's come from their permissions list."""
    p = policy or access_policy()
    ctx = dict(p.setting("context") or {})
    if not isinstance(data, dict):
        raise AuthError(401, "Unknown caller.")
    role = str(data.get(ctx["role_field"]) or "").strip().lower()
    key = str(data.get(ctx["key_field"]) or "").strip()
    facility = data.get(ctx["facility_field"])
    active = data.get(ctx.get("active_field", "active"), True) is not False
    if role in [str(r).lower() for r in ctx.get("admin_roles") or []]:
        kind = "admin"
    elif role in [str(r).lower() for r in ctx.get("staff_roles") or []]:
        kind = "staff"
    else:
        raise AuthError(401, "Unknown caller role.")
    if not key:
        raise AuthError(401, "Unknown caller.")
    if kind == "staff" and not facility:
        raise AuthError(401, "The caller has no active facility.")
    principal = Principal(
        kind=kind, id=key, facility_id=str(facility) if facility else None, active=active,
        modules=readable_modules(data.get(ctx["permissions_field"]), p.allowed_when) if kind == "staff" else frozenset(),
        scope_key=key if kind == "staff" else None, name=str(data.get(ctx.get("name_field", "name")) or ""),
    )  # fmt: skip
    if not active and (kind != "staff" or p.setting("deny_inactive_staff", True)):
        raise AuthError(403, "This account is not active.")
    return principal


def principal_from_caller(caller: dict, policy: AccessPolicy | None = None) -> Principal:
    """snf_mcp's `caller` (+ facility_id) → Principal. A staff member without a staff record reads nothing."""
    p = policy or access_policy()
    ctx = dict(p.setting("context") or {})
    groups = {str(g).lower() for g in caller.get("groups") or []}
    roles = [*(ctx.get("admin_roles") or []), *(ctx.get("staff_roles") or [])]  # admin first
    role = next((r for r in roles if str(r).lower() in groups), "")
    profile = caller.get("profile") or {}
    return principal_from_context({
        ctx["role_field"]: role, ctx["key_field"]: caller.get("c_name") or "",
        ctx["facility_field"]: caller.get("facility_id"), ctx["permissions_field"]: profile.get("access_permissions"),
        ctx.get("active_field", "active"): profile.get("active", True), ctx.get("name_field", "name"): profile.get("name") or "",
    }, p)  # fmt: skip
