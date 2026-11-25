"""Who is asking: the signed-in user's data, as the Shashi Care app holds it → a Principal (app/auth/principal.py).

The app already has, for every signed-in user, their key (cName), active facility (facilityId), role (scope: admin /
staff) and app permissions (accessPermissions). The chatbot builds its principal from exactly that data; the field
names and role values are access_policy.yaml `principal.context`.

AUTH_MODE (Settings.auth_mode):
  off   no caller (today's behaviour): the unrestricted system principal.
  file  until the chatbot is integrated into the app: every request is the user in Settings.auth_user_context_file
        (re-read when the file changes, so a test can switch user or role without a restart). Refused in production.

When integrated, the app's BACKEND will send the same data with each request (server to server, behind API_TOKEN):
never accept it from a browser, which could claim any role or permission.
"""

import json
from pathlib import Path

from app.auth.policy import AccessPolicy, access_policy
from app.auth.principal import SYSTEM, Principal, readable_modules


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


class IdentityResolver:
    """The API's single way to turn a request into a Principal."""

    def __init__(self, settings, policy: AccessPolicy | None = None):
        self.mode = settings.auth_mode
        self.policy = policy or access_policy()
        self.path = Path(settings.auth_user_context_file)
        self._loaded: tuple[float, Principal] | None = None

    async def resolve(self, headers=None) -> Principal:
        if self.mode == "off":
            return SYSTEM
        if self.mode == "file":
            return self._from_file()
        raise AuthError(401, "Caller identity is not configured.")

    def _from_file(self) -> Principal:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            raise AuthError(401, "The user context file is missing.") from None
        if self._loaded is None or self._loaded[0] != mtime:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except ValueError:
                raise AuthError(401, "The user context file is not valid JSON.") from None
            self._loaded = (mtime, principal_from_context(data, self.policy))
        return self._loaded[1]
