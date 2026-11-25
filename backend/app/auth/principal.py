"""The caller of ONE question, built server-side (app/auth/identity.py): role, readable app modules and resident scope.

Never built from what a client says about its permissions: the identity is verified (or, in test mode, picked from the
test principals file) and everything else is loaded from the database.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Literal

from app.auth.policy import AccessPolicy, access_policy, norm_module
from app.data.scope import Assignment
from app.engine.caller import CallerContext

Kind = Literal["system", "admin", "staff"]


def readable_modules(permissions: list | None, allowed_when: tuple[str, ...]) -> frozenset[str]:
    """accessPermissions → the normalised names of every module the user can read (a child only with its parent)."""

    def ok(p) -> bool:
        return isinstance(p, dict) and all(p.get(flag) is True for flag in allowed_when)

    out: set[str] = set()
    for top in permissions or []:
        if not ok(top):
            continue
        out.add(norm_module(top.get("name")))
        out.update(norm_module(c.get("name")) for c in top.get("children") or [] if ok(c))
    out.discard("")
    return frozenset(out)


@dataclass(frozen=True)
class Principal:
    kind: Kind
    id: str  # admins / staffs key (cName); "system" when there is no caller identity
    facility_id: str | None
    active: bool = True
    modules: frozenset[str] = field(default_factory=frozenset)  # readable, normalised; ignored for system / admin
    scope_key: str | None = None  # staff: their key in the residents' assigned-staff field
    name: str = ""

    @property
    def unrestricted(self) -> bool:
        return self.kind in ("system", "admin")

    def can(self, capability: str, policy: AccessPolicy | None = None) -> bool:
        if self.unrestricted:
            return True
        cap = (policy or access_policy()).capabilities[capability]
        return all(m in self.modules for m in cap.modules)

    def facilities(self, policy: AccessPolicy | None = None) -> frozenset[str] | None:
        """The resident facility scope: None = every facility."""
        if self.kind == "system":
            return None
        if self.kind == "admin" and (policy or access_policy()).setting("admin_facility_scope", "all") == "all":
            return None
        return frozenset([self.facility_id]) if self.facility_id else frozenset()

    def assignment(self, policy: AccessPolicy | None = None) -> Assignment | None:
        """R3: a staff member sees the residents whose assigned-staff field holds their key."""
        if self.kind != "staff":
            return None
        p = policy or access_policy()
        return Assignment(str(p.setting("resident_link")), self.scope_key or f"<no key for {self.id}>")

    def caller(self, base: CallerContext, policy: AccessPolicy | None = None) -> CallerContext:
        """The engine's CallerContext: facility scope and family permissions, never wider than `base` (Settings)."""
        p = policy or access_policy()
        facilities = self.facilities(p)
        if base.facilities is not None:
            facilities = base.facilities if facilities is None else facilities & base.facilities
        return CallerContext(
            facilities=facilities,
            can_read_family=base.can_read_family and self.can("family", p),
            can_read_family_contacts=base.can_read_family_contacts and self.can("family_contacts", p),
        )

    def policy_version(self, policy: AccessPolicy | None = None) -> str:
        """Everything that decides what this caller may see: part of cache and session keys."""
        p = policy or access_policy()
        body = "|".join([p.version, self.kind, self.id, self.facility_id or "", ",".join(sorted(self.modules)),
                         self.scope_key or ""])  # fmt: skip
        return hashlib.sha256(body.encode()).hexdigest()[:16]


SYSTEM = Principal(kind="system", id="system", facility_id=None)
