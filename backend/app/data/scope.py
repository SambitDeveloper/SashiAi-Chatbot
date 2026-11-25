"""Row scope of the CURRENT request: which residents (and so which family members, care teams, ...) the caller may see.

Set by the service for the duration of one question, read by both repositories, so every query, lookup, count, grouping
and vocabulary candidate of that request is limited the same way, without changing any repository signature.

Two independent parts, both None = no restriction (an admin, or no caller identity at all):
  • facility scope (`facility_scope`): the resident's `facilityId` (a string on the resident document) is one of the
    allowed ids. A resident without a facilityId is out of scope whenever a facility scope is set.
  • assignment scope (`assignment_scope`): the resident's assigned-staff field (resident_schema.yaml
    relations.staff.resident_link, e.g. `assignedStaff`) contains the caller's staff key (their `cName`). This is the
    staff guardrail of app/auth: "a staff member sees only their own residents".

Semantics are identical in MongoDB (`mongo_match`, on the raw resident document, right after the base match and
before any join) and in memory (`record_in_scope`).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class Assignment:
    field: str  # resident field holding staff keys (a list or a single value)
    staff_key: str  # the caller's key in that field


_SCOPE: ContextVar[frozenset[str] | None] = ContextVar("facility_scope", default=None)
_ASSIGNED: ContextVar[Assignment | None] = ContextVar("assignment_scope", default=None)


@contextmanager
def facility_scope(facility_ids) -> Iterator[None]:
    token = _SCOPE.set(frozenset(str(f) for f in facility_ids) if facility_ids is not None else None)
    try:
        yield
    finally:
        _SCOPE.reset(token)


@contextmanager
def assignment_scope(assignment: Assignment | None) -> Iterator[None]:
    token = _ASSIGNED.set(assignment)
    try:
        yield
    finally:
        _ASSIGNED.reset(token)


def current() -> frozenset[str] | None:
    return _SCOPE.get()


def current_assignment() -> Assignment | None:
    return _ASSIGNED.get()


def is_restricted() -> bool:
    return _SCOPE.get() is not None or _ASSIGNED.get() is not None


def in_scope(facility_id) -> bool:
    allowed = _SCOPE.get()
    return allowed is None or (facility_id is not None and str(facility_id) in allowed)


def record_in_scope(record: dict) -> bool:
    """In memory, the same as `mongo_match` (facility AND assignment)."""
    if not in_scope(record.get("facilityId")):
        return False
    a = _ASSIGNED.get()
    if a is None:
        return True
    value = record.get(a.field)
    values = value if isinstance(value, list) else [value]
    return a.staff_key in [str(v) for v in values if v is not None]


def mongo_match() -> dict | None:
    """The $match on the raw resident document, or None when unrestricted."""
    allowed, a = _SCOPE.get(), _ASSIGNED.get()
    if allowed is None and a is None:
        return None
    cond: dict = {}
    if allowed is not None:
        cond["facilityId"] = {"$in": sorted(allowed)}
    if a is not None:
        cond[a.field] = a.staff_key  # matches a list element or a single value, as MongoDB does
    return {"$match": cond}
