"""Facility scope of the CURRENT request: which residents (and so which family members) the caller may see.

Set by the service for the duration of one question (`facility_scope(...)`), read by both repositories, so every
query, lookup, count and vocabulary candidate of that request is limited the same way, without changing any repository
signature. None = no restriction (today's behaviour: the API has no caller identity yet; see Settings).

Semantics, identical in MongoDB and in memory: a resident is in scope when its `facilityId` (the resident document's
own field, a string) is one of the allowed ids. A resident without a facilityId is out of scope whenever a scope is set.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_SCOPE: ContextVar[frozenset[str] | None] = ContextVar("facility_scope", default=None)


@contextmanager
def facility_scope(facility_ids) -> Iterator[None]:
    token = _SCOPE.set(frozenset(str(f) for f in facility_ids) if facility_ids is not None else None)
    try:
        yield
    finally:
        _SCOPE.reset(token)


def current() -> frozenset[str] | None:
    return _SCOPE.get()


def in_scope(facility_id) -> bool:
    allowed = _SCOPE.get()
    return allowed is None or (facility_id is not None and str(facility_id) in allowed)


def mongo_match() -> dict | None:
    """The $match on the raw resident document, or None when unrestricted."""
    allowed = _SCOPE.get()
    return None if allowed is None else {"$match": {"facilityId": {"$in": sorted(allowed)}}}
