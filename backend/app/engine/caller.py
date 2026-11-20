"""Who is asking: the facility scope and family-data permissions applied to ONE question.

The API has no trusted caller identity yet (one shared API_TOKEN), so today a caller is built from Settings: no
facility restriction and the configured family permissions. When trusted identity exists, the API builds a
CallerContext per request from it and passes it to QueryService.handle; nothing else changes.

Permissions are separate on purpose: reading family names and relationships does not grant phone and email.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class CallerContext:
    facilities: frozenset[str] | None = None  # allowed resident facilityIds; None = unrestricted
    can_read_family: bool = True  # family members' names and relationships
    can_read_family_contacts: bool = True  # family members' phone and email (single-resident answers only)

    @classmethod
    def from_settings(cls, settings) -> "CallerContext":
        ids = settings.caller_facility_ids
        return cls(
            facilities=frozenset(ids) if ids is not None else None,
            can_read_family=settings.family_names_permitted,
            can_read_family_contacts=settings.family_names_permitted and settings.family_contacts_permitted,
        )

    def allows_facility(self, facility_id) -> bool:
        return self.facilities is None or (facility_id is not None and str(facility_id) in self.facilities)


# the caller of the question being handled; read by resolution, execution and the answer (one per request)
_CALLER: ContextVar[CallerContext | None] = ContextVar("caller", default=None)


@contextmanager
def use_caller(caller: CallerContext) -> Iterator[None]:
    token = _CALLER.set(caller)
    try:
        yield
    finally:
        _CALLER.reset(token)


def current_caller() -> CallerContext:
    return _CALLER.get() or CallerContext()
