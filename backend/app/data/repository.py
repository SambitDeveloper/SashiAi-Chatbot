"""Storage abstraction. The engine only talks to this interface, so the data store can change
(seed JSON in development, MongoDB in production) without touching understanding or answering logic."""

from dataclasses import dataclass, field
from typing import Any, Protocol

from app.models.plan import Filters, GroupField, Sort

# Plan field name → record field name (records use the Care App's camelCase JSON fields)
FIELD_MAP: dict[str, str] = {
    "status": "status",
    "care_type": "careType",
    "wing": "wing",
    "gender": "gender",
    "primary_caregiver": "primaryCaregiver",
    "room_number": "roomNumber",
    "age": "age",
    "admission_date": "admissionDate",
    "created_date": "createdDate",
    "discharge_date": "dischargeDate",
    "last_name": "lastName",
}

Resident = dict[str, Any]


@dataclass
class Vocabulary:
    """Real values from the database, used to mask names and resolve what the model wrote."""

    residents: list[dict[str, str]] = field(default_factory=list)  # residentId, firstName, lastName
    wings: list[str] = field(default_factory=list)
    caregivers: list[str] = field(default_factory=list)
    rooms: list[str] = field(default_factory=list)


@dataclass
class GroupRow:
    key: tuple[str, ...]
    count: int


class ResidentRepository(Protocol):
    name: str

    async def init(self) -> None: ...
    async def close(self) -> None: ...
    async def total(self) -> int: ...
    async def count(self, filters: Filters) -> int: ...
    async def find(self, filters: Filters, sort: Sort | None = None, limit: int | None = None) -> list[Resident]: ...
    async def find_by_ids(self, resident_ids: list[str]) -> list[Resident]: ...
    async def group_count(self, filters: Filters, fields: list[GroupField]) -> list[GroupRow]: ...
    async def aggregate(self, filters: Filters, op: str, field_name: str) -> float | None: ...
    async def vocabulary(self) -> Vocabulary: ...


def matches(record: Resident, filters: Filters) -> bool:
    """In-process filter semantics. `mongo_repository.to_mongo_filter` must mirror this exactly."""
    for plan_field in ("status", "care_type", "wing", "gender", "primary_caregiver", "room_number"):
        f = getattr(filters, plan_field)
        if f is None:
            continue
        value = record.get(FIELD_MAP[plan_field])
        if f.include and value not in f.include:
            return False
        if f.exclude and value in f.exclude:
            return False

    if filters.age:
        age = record.get("age")
        if age is None:
            return False
        if filters.age.min is not None and age < filters.age.min:
            return False
        if filters.age.max is not None and age > filters.age.max:
            return False

    for plan_field in ("admission_date", "created_date", "discharge_date"):
        r = getattr(filters, plan_field)
        if r is None:
            continue
        value = record.get(FIELD_MAP[plan_field])
        if not value:
            return False
        if r.start and value < r.start.isoformat():
            return False
        if r.end and value > r.end.isoformat():
            return False
    return True


def sort_key(field_name: str):
    record_field = FIELD_MAP[field_name]
    return lambda r: (r.get(record_field) is None, r.get(record_field) or "", r.get("residentId", ""))


def list_view(record: Resident) -> Resident:
    """Bulk results never include emergency contact details; those are only returned for one resident."""
    out = dict(record)
    out["emergencyContact"] = {"name": "", "relationship": "", "phone": ""}
    return out
