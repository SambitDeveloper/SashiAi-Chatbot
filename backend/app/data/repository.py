"""Storage abstraction. The engine only talks to this interface: the Care App's MongoDB (through the resident view in
resident_schema.yaml) in every environment, and an in-memory list in tests."""

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Protocol

from app.data import text_match
from app.data.relations import FAMILY_VIEW_FIELDS
from app.data.schema_config import identity_fields, is_missing, list_fields
from app.data.staff import STAFF_RESIDENT_FIELDS
from app.models.plan import Filters, GroupField, Sort

# Plan field name → field name in the resident view (app/data/resident_schema.yaml `fields`)
FIELD_MAP: dict[str, str] = {
    "status": "status",
    "care_type": "careType",
    "facility": "wing",  # plan field → view field (the view keeps the key "wing" for API compatibility)
    "gender": "gender",
    "primary_caregiver": "primaryCaregiver",
    "room_number": "roomNumber",
    "age": "age",
    "admission_date": "admissionDate",
    "created_date": "createdDate",
    "discharge_date": "dischargeDate",
    "last_name": "lastName",
    # derived from the `family` relation (app/data/relations.py), present when a query uses them
    "family_relation": "familyRelations",
    "family_type": "familyTypes",
    "family_flags": "familyFlags",
    "family_count": "familyCount",
    "family_members": "familyMembers",
    "family_member": "familyIds",  # specific family members, by record id (resolved server side, never by name)
    # derived from the `staff` relation (app/data/staff.py): the resident's ACTIVE assigned staff
    "staff_role": "staffRoles",
    "staff_designation": "staffDesignations",
    "staff_count": "staffCount",
    "staff_name": "staffNames",  # any assigned staff member (recorded names, resolved by the backend)
}
# the plan's set-valued fields, in the order both filter engines (matches / to_mongo_filter) apply them
SET_FILTER_FIELDS = ("status", "care_type", "facility", "gender", "primary_caregiver", "room_number",
                     "family_relation", "family_type", "family_flags", "family_member", "staff_role",
                     "staff_designation", "staff_name")  # fmt: skip

Resident = dict[str, Any]


class DataUnavailable(RuntimeError):
    """The data source has no tool for this data, or refused the call: the user is told it is not supported."""


class NotSupported(DataUnavailable):
    """The caller MAY ask this, but the tools their role uses cannot provide it (a missing tool, field or value): a
    data limitation, never an authorization denial. `fields`: the plan fields that are not provided."""

    def __init__(self, message: str, fields: tuple[str, ...] = ()):
        super().__init__(message)
        self.fields = fields


@dataclass
class Vocabulary:
    """Real values from the database, used to mask names and resolve what the model wrote."""

    residents: list[dict[str, str]] = field(default_factory=list)  # residentId, firstName, lastName
    wings: list[str] = field(default_factory=list)
    caregivers: list[str] = field(default_factory=list)
    rooms: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)  # values present in the database (empty = unknown)
    care_types: list[str] = field(default_factory=list)
    family_relations: list[str] = field(default_factory=list)  # normalised relation values (familymembers)
    # family members of live residents for the entity index: {id, name, rel, relation, residentId, facilityId}
    family_members: list[dict] = field(default_factory=list)
    # every staff member who is a person (app/data/staff.py), any status: {name, designation, role, status}
    staff: list[dict] = field(default_factory=list)


@dataclass
class GroupRow:
    key: tuple[str, ...]
    count: int


@dataclass
class GroupAggregateRow:
    """One group of a grouped aggregate ("average age by status").

    count     residents in the group (every resident, whether or not the aggregated field is recorded)
    value     the aggregate over the residents whose field IS recorded as a number; None when none of them is
    n_values  how many residents the value is computed from (count - n_values = field not recorded)
    """

    key: tuple[str, ...]
    count: int
    value: float | None
    n_values: int


# Missing-value rule for every grouping (breakdowns and grouped aggregates), identical in MongoDB and in memory:
# a resident whose grouped field is not recorded (absent, null, "", an empty list, or a placeholder declared in
# resident_schema.yaml `missing_values`, e.g. status "unknown") is counted in ONE group with this key — never dropped,
# never split into "unknown"/"Unknown"/"" variants, never turned back into a filter value.
MISSING_GROUP = "Not recorded"


def group_key(value: Any, view_field: str) -> str:
    """A text or number is its own group; anything else (a list or object where one value belongs, a bool) is "not
    recorded" — the same as the MongoDB key expression in resident_view._grouping ($convert onError)."""
    if is_missing(value, view_field) or not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return MISSING_GROUP
    return str(value)


def group_keys(record: Resident, fields: list[GroupField]) -> list[tuple[str, ...]]:
    """Return every group key for a resident, including one key per value of list-valued fields."""
    per_field = []
    for field_name in fields:
        view_field = FIELD_MAP[field_name]
        value = record.get(view_field)
        if view_field in list_fields() and isinstance(value, list) and value:
            per_field.append([group_key(item, view_field) for item in value])
        else:
            per_field.append([group_key(value, view_field)])
    return list(product(*per_field))


def numeric(value: Any) -> float | int | None:
    """A value an aggregate may use: a real number (not a bool, not a numeric-looking string). Anything else is
    treated as "not recorded", as MongoDB's $avg/$min/$max ignore non-numeric values."""
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


class ResidentRepository(Protocol):
    name: str

    async def init(self) -> None: ...
    async def close(self) -> None: ...
    async def total(self) -> int: ...
    async def count(self, filters: Filters) -> int: ...
    async def find(
        self, filters: Filters, sort: Sort | None = None, limit: int | None = None, fields: list[str] | None = None
    ) -> list[Resident]: ...
    async def find_by_ids(self, resident_ids: list[str], contacts: bool = True) -> list[Resident]: ...
    async def group_count(self, filters: Filters, fields: list[GroupField]) -> list[GroupRow]: ...
    async def aggregate(self, filters: Filters, op: str, field_name: str) -> float | None: ...
    async def group_aggregate(
        self, filters: Filters, fields: list[GroupField], op: str, field_name: str
    ) -> list[GroupAggregateRow]: ...
    async def vocabulary(self) -> Vocabulary: ...


def matches(record: Resident, filters: Filters) -> bool:
    """In-process filter semantics. `resident_view.to_mongo_filter` must mirror this exactly.

    A list value (several assigned staff) matches an include when ANY element is included, like MongoDB's $in.
    Within one field, `include` OR `missing: true`; `exclude` and `missing: false` are AND-ed. What counts as missing
    comes from resident_schema.yaml (schema_config.is_missing), the same source the MongoDB filter uses.
    """
    for plan_field in SET_FILTER_FIELDS:
        f = getattr(filters, plan_field)
        if f is None:
            continue
        view_field = FIELD_MAP[plan_field]
        value = record.get(view_field)
        values = value if isinstance(value, list) else [value]
        missing = is_missing(value, view_field)
        if f.include or f.missing is True:
            hit = bool(f.include) and any(v in f.include for v in values)
            if not (hit or (f.missing is True and missing)):
                return False
        if f.exclude and any(v in f.exclude for v in values):
            return False
        if f.missing is False and missing:
            return False
        if f.has_text_match and not text_match.value_matches(value, view_field, f.starts_with, f.contains):
            return False

    checks: list[tuple[str, Any, Any, Any]] = []
    if filters.age:
        checks.append(("age", filters.age, filters.age.min, filters.age.max))
    if filters.family_count:
        checks.append(("familyCount", filters.family_count, filters.family_count.min, filters.family_count.max))
    if filters.staff_count:
        checks.append(("staffCount", filters.staff_count, filters.staff_count.min, filters.staff_count.max))
    for plan_field in ("admission_date", "created_date", "discharge_date"):
        r = getattr(filters, plan_field)
        if r is not None:
            checks.append(
                (FIELD_MAP[plan_field], r, r.start.isoformat() if r.start else None, r.end.isoformat() if r.end else None)
            )
    for view_field, r, lo, hi in checks:
        value = record.get(view_field)
        missing = is_missing(value, view_field)
        if r.has_range:
            # a value of another type than the bound (text where a number belongs) never satisfies a range, like
            # MongoDB's type-bracketed comparisons — and never raises
            in_range = not missing and _comparable(value, lo, hi) and (lo is None or value >= lo) and (hi is None or value <= hi)
            if not (in_range or (r.missing is True and missing)):
                return False
        elif r.missing is not missing:  # missing-only condition: true → must be missing, false → must be recorded
            return False
    return True


def project(record: Resident, fields: list[str] | None) -> Resident:
    """Keep the identity fields plus the requested view fields (lists and nested objects whole). None → unchanged.
    Mirrors the `$project` stage ViewRepository.find adds for a projection."""
    if fields is None:
        return record
    keep = [*identity_fields(), *fields]
    return {k: record[k] for k in keep if k in record}


def _comparable(value: Any, *bounds: Any) -> bool:
    numbers = (int, float)
    for b in bounds:
        if b is None:
            continue
        if isinstance(b, numbers) and not (isinstance(value, numbers) and not isinstance(value, bool)):
            return False
        if isinstance(b, str) and not isinstance(value, str):
            return False
    return True


def sort_key(field_name: str):
    record_field = FIELD_MAP[field_name]
    return lambda r: (r.get(record_field) is None, r.get(record_field) or "", r.get("residentId", ""))


def typed_order(value: Any) -> tuple:
    """Sort key for one value: numbers, then text, then anything else (MongoDB's type order). Never raises a
    TypeError when a malformed text value sits among numbers."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, float(value), "")
    if isinstance(value, str):
        return (1, 0.0, value)
    return (2, 0.0, str(value))


def list_view(record: Resident) -> Resident:
    """Bulk results never include emergency contact details; those are only returned for one resident."""
    out = dict(record)
    out["emergencyContact"] = {"name": "", "relationship": "", "phone": ""}
    for key in FAMILY_VIEW_FIELDS:  # family details only when asked for (projection) or for one resident
        if key != "familyCount":  # the number of family members is not personal data (shown when it was joined)
            out.pop(key, None)
    for key in STAFF_RESIDENT_FIELDS:  # staff details only when asked for; the number of assigned staff is not
        if key != "staffCount":
            out.pop(key, None)
    return out
