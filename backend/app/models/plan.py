"""QueryPlan — the restricted executable contract produced by the RequestIR compiler.

The semantic interpreter and deterministic fast path produce a candidate RequestIR. A conservative compiler lowers
only supported, lossless operations to QueryPlan; the plan is resolved against real database values and validated
before anything touches the data store. Every number in an answer comes from the database.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Status = Literal["active", "away", "inactive", "pending", "admitted", "discharged"]
CareType = Literal["assisted_living", "memory_care", "independent_living", "skilled_nursing", "respite_care"]
Gender = Literal["Female", "Male", "Other"]
ResidentGroupField = Literal["status", "care_type", "facility", "gender", "primary_caregiver"]
# Staff (staffs collection; app/data/staff.py). A plan is about residents (default) or about staff members (`entity`).
# The staff_* fields describe a STAFF MEMBER: in a resident plan the resident's assigned staff (ANY of them), in a staff
# plan the staff member itself; in a staff plan every other condition applies to the staff member's assigned residents.
Entity = Literal["resident", "staff"]
StaffStatus = Literal["active", "inactive"]
StaffType = Literal["provider", "facility_staff", "external_user"]
StaffGroupField = Literal["staff_role", "staff_designation", "staff_status", "staff_facility", "staff_type"]
GroupField = Literal[
    "status", "care_type", "facility", "gender", "primary_caregiver",
    "staff_role", "staff_designation", "staff_status", "staff_facility", "staff_type",
]  # fmt: skip
SortField = Literal[
    "admission_date", "created_date", "discharge_date", "age", "last_name", "family_count",
    "staff_name", "staff_resident_count",
]  # fmt: skip
# Family relation (familymembers; app/data/relations.py): the kinds of contact and the access flags of ANY member.
FamilyType = Literal["family", "emergency"]
FamilyFlag = Literal["portal_access", "app_access", "health_access", "guarantor", "has_email", "has_phone"]
Attribute = Literal[
    "primary_caregiver",
    "room_number",
    "facility",
    "admission_date",
    "created_date",
    "discharge_date",
    "emergency_contact",
    "resident_id",
    "status",
    "care_type",
    "age",
    "gender",
    "family_members",  # names + relations; one resident's answer also shows phone and email
    "family_count",
    "family_relationship",  # how the selected family member(s) are related to ONE looked-up resident
    "care_team",  # the resident's assigned staff, "Name (Designation)"; inactive staff marked
    "primary_physician",  # residents.primaryPhysicianCName (distinct from assigned staff and from designations)
    # a staff member's own facts (staff plans only)
    "staff_designation", "staff_role", "staff_status", "staff_facility", "staff_type", "staff_resident_count",
    "staff_email", "staff_phone", "staff_residents",  # one staff member's answer only
]
Action = Literal["count", "list", "breakdown", "aggregate", "lookup", "clarify", "unsupported", "help"]

# "wing" was the earlier name of `facility` (the data has no wing: the field is the resident's facility). It is still
# accepted wherever a plan is READ — cached plans, older clients, model output — and normalised to the one canonical
# name, so there is never a second way to express the condition.
LEGACY_FIELD_NAMES = {"wing": "facility"}


def canonical_field(name):
    return LEGACY_FIELD_NAMES.get(name, name) if isinstance(name, str) else name


RESIDENT_GROUP_FIELDS: tuple[str, ...] = ResidentGroupField.__args__  # type: ignore[attr-defined]
STAFF_GROUP_FIELDS: tuple[str, ...] = StaffGroupField.__args__  # type: ignore[attr-defined]
# staff-member conditions; in a resident plan only staff_role / staff_designation / staff_count (ANY active assigned staff)
STAFF_ENTITY_SET_FIELDS = ("staff_role", "staff_designation", "staff_status", "staff_facility", "staff_type", "staff_name")
STAFF_ONLY_FIELDS = ("staff_status", "staff_facility", "staff_type", "staff_name", "staff_resident_count")  # staff plans
STAFF_RELATION_FIELDS = ("staff_role", "staff_designation", "staff_count")  # resident plans: the assigned staff
STAFF_SORT_FIELDS = ("staff_name", "staff_resident_count")
STAFF_ATTRIBUTES = ("staff_designation", "staff_role", "staff_status", "staff_facility", "staff_type",
                    "staff_resident_count", "staff_email", "staff_phone", "staff_residents")  # fmt: skip
STAFF_SINGLE_RECORD = ("staff_email", "staff_phone", "staff_residents")
STAFF_RELATION_ATTRIBUTES = ("care_team", "primary_physician")
# plan field → field of the staff view (app/data/staff.py)
STAFF_FIELD_MAP = {
    "staff_role": "role", "staff_designation": "designation", "staff_status": "status", "staff_facility": "facility",
    "staff_type": "type", "staff_name": "name", "staff_resident_count": "residentCount", "staff_email": "email",
    "staff_phone": "phone", "staff_residents": "residents",
}  # fmt: skip
STATUSES: tuple[str, ...] = Status.__args__  # type: ignore[attr-defined]
CARE_TYPES: tuple[str, ...] = CareType.__args__  # type: ignore[attr-defined]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Text matching ("room starts with A", "facility contains Garden") is allowed on these free-text plan fields ONLY.
# Fixed-vocabulary fields (status, care type, gender) are matched by exact value. Semantics: app/data/text_match.py.
TEXT_MATCH_FIELDS: tuple[str, ...] = ("room_number", "facility", "primary_caregiver")
STAFF_TEXT_MATCH_FIELDS: tuple[str, ...] = ("staff_designation", "staff_facility", "staff_name")  # same semantics
ALL_TEXT_MATCH_FIELDS: tuple[str, ...] = (*TEXT_MATCH_FIELDS, *STAFF_TEXT_MATCH_FIELDS)
TEXT_MATCH_MAX_CHARS = 60


def _text_needles(values: list[str] | None, name: str) -> list[str] | None:
    if values is None:
        return None
    cleaned: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} needs non-empty text")
        if len(value) > TEXT_MATCH_MAX_CHARS or any(not ch.isprintable() for ch in value):
            raise ValueError(f"{name} text must be at most {TEXT_MATCH_MAX_CHARS} printable characters")
        cleaned.append(value.strip())
    return list(dict.fromkeys(cleaned))


class SetFilter[T](_Strict):
    """A condition on one field. `include` values are OR-ed; `missing: true` adds "or no value recorded" to them;
    `exclude` and `missing: false` are AND-ed on top ("recorded and not X").

    `starts_with` / `contains` (TEXT_MATCH_FIELDS only, enforced by Filters): case-insensitive literal text match. Within
    one list the needles are OR-ed ("starts with A or B"); each list is AND-ed with everything else. A value that is not
    recorded never matches a needle (app/data/text_match.py)."""

    include: list[T] | None = Field(default=None, max_length=20)
    exclude: list[T] | None = Field(default=None, max_length=20)
    missing: bool | None = None  # True: no value recorded · False: a value is recorded
    starts_with: list[str] | None = Field(default=None, min_length=1, max_length=10)
    contains: list[str] | None = Field(default=None, min_length=1, max_length=10)

    @model_validator(mode="after")
    def _consistent(self):
        self.starts_with = _text_needles(self.starts_with, "starts_with")
        self.contains = _text_needles(self.contains, "contains")
        has_text = bool(self.starts_with or self.contains)
        if not self.include and not self.exclude and self.missing is None and not has_text:
            raise ValueError("filter needs include, exclude, missing, starts_with or contains")
        if self.missing is True and self.exclude:
            raise ValueError("missing=true cannot be combined with exclude (a missing value never equals an excluded one)")
        if self.missing is False and self.include:
            raise ValueError("missing=false is implied by include; remove one of them")
        if self.missing is not None and has_text:
            # a value that is not recorded never matches a text: "or not recorded" would be dead, "recorded" implied
            raise ValueError("missing cannot be combined with starts_with/contains")
        if self.include and self.exclude and set(self.include) & set(self.exclude):
            raise ValueError("a value cannot be both included and excluded")
        return self

    @property
    def has_text_match(self) -> bool:
        return bool(self.starts_with or self.contains)


class CountRange(_Strict):
    """A number of related records per resident (always known: 0 when none is recorded). min/max are inclusive."""

    min: int | None = Field(default=None, ge=0, le=1000)
    max: int | None = Field(default=None, ge=0, le=1000)

    @model_validator(mode="after")
    def _order(self):
        if self.min is None and self.max is None:
            raise ValueError("count range needs min or max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("count min is greater than max")
        return self

    @property
    def has_range(self) -> bool:
        return True

    @property
    def missing(self) -> None:  # a count is never "not recorded" (shared range code reads .missing)
        return None


class AgeRange(_Strict):
    """min/max bound the age; `missing: true` adds "or no age recorded"; `missing: false` alone means "age recorded"."""

    min: int | None = Field(default=None, ge=0, le=130)
    max: int | None = Field(default=None, ge=0, le=130)
    missing: bool | None = None

    @model_validator(mode="after")
    def _order(self):
        has_range = self.min is not None or self.max is not None
        if not has_range and self.missing is None:
            raise ValueError("age range needs min, max or missing")
        if has_range and self.missing is False:
            raise ValueError("missing=false is implied by an age range; remove one of them")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("age min is greater than max")
        return self

    @property
    def has_range(self) -> bool:
        return self.min is not None or self.max is not None


class DateRange(_Strict):
    """start/end bound the date; `missing: true` adds "or no date recorded"; `missing: false` alone means "date recorded"."""

    start: date | None = None
    end: date | None = None
    missing: bool | None = None

    @model_validator(mode="after")
    def _order(self):
        has_range = self.start is not None or self.end is not None
        if not has_range and self.missing is None:
            raise ValueError("date range needs start, end or missing")
        if has_range and self.missing is False:
            raise ValueError("missing=false is implied by a date range; remove one of them")
        if self.start and self.end and self.start > self.end:
            raise ValueError("date range start is after end")
        return self

    @property
    def has_range(self) -> bool:
        return self.start is not None or self.end is not None


class Filters(_Strict):
    @model_validator(mode="before")
    @classmethod
    def _legacy_names(cls, data):
        if isinstance(data, dict) and any(k in data for k in LEGACY_FIELD_NAMES):
            data = dict(data)
            for old, new in LEGACY_FIELD_NAMES.items():
                if old in data:
                    if new in data:
                        raise ValueError(f"both '{old}' and '{new}' given; use '{new}'")
                    data[new] = data.pop(old)
        return data

    status: SetFilter[Status] | None = None
    care_type: SetFilter[CareType] | None = None
    facility: SetFilter[str] | None = None  # the resident's facility (formerly "wing")
    gender: SetFilter[Gender] | None = None
    primary_caregiver: SetFilter[str] | None = None
    room_number: SetFilter[str] | None = None
    age: AgeRange | None = None
    admission_date: DateRange | None = None
    created_date: DateRange | None = None
    discharge_date: DateRange | None = None
    # family relation (familymembers): relations / kinds / access flags of ANY member, and the number of members
    family_relation: SetFilter[str] | None = None
    family_type: SetFilter[FamilyType] | None = None
    family_flags: SetFilter[FamilyFlag] | None = None
    family_count: CountRange | None = None
    # specific family members, as familymembers record ids (resolved by the backend from family_N / person_N
    # placeholders; a model never writes an id or a name here)
    family_member: SetFilter[str] | None = None
    # staff (app/data/staff.py). Roles are canonical role names (resolved from question words by the backend);
    # designations are as recorded. Resident plan: ANY ACTIVE assigned staff member has it. Staff plan: the staff member.
    staff_role: SetFilter[str] | None = None
    staff_designation: SetFilter[str] | None = None
    staff_count: CountRange | None = None  # resident plans: number of active assigned staff
    # staff plans only
    staff_status: SetFilter[StaffStatus] | None = None  # not given → active staff only
    staff_facility: SetFilter[str] | None = None
    staff_type: SetFilter[StaffType] | None = None
    staff_name: SetFilter[str] | None = None  # staff names as recorded (resolved by the backend)
    staff_resident_count: CountRange | None = None  # assigned residents (matching the plan's resident conditions)

    @model_validator(mode="after")
    def _text_match_fields(self):
        for name in ("status", "care_type", "gender", "family_relation", "family_type", "family_flags", "family_member",
                     "staff_role", "staff_status", "staff_type"):
            sf = getattr(self, name)
            if sf is not None and sf.has_text_match:
                raise ValueError(f"starts_with/contains apply only to {', '.join(TEXT_MATCH_FIELDS)}, not {name}")
        return self

    @property
    def has_text_match(self) -> bool:
        return any(getattr(self, n) is not None and getattr(self, n).has_text_match for n in ALL_TEXT_MATCH_FIELDS)

    def is_empty(self) -> bool:
        return all(getattr(self, f) is None for f in type(self).model_fields)


class Aggregate(_Strict):
    op: Literal["avg", "min", "max"]
    field: Literal["age"] = "age"


class Sort(_Strict):
    field: SortField
    direction: Literal["asc", "desc"]


class Lookup(_Strict):
    residents: list[str] = Field(default_factory=list, max_length=5)
    staff: list[str] | None = Field(default=None, max_length=5)  # staff names (resolved by the backend)
    attributes: list[Attribute] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_kind(self):
        if bool(self.residents) == bool(self.staff):
            raise ValueError("lookup needs residents or staff (one of them, at least one)")
        return self


class Option(_Strict):
    label: str = Field(max_length=120)
    query: str = Field(max_length=300)


class QueryPlan(_Strict):
    @model_validator(mode="before")
    @classmethod
    def _legacy_names(cls, data):
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for key in ("fields", "group_by"):
            if isinstance(data.get(key), list):
                data[key] = [canonical_field(v) for v in data[key]]
        if isinstance(data.get("lookup"), dict) and isinstance(data["lookup"].get("attributes"), list):
            data["lookup"] = {**data["lookup"], "attributes": [canonical_field(v) for v in data["lookup"]["attributes"]]}
        return data

    action: Action
    # None = residents (the default; left out of serialised plans so resident plans are exactly what they were)
    entity: Entity | None = None
    filters: Filters = Field(default_factory=Filters)
    group_by: list[GroupField] | None = Field(default=None, min_length=1, max_length=2)
    aggregate: Aggregate | None = None
    sort: Sort | None = None
    limit: int | None = Field(default=None, ge=1, le=200)
    lookup: Lookup | None = None
    # Projection (SELECT), independent of the conditions (WHERE): the attributes to show for each resident. The
    # resident's name/identity is always included. Breakdowns can project residents within each group. `lookup.attributes`
    # is the older form for one resident; both are read through `output_fields`. No projection → default result shape.
    fields: list[Attribute] | None = Field(default=None, min_length=1, max_length=len(Attribute.__args__))  # type: ignore[attr-defined]
    message: str | None = Field(default=None, max_length=500)
    options: list[Option] | None = Field(default=None, max_length=6)

    @field_validator("entity")
    @classmethod
    def _default_entity(cls, v):
        return None if v == "resident" else v

    @field_validator("group_by")
    @classmethod
    def _unique_groups(cls, v):
        if v and len(set(v)) != len(v):
            raise ValueError("group_by fields must be unique")
        return v

    @model_validator(mode="after")
    def _action_requirements(self):
        if self.action == "breakdown" and not self.group_by:
            raise ValueError("breakdown requires group_by")
        if self.group_by and self.action not in ("breakdown", "aggregate"):
            raise ValueError("group_by applies to breakdown and aggregate only")
        if self.action == "aggregate" and not self.aggregate:
            raise ValueError("aggregate requires aggregate")
        if self.aggregate and self.action != "aggregate":
            raise ValueError("aggregate applies to aggregate only")
        if self.action == "lookup" and not self.lookup:
            raise ValueError("lookup requires residents")
        if self.lookup and self.action != "lookup":
            raise ValueError("lookup applies to lookup only")
        if (self.sort or self.limit is not None) and self.action != "list":
            raise ValueError("sort and limit apply to list only")
        if self.fields is not None:
            if self.action not in ("list", "breakdown", "lookup"):
                raise ValueError("fields (what to show) apply to list, breakdown and lookup only")
            if len(set(self.fields)) != len(self.fields):
                raise ValueError("fields must be unique")
        return self._entity_requirements()

    def _entity_requirements(self):
        """Which fields belong to which entity (see the staff_* comment at the top of this module)."""
        f, shown, groups = self.filters, self.output_fields, self.group_by or []
        sort = self.sort.field if self.sort else None
        if self.entity == "staff":
            wrong = [n for n in ("staff_count",) if getattr(f, n) is not None]
            wrong += [a for a in shown if a not in STAFF_ATTRIBUTES]
            wrong += [g for g in groups if g not in STAFF_GROUP_FIELDS]
            wrong += [sort] if sort and sort not in STAFF_SORT_FIELDS else []
            if wrong:
                raise ValueError(f"not available for staff: {', '.join(dict.fromkeys(wrong))}")
            if self.action == "aggregate":
                raise ValueError("aggregate is not available for staff")
            if self.lookup and self.lookup.residents:
                raise ValueError("a staff lookup names staff (lookup.staff), not residents")
            if self.action != "lookup" and set(shown) & set(STAFF_SINGLE_RECORD):
                raise ValueError("staff_email, staff_phone and staff_residents are shown for one looked-up staff member only")
        else:
            wrong = [n for n in STAFF_ONLY_FIELDS if getattr(f, n) is not None]
            wrong += [a for a in shown if a in STAFF_ATTRIBUTES]
            wrong += [g for g in groups if g in STAFF_GROUP_FIELDS]
            wrong += [sort] if sort in STAFF_SORT_FIELDS else []
            if wrong:
                raise ValueError(f"staff fields need entity staff: {', '.join(dict.fromkeys(wrong))}")
            if self.lookup and self.lookup.staff:
                raise ValueError("lookup.staff needs entity staff")
        return self

    @property
    def output_fields(self) -> list[str]:
        """The requested projection, normalised: explicit `fields` plus legacy `lookup.attributes`, in order."""
        legacy = list(self.lookup.attributes) if self.lookup else []
        return list(dict.fromkeys([*(self.fields or []), *legacy]))

    @property
    def is_data_query(self) -> bool:
        return self.action in ("count", "list", "breakdown", "aggregate", "lookup")
