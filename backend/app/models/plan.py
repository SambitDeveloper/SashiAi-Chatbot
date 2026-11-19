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
GroupField = Literal["status", "care_type", "facility", "gender", "primary_caregiver"]
SortField = Literal["admission_date", "created_date", "discharge_date", "age", "last_name"]
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
]
Action = Literal["count", "list", "breakdown", "aggregate", "lookup", "clarify", "unsupported", "help"]

# "wing" was the earlier name of `facility` (the data has no wing: the field is the resident's facility). It is still
# accepted wherever a plan is READ — cached plans, older clients, model output — and normalised to the one canonical
# name, so there is never a second way to express the condition.
LEGACY_FIELD_NAMES = {"wing": "facility"}


def canonical_field(name):
    return LEGACY_FIELD_NAMES.get(name, name) if isinstance(name, str) else name


STATUSES: tuple[str, ...] = Status.__args__  # type: ignore[attr-defined]
CARE_TYPES: tuple[str, ...] = CareType.__args__  # type: ignore[attr-defined]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Text matching ("room starts with A", "facility contains Garden") is allowed on these free-text plan fields ONLY.
# Fixed-vocabulary fields (status, care type, gender) are matched by exact value. Semantics: app/data/text_match.py.
TEXT_MATCH_FIELDS: tuple[str, ...] = ("room_number", "facility", "primary_caregiver")
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

    @model_validator(mode="after")
    def _text_match_fields(self):
        for name in ("status", "care_type", "gender"):
            sf = getattr(self, name)
            if sf is not None and sf.has_text_match:
                raise ValueError(f"starts_with/contains apply only to {', '.join(TEXT_MATCH_FIELDS)}, not {name}")
        return self

    @property
    def has_text_match(self) -> bool:
        return any(getattr(self, n) is not None and getattr(self, n).has_text_match for n in TEXT_MATCH_FIELDS)

    def is_empty(self) -> bool:
        return all(getattr(self, f) is None for f in type(self).model_fields)


class Aggregate(_Strict):
    op: Literal["avg", "min", "max"]
    field: Literal["age"] = "age"


class Sort(_Strict):
    field: SortField
    direction: Literal["asc", "desc"]


class Lookup(_Strict):
    residents: list[str] = Field(min_length=1, max_length=5)
    attributes: list[Attribute] = Field(default_factory=list)


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
        return self

    @property
    def output_fields(self) -> list[str]:
        """The requested projection, normalised: explicit `fields` plus legacy `lookup.attributes`, in order."""
        legacy = list(self.lookup.attributes) if self.lookup else []
        return list(dict.fromkeys([*(self.fields or []), *legacy]))

    @property
    def is_data_query(self) -> bool:
        return self.action in ("count", "list", "breakdown", "aggregate", "lookup")
