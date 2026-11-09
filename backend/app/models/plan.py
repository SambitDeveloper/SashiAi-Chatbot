"""QueryPlan — the single contract between understanding and execution.

Claude only ever produces a QueryPlan. The plan is resolved against real database values and then
validated by these models before anything touches the data store, so the model cannot reach data
the schema does not describe, and every number in an answer comes from the database.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Status = Literal["active", "inactive", "pending", "admitted", "discharged"]
CareType = Literal["assisted_living", "memory_care", "independent_living", "skilled_nursing", "respite_care"]
Gender = Literal["Female", "Male", "Other"]
GroupField = Literal["status", "care_type", "wing", "gender", "primary_caregiver"]
SortField = Literal["admission_date", "created_date", "discharge_date", "age", "last_name"]
Attribute = Literal[
    "primary_caregiver",
    "room_number",
    "wing",
    "admission_date",
    "created_date",
    "discharge_date",
    "emergency_contact",
    "status",
    "care_type",
    "age",
    "gender",
]
Action = Literal["count", "list", "breakdown", "aggregate", "lookup", "clarify", "unsupported", "help"]

STATUSES: tuple[str, ...] = Status.__args__  # type: ignore[attr-defined]
CARE_TYPES: tuple[str, ...] = CareType.__args__  # type: ignore[attr-defined]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SetFilter[T](_Strict):
    include: list[T] | None = Field(default=None, max_length=20)
    exclude: list[T] | None = Field(default=None, max_length=20)

    @model_validator(mode="after")
    def _not_empty(self):
        if not self.include and not self.exclude:
            raise ValueError("filter needs include or exclude values")
        return self


class AgeRange(_Strict):
    min: int | None = Field(default=None, ge=0, le=130)
    max: int | None = Field(default=None, ge=0, le=130)

    @model_validator(mode="after")
    def _order(self):
        if self.min is None and self.max is None:
            raise ValueError("age range needs min or max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("age min is greater than max")
        return self


class DateRange(_Strict):
    start: date | None = None
    end: date | None = None

    @model_validator(mode="after")
    def _order(self):
        if self.start is None and self.end is None:
            raise ValueError("date range needs start or end")
        if self.start and self.end and self.start > self.end:
            raise ValueError("date range start is after end")
        return self


class Filters(_Strict):
    status: SetFilter[Status] | None = None
    care_type: SetFilter[CareType] | None = None
    wing: SetFilter[str] | None = None
    gender: SetFilter[Gender] | None = None
    primary_caregiver: SetFilter[str] | None = None
    room_number: SetFilter[str] | None = None
    age: AgeRange | None = None
    admission_date: DateRange | None = None
    created_date: DateRange | None = None
    discharge_date: DateRange | None = None

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
    action: Action
    filters: Filters = Field(default_factory=Filters)
    group_by: list[GroupField] | None = Field(default=None, min_length=1, max_length=2)
    aggregate: Aggregate | None = None
    sort: Sort | None = None
    limit: int | None = Field(default=None, ge=1, le=200)
    lookup: Lookup | None = None
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
        if self.action == "aggregate" and not self.aggregate:
            raise ValueError("aggregate requires aggregate")
        if self.action == "lookup" and not self.lookup:
            raise ValueError("lookup requires residents")
        return self

    @property
    def is_data_query(self) -> bool:
        return self.action in ("count", "list", "breakdown", "aggregate", "lookup")
