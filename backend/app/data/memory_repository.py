"""Repository over an in-memory list. Used by the tests only (fixed data in tests/data/residents.json)."""

import json
from collections import Counter
from pathlib import Path

from app.data import scope
from app.data.relations import FAMILY_VIEW_FIELDS, family_relation
from app.data.repository import (
    FIELD_MAP,
    GroupAggregateRow,
    GroupRow,
    Resident,
    Vocabulary,
    group_keys,
    matches,
    numeric,
    project,
    typed_order,
)
from app.data.schema_config import list_fields
from app.models.plan import Filters, GroupField, Sort

# test data only: production always reads the Care App MongoDB (see app/main.py build_repository)
SEED_FILE = Path(__file__).resolve().parents[2] / "tests" / "data" / "residents.json"
FAMILY_FILE = SEED_FILE.with_name("familymembers.json")  # raw familymembers documents, keyed by residentId


class MemoryResidentRepository:
    name = "memory"

    def __init__(self, records: list[Resident] | None = None, family: list[dict] | None = None):
        self._records = records if records is not None else json.loads(SEED_FILE.read_text(encoding="utf-8"))
        if family is None and records is None and FAMILY_FILE.exists():
            family = json.loads(FAMILY_FILE.read_text(encoding="utf-8"))
        # the family relation's derived fields, by the same rules as the MongoDB view (app/data/relations.py); a
        # family record whose resident does not exist (an orphan) is never attached to anyone
        rel = family_relation()
        if rel is not None:
            by_resident: dict[str, list[dict]] = {}
            for member in family or []:
                by_resident.setdefault(str(member.get("residentId")), []).append(member)
            self._family_rows = [
                {**row, "residentId": r["residentId"], "facilityId": r.get("facilityId")}
                for r in self._records for row in rel.vocabulary_rows(by_resident.get(r["residentId"], []))
            ]  # fmt: skip
            self._records = [{**r, **rel.derive(by_resident.get(r["residentId"], []))} for r in self._records]
        else:
            self._family_rows = []

    async def init(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def _scoped(self) -> list[Resident]:
        """The residents of the request's facility scope (app/data/scope.py), like the MongoDB view's scope stage."""
        return [r for r in self._records if scope.in_scope(r.get("facilityId"))]

    def _filtered(self, filters: Filters) -> list[Resident]:
        return [r for r in self._scoped() if matches(r, filters)]

    async def total(self) -> int:
        return len(self._scoped())

    async def count(self, filters: Filters) -> int:
        return len(self._filtered(filters))

    async def find(
        self, filters: Filters, sort: Sort | None = None, limit: int | None = None, fields: list[str] | None = None
    ) -> list[Resident]:
        rows = self._filtered(filters)
        if sort:
            field_name = FIELD_MAP[sort.field]
            present = [r for r in rows if r.get(field_name) not in (None, "")]
            missing = [r for r in rows if r.get(field_name) in (None, "")]
            # ties are always broken by residentId ascending, the same as the MongoDB sort {field, residentId: 1}
            present.sort(key=lambda r: r.get("residentId", ""))
            present.sort(key=lambda r: typed_order(r[field_name]), reverse=sort.direction == "desc")
            rows = present + missing
        rows = rows[:limit] if limit else rows
        if fields is None:
            return rows
        # a projected list field is shown as one text, like ResidentView.display_stage ("A (Son), B (Daughter)")
        shown_lists = [f for f in fields if f in FAMILY_VIEW_FIELDS and f in list_fields()]
        out = [project(r, fields) for r in rows]
        for r in out:
            for f in shown_lists:
                if isinstance(r.get(f), list):
                    r[f] = ", ".join(str(v) for v in r[f])
        return out

    async def find_by_ids(self, resident_ids: list[str], contacts: bool = True) -> list[Resident]:
        wanted = {i.upper() for i in resident_ids}
        found = [r for r in self._scoped() if r["residentId"].upper() in wanted]
        if contacts:
            return found
        # like the MongoDB view without contact details: family members' phone and email are not read at all
        strip = ("phone", "email")
        return [{**r, "familyContacts": [{k: v for k, v in c.items() if k not in strip} for c in r.get("familyContacts") or []]}
                if "familyContacts" in r else r for r in found]  # fmt: skip

    @staticmethod
    def _group_keys(record: Resident, fields: list[GroupField]) -> list[tuple[str, ...]]:
        """The group key(s) of one record — several for a list field (one per value), like MongoDB's $unwind with
        preserveNullAndEmptyArrays; a missing or empty value is the single "not recorded" group (repository.MISSING_GROUP)."""
        return group_keys(record, fields)

    async def group_count(self, filters: Filters, fields: list[GroupField]) -> list[GroupRow]:
        counter = Counter(k for r in self._filtered(filters) for k in self._group_keys(r, fields))
        return [GroupRow(key=k, count=c) for k, c in counter.most_common()]

    @staticmethod
    def _apply(op: str, values: list) -> float | None:
        if not values:
            return None
        if op == "min":
            return min(values)
        if op == "max":
            return max(values)
        return round(sum(values) / len(values), 1)

    async def aggregate(self, filters: Filters, op: str, field_name: str) -> float | None:
        vf = FIELD_MAP[field_name]
        return self._apply(op, [v for r in self._filtered(filters) if (v := numeric(r.get(vf))) is not None])

    async def group_aggregate(
        self, filters: Filters, fields: list[GroupField], op: str, field_name: str
    ) -> list[GroupAggregateRow]:
        vf = FIELD_MAP[field_name]
        counts: Counter = Counter()
        values: dict[tuple[str, ...], list] = {}
        for r in self._filtered(filters):
            v = numeric(r.get(vf))
            for k in self._group_keys(r, fields):
                counts[k] += 1
                if v is not None:
                    values.setdefault(k, []).append(v)
        return [
            GroupAggregateRow(key=k, count=c, value=self._apply(op, values.get(k, [])), n_values=len(values.get(k, [])))
            for k, c in counts.most_common()
        ]

    async def vocabulary(self) -> Vocabulary:
        return Vocabulary(
            residents=[{k: r[k] for k in ("residentId", "firstName", "lastName")} for r in self._records],
            wings=sorted({r["wing"] for r in self._records}),
            caregivers=sorted({r["primaryCaregiver"] for r in self._records}),
            rooms=sorted({r["roomNumber"] for r in self._records}),
            statuses=sorted({r["status"] for r in self._records if r.get("status")}),
            care_types=sorted({r["careType"] for r in self._records if r.get("careType")}),
            family_relations=sorted({v for r in self._records for v in r.get("familyRelations") or []}),
            family_members=[dict(row) for row in self._family_rows],
        )
