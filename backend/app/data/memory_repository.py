"""Repository over an in-memory list. Used by the tests only (fixed data in tests/data/residents.json)."""

import json
from collections import Counter
from itertools import product
from pathlib import Path

from app.data.repository import (
    FIELD_MAP,
    GroupAggregateRow,
    GroupRow,
    Resident,
    Vocabulary,
    group_key,
    matches,
    numeric,
    project,
    typed_order,
)
from app.data.schema_config import list_fields
from app.models.plan import Filters, GroupField, Sort

# test data only: production always reads the Care App MongoDB (see app/main.py build_repository)
SEED_FILE = Path(__file__).resolve().parents[2] / "tests" / "data" / "residents.json"


class MemoryResidentRepository:
    name = "memory"

    def __init__(self, records: list[Resident] | None = None):
        self._records = records if records is not None else json.loads(SEED_FILE.read_text(encoding="utf-8"))

    async def init(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def _filtered(self, filters: Filters) -> list[Resident]:
        return [r for r in self._records if matches(r, filters)]

    async def total(self) -> int:
        return len(self._records)

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
        return [project(r, fields) for r in rows] if fields is not None else rows

    async def find_by_ids(self, resident_ids: list[str]) -> list[Resident]:
        wanted = {i.upper() for i in resident_ids}
        return [r for r in self._records if r["residentId"].upper() in wanted]

    @staticmethod
    def _group_keys(record: Resident, fields: list[GroupField]) -> list[tuple[str, ...]]:
        """The group key(s) of one record — several for a list field (one per value), like MongoDB's $unwind with
        preserveNullAndEmptyArrays; a missing or empty value is the single "not recorded" group (repository.MISSING_GROUP)."""
        per_field = []
        for f in fields:
            vf = FIELD_MAP[f]
            value = record.get(vf)
            if vf in list_fields() and isinstance(value, list) and value:
                per_field.append([group_key(v, vf) for v in value])
            else:
                per_field.append([group_key(value, vf)])
        return list(product(*per_field))

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
        )
