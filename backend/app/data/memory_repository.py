"""Repository over an in-memory list. Used for development, tests and demos (seed data)."""

import json
from collections import Counter
from pathlib import Path

from app.data.repository import FIELD_MAP, GroupRow, Resident, Vocabulary, matches, sort_key
from app.models.plan import Filters, GroupField, Sort

SEED_FILE = Path(__file__).with_name("residents.json")


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

    async def find(self, filters: Filters, sort: Sort | None = None, limit: int | None = None) -> list[Resident]:
        rows = self._filtered(filters)
        if sort:
            key = sort_key(sort.field)
            present = [r for r in rows if r.get(FIELD_MAP[sort.field]) not in (None, "")]
            missing = [r for r in rows if r.get(FIELD_MAP[sort.field]) in (None, "")]
            rows = sorted(present, key=key, reverse=sort.direction == "desc") + missing
        return rows[:limit] if limit else rows

    async def find_by_ids(self, resident_ids: list[str]) -> list[Resident]:
        wanted = {i.upper() for i in resident_ids}
        return [r for r in self._records if r["residentId"].upper() in wanted]

    async def group_count(self, filters: Filters, fields: list[GroupField]) -> list[GroupRow]:
        counter = Counter(tuple(str(r.get(FIELD_MAP[f]) or "Unknown") for f in fields) for r in self._filtered(filters))
        return [GroupRow(key=k, count=c) for k, c in counter.most_common()]

    async def aggregate(self, filters: Filters, op: str, field_name: str) -> float | None:
        values = [r[FIELD_MAP[field_name]] for r in self._filtered(filters) if r.get(FIELD_MAP[field_name]) is not None]
        if not values:
            return None
        if op == "min":
            return min(values)
        if op == "max":
            return max(values)
        return round(sum(values) / len(values), 1)

    async def vocabulary(self) -> Vocabulary:
        return Vocabulary(
            residents=[{k: r[k] for k in ("residentId", "firstName", "lastName")} for r in self._records],
            wings=sorted({r["wing"] for r in self._records}),
            caregivers=sorted({r["primaryCaregiver"] for r in self._records}),
            rooms=sorted({r["roomNumber"] for r in self._records}),
        )
