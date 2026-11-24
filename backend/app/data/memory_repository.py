"""Repository over an in-memory list. Used by the tests only (fixed data in tests/data/residents.json)."""

import json
from collections import Counter
from pathlib import Path

from app.data import scope
from app.data.relations import FAMILY_VIEW_FIELDS, family_relation
from app.data.repository import (
    FIELD_MAP,
    MISSING_GROUP,
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
from app.data.staff import STAFF_RESIDENT_FIELDS, resident_part, staff_config, staff_matches, staff_part
from app.models.plan import STAFF_FIELD_MAP, Filters, GroupField, Sort

# test data only: production always reads the Care App MongoDB (see app/main.py build_repository)
SEED_FILE = Path(__file__).resolve().parents[2] / "tests" / "data" / "residents.json"
FAMILY_FILE = SEED_FILE.with_name("familymembers.json")  # raw familymembers documents, keyed by residentId
STAFF_FILE = SEED_FILE.with_name("staffs.json")  # raw staffs documents


def _names(value) -> list[str]:
    return [v for v in (value if isinstance(value, list) else [value]) if isinstance(v, str) and v]


class MemoryResidentRepository:
    name = "memory"

    def __init__(self, records: list[Resident] | None = None, family: list[dict] | None = None, staff: list[dict] | None = None):
        self._records = records if records is not None else json.loads(SEED_FILE.read_text(encoding="utf-8"))
        if family is None and records is None and FAMILY_FILE.exists():
            family = json.loads(FAMILY_FILE.read_text(encoding="utf-8"))
        if staff is None and records is None and STAFF_FILE.exists():
            staff = json.loads(STAFF_FILE.read_text(encoding="utf-8"))
        self._staff_raw: list[dict] = list(staff or [])
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
        self._attach_staff()

    # ------------------------------------------------------------------ staff (app/data/staff.py rules)
    def _assigned(self, staff_raw: dict, resident: Resident) -> bool:
        """A resident's assigned staff: their assignedStaff keys (like MongoDB), or in the fixture the staff member
        whose name is the resident's caregiver, or who lists the resident in fixtureResidents."""
        cfg = staff_config()
        key = staff_raw.get(cfg.key)
        if isinstance(resident.get(cfg.resident_link), list):
            return key in resident[cfg.resident_link]
        return staff_raw.get("name") in _names(resident.get("primaryCaregiver")) or (
            resident.get("residentId") in (staff_raw.get("fixtureResidents") or [])
        )

    def _attach_staff(self) -> None:
        cfg = staff_config()
        if cfg is None:
            return
        out = []
        for r in self._records:
            team = [s for s in self._staff_raw if self._assigned(s, r)]
            link = r.get(cfg.physician_link)
            physician = next((s for s in self._staff_raw if (link and s.get(cfg.key) == link)
                              or r.get("residentId") in (s.get("fixturePrimaryPhysicianOf") or [])), None)  # fmt: skip
            out.append({**r, **cfg.derive(team, physician)})
        self._records = out

    def _staff_rows(self, contacts: bool = False) -> list[dict]:
        cfg = staff_config()
        return [cfg.entity_record(s, {}, contacts) for s in self._staff_raw if not cfg.is_hidden(s.get("designation"))]

    def _staff_filtered(self, filters: Filters, residents: bool = False, contacts: bool = False, rows=None) -> list[dict]:
        """Mirrors ViewRepository._staff_pipeline: own conditions, assigned residents matching the resident conditions
        (at least one when there are any), then the resident-count conditions."""
        about = resident_part(filters)
        conditional = not about.is_empty()
        own = staff_part(filters)
        by_id = {s.get(staff_config().key): s for s in self._staff_raw}
        out = []
        for row in rows if rows is not None else self._staff_rows(contacts):
            if not staff_matches(row, own.model_copy(update={"staff_resident_count": None})):
                continue
            raw = by_id.get(row["staffId"], {})
            assigned = [r for r in self._scoped() if self._assigned(raw, r) and (not conditional or matches(r, about))]
            row = {**row, "residentCount": len(assigned)}
            if conditional and not assigned:
                continue
            if not staff_matches(row, Filters(staff_resident_count=filters.staff_resident_count)):
                continue
            if residents:
                row["residents"] = [f"{r.get('firstName', '')} {r.get('lastName', '')}".strip() for r in assigned]
            out.append(row)
        return out

    async def staff_total(self) -> int:
        return len(self._staff_rows())

    async def staff_count(self, filters: Filters) -> int:
        return len(self._staff_filtered(filters))

    async def staff_find(self, filters: Filters, sort=None, limit: int | None = None, fields: list[str] | None = None) -> list[dict]:
        rows = sorted(self._staff_filtered(filters), key=lambda r: (r["name"], r["staffId"]))
        if sort:
            field = STAFF_FIELD_MAP[sort.field]
            rows.sort(key=lambda r: typed_order(r.get(field)), reverse=sort.direction == "desc")
        rows = rows[:limit] if limit else rows
        keep = ["staffId", "name", *(fields or [])]
        return [{k: r[k] for k in keep if k in r} for r in rows]

    async def staff_group_count(self, filters: Filters, fields: list[str]) -> list[GroupRow]:
        def key(r: dict) -> tuple[str, ...]:
            return tuple(str(r.get(STAFF_FIELD_MAP[f])) if r.get(STAFF_FIELD_MAP[f]) not in (None, "") else MISSING_GROUP
                         for f in fields)  # fmt: skip

        counter = Counter(key(r) for r in self._staff_filtered(filters))
        return [GroupRow(key=k, count=c) for k, c in counter.most_common()]

    async def staff_by_names(self, names: list[str], contacts: bool = False) -> list[dict]:
        rows = [r for r in self._staff_rows(contacts) if r["name"] in names]
        return sorted(self._staff_filtered(Filters(), residents=True, rows=rows), key=lambda r: (r["name"], r["staffId"]))[:20]

    async def staff_access(self, name: str, designation: str) -> list[list]:
        cfg = staff_config()
        return [s.get("accessPermissions") or [] for s in self._staff_raw
                if not cfg.is_hidden(s.get("designation")) and str(s.get("name") or "").strip() == name
                and cfg.member(s)["designation"] == designation][:2]  # fmt: skip

    async def staff_vocabulary(self) -> list[dict]:
        return [{k: r[k] for k in ("name", "designation", "role", "status")} for r in self._staff_rows() if r["name"]]

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
        shown_lists = [f for f in fields if f in (*FAMILY_VIEW_FIELDS, *STAFF_RESIDENT_FIELDS) and f in list_fields()]
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
            staff=await self.staff_vocabulary(),
        )
