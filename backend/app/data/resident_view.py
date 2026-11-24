"""The Care App's real `residents` collection, seen through app/data/resident_schema.yaml.

Every query is ONE MongoDB aggregation:  resident view (YAML)  →  the plan's filters  →  count / list / group / avg.
The view does the soft-delete filter, the staff and facility joins, the field mapping and the field whitelist inside
MongoDB, so documents leave the database already in the shape the engine expects and without any other field.

`ViewRepository` implements the whole repository interface on top of a single `_aggregate(pipeline)` call; the MCP
repository and the PyMongo repository only differ in how they run that call.
"""

from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

from app.data import scope, text_match
from app.data.relations import FAMILY_VIEW_FIELDS, family_relation, plan_uses_family
from app.data.repository import (
    FIELD_MAP,
    MISSING_GROUP,
    SET_FILTER_FIELDS,
    GroupAggregateRow,
    GroupRow,
    Resident,
    Vocabulary,
)
from app.data.schema_config import identity_fields, list_fields, missing_markers
from app.data.staff import (
    STAFF_RESIDENT_FIELDS,
    plan_uses_staff,
    resident_part,
    staff_config,
    staff_mongo_filter,
    staff_part,
)
from app.models.plan import STAFF_FIELD_MAP, Filters, GroupField, Sort

SCHEMA_FILE = Path(__file__).with_name("resident_schema.yaml")
MAX_ROWS = 5000


def _missing_clause(view_field: str, missing: bool) -> dict[str, Any]:
    """"No value recorded" (or "a value is recorded") for one view field, per resident_schema.yaml.

    Missing = null, an absent field, "", a declared placeholder (`missing_values`), or for a list field
    (`list_fields`) a list with no real value. Mirrors schema_config.is_missing exactly.
    """
    markers = missing_markers(view_field)
    if view_field in list_fields():
        has_value = {"$elemMatch": {"$nin": markers}}
        return {view_field: {"$not": has_value}} if missing else {view_field: has_value}
    return {view_field: {"$in" if missing else "$nin": markers}}


def to_mongo_filter(filters: Filters) -> dict[str, Any]:
    """Translate plan filters to a Mongo query on the view's fields. Must mirror `repository.matches`.

    On a list field (caregivers) `$in` matches when ANY element is included and `$nin` when NONE is excluded.
    Within one field, `include` OR `missing: true`; `exclude` and `missing: false` are AND-ed. Plans that do not use
    `missing` produce exactly the same query as before it existed.
    """
    q: dict[str, Any] = {}
    extra: list[dict[str, Any]] = []  # clauses that need $or / a second condition on the same field
    for plan_field in SET_FILTER_FIELDS:
        f = getattr(filters, plan_field)
        if f is None:
            continue
        field = FIELD_MAP[plan_field]
        if f.has_text_match:  # AND-ed clauses; the semantics live in text_match (shared with repository.matches)
            extra.extend(text_match.mongo_clauses(field, f.starts_with, f.contains))
        if f.missing is None:
            cond: dict[str, Any] = {}
            if f.include:
                cond["$in"] = list(f.include)
            if f.exclude:
                cond["$nin"] = list(f.exclude)
            if cond:  # a text-only filter has no value condition ({field: {}} would mean "equals {}")
                q[field] = cond
            continue
        if f.missing:
            alternatives = [_missing_clause(field, True)]
            if f.include:
                alternatives.insert(0, {field: {"$in": list(f.include)}})
            extra.append(alternatives[0] if len(alternatives) == 1 else {"$or": alternatives})
        else:
            extra.append(_missing_clause(field, False))
            if f.exclude:
                extra.append({field: {"$nin": list(f.exclude)}})

    ranges: list[tuple[str, Any, Any, Any]] = []
    if filters.age:
        ranges.append(("age", filters.age, filters.age.min, filters.age.max))
    if filters.family_count:
        ranges.append(("familyCount", filters.family_count, filters.family_count.min, filters.family_count.max))
    if filters.staff_count:
        ranges.append(("staffCount", filters.staff_count, filters.staff_count.min, filters.staff_count.max))
    for plan_field in ("admission_date", "created_date", "discharge_date"):
        r = getattr(filters, plan_field)
        if r is not None:
            lo = r.start.isoformat() if r.start else None
            hi = r.end.isoformat() if r.end else None
            ranges.append((FIELD_MAP[plan_field], r, lo, hi))
    for field, r, lo, hi in ranges:
        if r.has_range:
            numeric = field in ("age", "familyCount", "staffCount")
            cond = {"$exists": True, "$ne": None} if numeric else {"$exists": True, "$nin": [None, ""]}
            if lo is not None:
                cond["$gte"] = lo
            if hi is not None:
                cond["$lte"] = hi
            if r.missing:  # in range OR not recorded
                extra.append({"$or": [{field: cond}, _missing_clause(field, True)]})
            else:
                q[field] = cond
        else:
            extra.append(_missing_clause(field, bool(r.missing)))

    if not extra:
        return q
    return {"$and": [*({k: v} for k, v in q.items()), *extra]}


class ResidentView:
    """Builds the aggregation stages described by resident_schema.yaml."""

    def __init__(self, schema_file: Path | str = SCHEMA_FILE):
        self.schema_file = Path(schema_file)
        self.config: dict = yaml.safe_load(self.schema_file.read_text(encoding="utf-8"))
        self.fields: dict[str, Any] = self.config["fields"]
        self.list_fields: list[str] = list(self.config.get("list_fields") or [])
        self.required_source_fields: list[str] = list(self.config.get("required_source_fields") or [])

    @cached_property
    def stages(self) -> list[dict]:
        """The view without related collections (every query that does not use them)."""
        return self._stages(False)

    @cached_property
    def stages_with_family(self) -> list[dict]:
        """The view plus the `family` relation's join and derived fields (app/data/relations.py), no contact details."""
        return self._stages(True, contacts=False)

    @cached_property
    def stages_with_family_contacts(self) -> list[dict]:
        """As stages_with_family, plus each family member's phone and email (one resident's answer that shows them)."""
        return self._stages(True, contacts=True)

    def stages_for(self, family: bool, contacts: bool = False, staff: bool = False) -> list[dict]:
        """The view stages for one query, with the request's facility scope (app/data/scope.py) right after the base
        match, so every count, list and lookup of the request is limited alike. `staff`: also join the resident's
        assigned staff and primary physician (app/data/staff.py)."""
        if staff and staff_config() is not None:
            key = (family and family_relation() is not None, contacts, True)
            if key not in self._staff_variants:
                self._staff_variants[key] = self._stages(key[0], contacts, staff=True)
            stages = self._staff_variants[key]
        elif family and family_relation() is not None:
            stages = self.stages_with_family_contacts if contacts else self.stages_with_family
        else:
            stages = self.stages
        scoped = scope.mongo_match()
        return [stages[0], scoped, *stages[1:]] if scoped else stages

    @cached_property
    def _staff_variants(self) -> dict[tuple, list[dict]]:
        return {}

    def _stages(self, family: bool, contacts: bool = False, staff: bool = False) -> list[dict]:
        rel = family_relation() if family else None
        team = staff_config() if staff else None
        stages: list[dict] = [{"$match": dict(self.config.get("match") or {})}]
        for join in self.config.get("joins") or []:
            stages.append(
                {
                    "$lookup": {
                        "from": join["from"],
                        "localField": join["localField"],
                        "foreignField": join["foreignField"],
                        "pipeline": [{"$project": {"_id": 0, **join["fields"]}}],
                        "as": join["as"],
                    }
                }
            )
        if rel is not None:
            stages.append(rel.lookup_stage)
        if team is not None:
            stages.extend(team.team_lookup_stages)
        # $project both computes the app fields and drops every other field (whitelist)
        stages.append({"$project": {"_id": 0, **self.fields, **(rel.fields(contacts) if rel is not None else {}),
                                    **(team.resident_fields if team is not None else {})}})  # fmt: skip
        return stages

    def pipeline(self, filters: Filters | None, *tail: dict, family: bool = False, staff: bool = False) -> list[dict]:
        """`family` / `staff`: also join that relation (added automatically when the filters use it)."""
        match = to_mongo_filter(filters) if filters is not None else {}
        family = family or plan_uses_family(filters)
        staff = staff or plan_uses_staff(filters)
        return [*self.stages_for(family, staff=staff), *([{"$match": match}] if match else []), *tail]

    def base_stages(self) -> list[dict]:
        """The resident base match plus the request's facility scope (no joins, no mapping)."""
        scoped = scope.mongo_match()
        return [self.base_match, *([scoped] if scoped else [])]

    @property
    def base_match(self) -> dict:
        """Which documents are residents at all (e.g. not soft-deleted): the first stage of every query."""
        return {"$match": dict(self.config.get("match") or {})}

    def id_pipeline(self, resident_ids: list[str], *tail: dict, family: bool = True, contacts: bool = True) -> list[dict]:
        """Only the requested residents, narrowed BEFORE the joins and field mapping.

        Matching after them (as `pipeline` does for filters on computed fields) joins and maps every resident just to
        keep one. The id is computed with the schema's own `residentId` expression, which reads only the resident
        document itself, so it can run before the joins. Ids are compared without case.
        """
        wanted = sorted({i.upper() for i in resident_ids})
        id_expr = self.fields["residentId"]
        early = {"$match": {"$expr": {"$in": [{"$toUpper": id_expr}, wanted]}}}
        stages = self.stages_for(family, contacts, staff=True)  # one resident's answer may show family / care team
        return [stages[0], early, *stages[1:], *tail]

    def display_stage(self, only: list[str] | None = None) -> dict | None:
        """List fields are joined into one text for display ("Nurse A, Aide B") after filtering. With a projection
        (`only`), just the projected list fields (a $set would otherwise add fields the projection left out). The
        family lists are shown only when projected (the default result shape never carries family details)."""
        hidden = (*FAMILY_VIEW_FIELDS, *STAFF_RESIDENT_FIELDS)  # relation details are shown only when projected
        fields = [f for f in self.list_fields if (f in only if only is not None else f not in hidden)]
        if not fields:
            return None
        return {
            "$set": {
                f: {
                    "$reduce": {
                        "input": {"$ifNull": [f"${f}", []]},
                        "initialValue": "",
                        "in": {
                            "$cond": [
                                {"$eq": ["$$value", ""]},
                                "$$this",
                                {"$concat": ["$$value", ", ", "$$this"]},
                            ]
                        },
                    }
                }
                for f in fields
            }
        }


class ViewRepository:
    """Repository interface implemented with aggregations over the resident view."""

    name = "view"

    def __init__(self, view: ResidentView | None = None, max_rows: int = MAX_ROWS):
        self.view = view or ResidentView()
        self._max_rows = max_rows

    async def _aggregate(self, pipeline: list[dict], collection: str | None = None) -> list[dict]:  # pragma: no cover
        """Run one aggregation on the residents collection, or on `collection` (e.g. staffs). Implemented by subclasses."""
        raise NotImplementedError

    resident_collection = "residents"  # set by subclasses: the collection the resident view reads

    def _out(self, *stages: dict | None, only: list[str] | None = None) -> list[dict]:
        return [s for s in (*stages, self.view.display_stage(only)) if s]

    async def total(self) -> int:
        # The census needs no joins or computed fields: count the resident documents directly.
        rows = await self._aggregate([*self.view.base_stages(), {"$count": "n"}])
        return int(rows[0]["n"]) if rows else 0

    async def count(self, filters: Filters) -> int:
        rows = await self._aggregate(self.view.pipeline(filters, {"$count": "n"}))
        return int(rows[0]["n"]) if rows else 0

    async def find(
        self, filters: Filters, sort: Sort | None = None, limit: int | None = None, fields: list[str] | None = None
    ) -> list[Resident]:
        """`fields`: a projection (view fields) — only the identity fields and these leave MongoDB."""
        tail: list[dict] = []
        if sort:
            field = FIELD_MAP[sort.field]
            # residents without a value come last in either direction; ties by residentId (same as the memory repo)
            tail += [
                {"$set": {"_missing": {"$cond": [{"$in": [{"$ifNull": [f"${field}", None]}, [None, ""]]}, 1, 0]}}},
                {"$sort": {"_missing": 1, field: 1 if sort.direction == "asc" else -1, "residentId": 1}},
                {"$unset": "_missing"},
            ]
        tail.append({"$limit": min(limit or self._max_rows, self._max_rows)})
        family = plan_uses_family(filters, fields, sort.field if sort else None)
        staff = plan_uses_staff(filters, fields)
        if fields is not None:  # projection after filtering/sorting/limiting; mirrors repository.project
            tail.append({"$project": {k: 1 for k in dict.fromkeys([*identity_fields(), *fields])}})
            return await self._aggregate(self.view.pipeline(filters, *tail, *self._out(only=fields), family=family, staff=staff))
        return await self._aggregate(self.view.pipeline(filters, *tail, *self._out(), family=family, staff=staff))

    async def find_by_ids(self, resident_ids: list[str], contacts: bool = True) -> list[Resident]:
        """`contacts`: also read family members' phone and email (only when the answer may show them)."""
        if not resident_ids:
            return []
        # ids are compared without case: the name index keeps them upper-case, MongoDB ObjectIds are lower-case hex.
        # The id match runs before the joins, so only the requested residents are joined and mapped.
        return await self._aggregate(
            self.view.id_pipeline(resident_ids, {"$limit": len(resident_ids)}, *self._out(), contacts=contacts)
        )

    def _grouping(self, fields: list[GroupField]) -> tuple[list[dict], dict]:
        """$unwind stages for list fields + the group _id, with the missing-value rule of repository.MISSING_GROUP:
        absent / null / "" / an empty list / a declared placeholder → one "not recorded" key. Must mirror
        MemoryResidentRepository._group_keys."""
        unwind = [
            {"$unwind": {"path": f"${FIELD_MAP[f]}", "preserveNullAndEmptyArrays": True}}
            for f in fields
            if FIELD_MAP[f] in self.view.list_fields
        ]
        key: dict[str, Any] = {}
        for f in fields:
            vf = FIELD_MAP[f]
            key[f] = {
                "$cond": [
                    {"$in": [{"$ifNull": [f"${vf}", None]}, missing_markers(vf)]},
                    MISSING_GROUP,
                    # a malformed value (an object where a scalar belongs) is "not recorded", never a failed query
                    {"$convert": {"input": f"${vf}", "to": "string", "onError": MISSING_GROUP, "onNull": MISSING_GROUP}},
                ]
            }
        return unwind, key

    async def group_count(self, filters: Filters, fields: list[GroupField]) -> list[GroupRow]:
        unwind, key = self._grouping(fields)
        group = {"$group": {"_id": key, "count": {"$sum": 1}}}
        rows = await self._aggregate(self.view.pipeline(filters, *unwind, group, {"$sort": {"count": -1}}))
        return [GroupRow(key=tuple(str((r.get("_id") or {}).get(f)) for f in fields), count=int(r["count"])) for r in rows]

    async def group_aggregate(
        self, filters: Filters, fields: list[GroupField], op: str, field_name: str
    ) -> list[GroupAggregateRow]:
        """Every group's count and aggregate in ONE aggregation. Only numeric values are aggregated (a missing or
        malformed value is skipped, exactly like repository.numeric in memory); `n` says how many were used."""
        unwind, key = self._grouping(fields)
        vf = FIELD_MAP[field_name]
        number = {"$cond": [{"$isNumber": f"${vf}"}, f"${vf}", None]}
        group = {
            "$group": {
                "_id": key,
                "count": {"$sum": 1},
                "value": {f"${op}": number},
                "n": {"$sum": {"$cond": [{"$isNumber": f"${vf}"}, 1, 0]}},
            }
        }
        rows = await self._aggregate(self.view.pipeline(filters, *unwind, group, {"$sort": {"count": -1}}))
        out = []
        for r in rows:
            value = r.get("value")
            out.append(
                GroupAggregateRow(
                    key=tuple(str((r.get("_id") or {}).get(f)) for f in fields),
                    count=int(r["count"]),
                    value=None if value is None else (round(value, 1) if op == "avg" else value),
                    n_values=int(r.get("n") or 0),
                )
            )
        return out

    async def aggregate(self, filters: Filters, op: str, field_name: str) -> float | None:
        vf = FIELD_MAP[field_name]
        number = {"$cond": [{"$isNumber": f"${vf}"}, f"${vf}", None]}  # same rule as repository.numeric
        group = {"$group": {"_id": None, "value": {f"${op}": number}}}
        rows = await self._aggregate(self.view.pipeline(filters, group))
        if not rows or rows[0].get("value") is None:
            return None
        value = rows[0]["value"]
        return round(value, 1) if op == "avg" else value

    async def vocabulary(self) -> Vocabulary:
        """Names, facilities, staff names, rooms, statuses and care types, in one aggregation (kept on this server)."""

        def distinct(field: str) -> list[dict]:
            unwind = [{"$unwind": f"${field}"}] if field in self.view.list_fields else []
            return [*unwind, {"$group": {"_id": f"${field}"}}]

        facet = {
            "$facet": {
                "residents": [{"$project": {"residentId": 1, "firstName": 1, "lastName": 1}}],
                "wings": distinct("wing"),
                "caregivers": distinct("primaryCaregiver"),
                "rooms": distinct("roomNumber"),
                "statuses": distinct("status"),
                "care_types": distinct("careType"),
                "family_relations": distinct("familyRelations"),
            }
        }
        rows = await self._aggregate(self.view.pipeline(None, facet, family=True))
        data = rows[0] if rows else {}

        def values(key: str) -> list[str]:
            return sorted({str(r["_id"]).strip() for r in data.get(key, []) if r.get("_id")} - {""})

        return Vocabulary(
            residents=[
                {"residentId": r["residentId"], "firstName": r.get("firstName", ""), "lastName": r.get("lastName", "")}
                for r in data.get("residents", [])
            ],
            wings=values("wings"),
            caregivers=values("caregivers"),
            rooms=values("rooms"),
            statuses=values("statuses"),
            care_types=values("care_types"),
            family_relations=values("family_relations"),
            family_members=await self.family_vocabulary(),
            staff=await self.staff_vocabulary(),
        )

    async def family_vocabulary(self) -> list[dict]:
        """Every live resident's family members for the entity index: names and relations only, never contacts.
        Not facility-scoped here: candidates are filtered by the request's scope when a name is resolved."""
        rel = family_relation()
        if rel is None:
            return []
        rows = await self._aggregate([self.view.base_match, *rel.vocabulary_pipeline_tail])
        return [{**m, "residentId": r["residentId"], "facilityId": r.get("facilityId")}
                for r in rows for m in r.get("members") or []]  # fmt: skip

    # ------------------------------------------------------------------ staff entity (app/data/staff.py)
    def _staff_pipeline(self, filters: Filters, *tail: dict, residents: bool = False, contacts: bool = False) -> list[dict]:
        """Staff view → the staff member's own conditions → (when needed) the assigned residents matching the plan's
        resident conditions → residentCount conditions. A staff member with resident conditions must have at least one
        such resident. `residents`: keep the residents' names (one staff member's answer)."""
        cfg = staff_config()
        own = staff_mongo_filter(staff_part(filters).model_copy(update={"staff_resident_count": None}))
        about = resident_part(filters)
        conditional = not about.is_empty()
        counted = conditional or residents or filters.staff_resident_count is not None or _needs_count(tail)
        stages = [*cfg.entity_stages(contacts), *([{"$match": own}] if own else [])]
        if counted:
            names = {"residentId": 1, "firstName": 1, "lastName": 1}
            if conditional:
                inner = self.view.pipeline(about, {"$project": {"_id": 0, **names}})
            else:
                f = self.view.fields
                inner = [*self.view.base_stages(),
                         {"$project": {"_id": 0, "residentId": f["residentId"], "firstName": f["firstName"], "lastName": f["lastName"]}}]  # fmt: skip
            stages += [
                {"$lookup": {"from": self.resident_collection, "localField": "staffId", "foreignField": cfg.resident_link,
                             "pipeline": inner, "as": "_res"}},
                {"$set": {"residentCount": {"$size": "$_res"}}},
            ]  # fmt: skip
            if conditional:
                stages.append({"$match": {"residentCount": {"$gte": 1}}})
            count_only = staff_mongo_filter(Filters(staff_resident_count=filters.staff_resident_count)) if filters.staff_resident_count else {}
            if count_only:
                stages.append({"$match": count_only})
            if residents:
                stages.append({"$set": {"residents": {"$map": {"input": "$_res", "as": "r", "in": {
                    "$trim": {"input": {"$concat": ["$$r.firstName", " ", "$$r.lastName"]}}}}}}})  # fmt: skip
            stages.append({"$unset": "_res"})
        return [*stages, *tail]

    async def staff_total(self) -> int:
        """Every staff member who is a person (any status): the staff census."""
        rows = await self._aggregate([staff_config().visible_match, {"$count": "n"}], staff_config().collection)
        return int(rows[0]["n"]) if rows else 0

    async def staff_count(self, filters: Filters) -> int:
        rows = await self._aggregate(self._staff_pipeline(filters, {"$count": "n"}), staff_config().collection)
        return int(rows[0]["n"]) if rows else 0

    async def staff_find(self, filters: Filters, sort=None, limit: int | None = None, fields: list[str] | None = None) -> list[dict]:
        tail: list[dict] = []
        if sort:
            field = STAFF_FIELD_MAP[sort.field]
            tail.append({"$sort": {field: 1 if sort.direction == "asc" else -1, "name": 1, "staffId": 1}})
        else:
            tail.append({"$sort": {"name": 1, "staffId": 1}})
        tail.append({"$limit": min(limit or self._max_rows, self._max_rows)})
        keep = ["staffId", "name", *(fields or [])]
        tail.append({"$project": {"_id": 0, **dict.fromkeys(keep, 1)}})
        return await self._aggregate(self._staff_pipeline(filters, *tail), staff_config().collection)

    async def staff_group_count(self, filters: Filters, fields: list[str]) -> list[GroupRow]:
        key = {
            f: {"$cond": [{"$in": [{"$ifNull": [f"${STAFF_FIELD_MAP[f]}", None]}, [None, ""]]}, MISSING_GROUP,
                          {"$convert": {"input": f"${STAFF_FIELD_MAP[f]}", "to": "string", "onError": MISSING_GROUP, "onNull": MISSING_GROUP}}]}
            for f in fields
        }  # fmt: skip
        group = {"$group": {"_id": key, "count": {"$sum": 1}}}
        rows = await self._aggregate(self._staff_pipeline(filters, group, {"$sort": {"count": -1}}), staff_config().collection)
        return [GroupRow(key=tuple(str((r.get("_id") or {}).get(f)) for f in fields), count=int(r["count"])) for r in rows]

    async def staff_by_names(self, names: list[str], contacts: bool = False) -> list[dict]:
        """The named staff members (any status), with their assigned residents' names; contact details only when
        `contacts` (exactly one staff member's answer)."""
        if not names:
            return []
        tail = [{"$sort": {"name": 1, "staffId": 1}}, {"$limit": 20}]
        pipeline = self._staff_pipeline(Filters(), *tail, residents=True, contacts=contacts)
        cut = len(staff_config().entity_stages(contacts))  # match the names right after the staff view
        pipeline.insert(cut, {"$match": {"name": {"$in": list(names)}}})
        return await self._aggregate(pipeline, staff_config().collection)

    async def staff_access(self, name: str, designation: str) -> list[list]:
        """The app permissions of the staff record(s) with exactly this name and designation (one per record)."""
        cfg = staff_config()
        if cfg is None:
            return []
        return [r.get("access") or [] for r in await self._aggregate(cfg.access_stages(name, designation), cfg.collection)]

    async def staff_vocabulary(self) -> list[dict]:
        """Every staff member who is a person, any status: names, designation, role and status (no contact details)."""
        cfg = staff_config()
        if cfg is None:
            return []
        rows = await self._aggregate([*cfg.entity_stages(False), {"$project": {"_id": 0, "name": 1, "designation": 1,
                                                                               "role": 1, "status": 1}}], cfg.collection)  # fmt: skip
        return [r for r in rows if r.get("name")]


def _needs_count(tail: tuple) -> bool:
    """Does the rest of a staff pipeline read residentCount (a sort, projection or grouping on it)?"""
    return "residentCount" in repr(tail)
