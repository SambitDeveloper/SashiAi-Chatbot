"""MongoDB-backed repository (async PyMongo).

ASSUMPTION — verify against the real Care App collection before go-live: documents use the same field
names and formats as residents.json (camelCase fields, dates as 'YYYY-MM-DD' strings, snake_case enum
values). If the real schema differs, only `to_mongo_filter` and FIELD_MAP need to change.

Recommended indexes: {residentId: 1} unique, {status: 1, careType: 1}, {admissionDate: 1}, {primaryCaregiver: 1}.
"""

from typing import Any

from pymongo import AsyncMongoClient

from app.data.repository import FIELD_MAP, GroupRow, Resident, Vocabulary
from app.models.plan import Filters, GroupField, Sort

MAX_FIND = 500


def to_mongo_filter(filters: Filters) -> dict[str, Any]:
    """Translate plan filters to a Mongo query. Must mirror `repository.matches`."""
    q: dict[str, Any] = {}
    for plan_field in ("status", "care_type", "wing", "gender", "primary_caregiver", "room_number"):
        f = getattr(filters, plan_field)
        if f is None:
            continue
        cond: dict[str, Any] = {}
        if f.include:
            cond["$in"] = list(f.include)
        if f.exclude:
            cond["$nin"] = list(f.exclude)
        q[FIELD_MAP[plan_field]] = cond
    if filters.age:
        cond = {"$exists": True, "$ne": None}
        if filters.age.min is not None:
            cond["$gte"] = filters.age.min
        if filters.age.max is not None:
            cond["$lte"] = filters.age.max
        q["age"] = cond
    for plan_field in ("admission_date", "created_date", "discharge_date"):
        r = getattr(filters, plan_field)
        if r is None:
            continue
        cond = {"$exists": True, "$nin": [None, ""]}
        if r.start:
            cond["$gte"] = r.start.isoformat()
        if r.end:
            cond["$lte"] = r.end.isoformat()
        q[FIELD_MAP[plan_field]] = cond
    return q


class MongoResidentRepository:
    name = "mongodb"

    def __init__(self, uri: str, db: str, collection: str):
        self._client: AsyncMongoClient = AsyncMongoClient(uri, maxPoolSize=20, serverSelectionTimeoutMS=5000)
        self._col = self._client[db][collection]

    async def init(self) -> None:
        await self._client.admin.command("ping")

    async def close(self) -> None:
        await self._client.close()

    async def total(self) -> int:
        return await self._col.count_documents({})

    async def count(self, filters: Filters) -> int:
        return await self._col.count_documents(to_mongo_filter(filters))

    async def find(self, filters: Filters, sort: Sort | None = None, limit: int | None = None) -> list[Resident]:
        cursor = self._col.find(to_mongo_filter(filters), {"_id": 0})
        if sort:
            cursor = cursor.sort([(FIELD_MAP[sort.field], 1 if sort.direction == "asc" else -1), ("residentId", 1)])
        cursor = cursor.limit(min(limit or MAX_FIND, MAX_FIND))
        return [doc async for doc in cursor]

    async def find_by_ids(self, resident_ids: list[str]) -> list[Resident]:
        cursor = self._col.find({"residentId": {"$in": resident_ids}}, {"_id": 0}).limit(len(resident_ids))
        return [doc async for doc in cursor]

    async def group_count(self, filters: Filters, fields: list[GroupField]) -> list[GroupRow]:
        group_id = {f: f"${FIELD_MAP[f]}" for f in fields}
        pipeline = [
            {"$match": to_mongo_filter(filters)},
            {"$group": {"_id": group_id, "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        rows = await (await self._col.aggregate(pipeline)).to_list()
        return [GroupRow(key=tuple(str(r["_id"].get(f) or "Unknown") for f in fields), count=r["count"]) for r in rows]

    async def aggregate(self, filters: Filters, op: str, field_name: str) -> float | None:
        pipeline = [
            {"$match": to_mongo_filter(filters)},
            {"$group": {"_id": None, "value": {f"${op}": f"${FIELD_MAP[field_name]}"}}},
        ]
        rows = await (await self._col.aggregate(pipeline)).to_list()
        if not rows or rows[0].get("value") is None:
            return None
        value = rows[0]["value"]
        return round(value, 1) if op == "avg" else value

    async def vocabulary(self) -> Vocabulary:
        residents = [
            doc async for doc in self._col.find({}, {"_id": 0, "residentId": 1, "firstName": 1, "lastName": 1})
        ]
        return Vocabulary(
            residents=residents,
            wings=sorted(str(w) for w in await self._col.distinct("wing") if w),
            caregivers=sorted(str(c) for c in await self._col.distinct("primaryCaregiver") if c),
            rooms=sorted(str(r) for r in await self._col.distinct("roomNumber") if r),
        )
