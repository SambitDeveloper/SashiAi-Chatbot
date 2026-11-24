"""MongoDB-backed repository using the PyMongo async driver directly (alternative to DATA_SOURCE=mcp).

Runs exactly the same aggregations as the MCP repository: the resident view from app/data/resident_schema.yaml
followed by the plan's filters. Use a database user with read-only rights.
"""

from pymongo import AsyncMongoClient

from app.data.resident_view import ResidentView, ViewRepository, to_mongo_filter

__all__ = ["MongoResidentRepository", "to_mongo_filter"]


class MongoResidentRepository(ViewRepository):
    name = "mongodb"

    def __init__(self, uri: str, db: str, collection: str, view: ResidentView | None = None):
        super().__init__(view)
        # a cold Atlas connection took 8-10 s from the dev machine (SRV + TLS + auth, 2026-11-18): 5 s failed startup
        self._client: AsyncMongoClient = AsyncMongoClient(uri, maxPoolSize=20, serverSelectionTimeoutMS=30000)
        self._col = self._client[db][collection]
        self._db = self._client[db]
        self.resident_collection = collection

    async def init(self) -> None:
        await self._client.admin.command("ping")
        await self._aggregate(self.view.pipeline(None, {"$limit": 1}))  # the mapping must run on this database

    async def close(self) -> None:
        await self._client.close()

    async def _aggregate(self, pipeline: list[dict], collection: str | None = None) -> list[dict]:
        col = self._db[collection] if collection else self._col  # e.g. staffs for staff questions
        return await (await col.aggregate(pipeline)).to_list()
