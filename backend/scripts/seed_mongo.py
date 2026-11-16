"""Write the test residents into a LOCAL MongoDB in the Care App's real document shape (testing only).

Usage:  uv run python -m scripts.seed_mongo [mongodb://localhost:27017] [care_app] [residents]

The fixed test data (tests/data/residents.json) is stored the way the Care App stores residents — capitalised status,
"F"/"M" genders, unitNo, facilityId + `facilities`, assignedStaff ids + `staffs`, birthDate instead of age, Date
types, soft-deleted documents — so the opt-in MCP test proves that app/data/resident_schema.yaml maps the real
shape back to exactly the answers the in-memory test data gives. Never point this at a real database: it replaces
the collections' contents (it refuses non-local hosts).
"""

import json
import sys
import uuid
from datetime import UTC, date, datetime
from pathlib import Path

from pymongo import MongoClient

SEED = Path(__file__).resolve().parents[1] / "tests" / "data" / "residents.json"


def to_care_app_shape(residents: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    wings = sorted({r["wing"] for r in residents})
    facility_ids = {w: f"F{100 + i}" for i, w in enumerate(wings)}
    staff_ids = {c: str(uuid.uuid5(uuid.NAMESPACE_URL, c)) for c in sorted({r["primaryCaregiver"] for r in residents})}
    today = date.today()

    docs = []
    for i, r in enumerate(residents):
        doc = {
            "_id": r["residentId"],
            "firstName": r["firstName"],
            "lastName": r["lastName"],
            "status": r["status"].capitalize(),
            "careType": r["careType"],
            "gender": r["gender"][0] if i % 2 else r["gender"],  # half stored as "F" / "M"
            "unitNo": r["roomNumber"],
            "facilityId": facility_ids[r["wing"]],
            "assignedStaff": [staff_ids[r["primaryCaregiver"]]],
            "admissionDate": r.get("admissionDate") or "",
            "createdAt": datetime.fromisoformat(r["createdDate"]).replace(tzinfo=UTC),
            "emergencyContact": (r.get("emergencyContact") or {}).get("name", ""),
            "deletedAt": None,
        }
        if r.get("age") is not None:
            if i % 3:  # most residents only have a birth date (1 January → age is exact on any day of the year)
                doc["birthDate"] = date(today.year - r["age"], 1, 1).isoformat()
            else:
                doc["age"] = r["age"]
        if r.get("dischargeDate"):
            doc["dischargeDate"] = datetime.fromisoformat(r["dischargeDate"]).replace(tzinfo=UTC)
        docs.append(doc)

    # soft-deleted residents must never appear in any answer
    for n in (1, 2):
        docs.append({**docs[0], "_id": f"DELETED-{n}", "deletedAt": datetime.now(UTC)})

    staffs = [{"cName": sid, "name": name, "designation": "Caregiver"} for name, sid in staff_ids.items()]
    facilities = [{"facilityId": fid, "facilityName": wing} for wing, fid in facility_ids.items()]
    return docs, staffs, facilities


def main() -> None:
    uri = sys.argv[1] if len(sys.argv) > 1 else "mongodb://localhost:27017"
    db_name = sys.argv[2] if len(sys.argv) > 2 else "care_app"
    collection = sys.argv[3] if len(sys.argv) > 3 else "residents"
    if not uri.startswith(("mongodb://localhost", "mongodb://127.0.0.1")):
        sys.exit("refusing to seed a non-local MongoDB")

    docs, staffs, facilities = to_care_app_shape(json.loads(SEED.read_text(encoding="utf-8")))
    db = MongoClient(uri, serverSelectionTimeoutMS=5000)[db_name]
    for name, rows in ((collection, docs), ("staffs", staffs), ("facilities", facilities)):
        db[name].drop()
        db[name].insert_many(rows)
    db["staffs"].create_index("cName")
    db["facilities"].create_index("facilityId")
    print(f"seeded {len(docs)} residents ({len(staffs)} staff, {len(facilities)} facilities) into {db_name}")


if __name__ == "__main__":
    main()
