"""Resident lookups: answer only what was asked, and narrow to the resident before any join (no full scans)."""

from app.data.resident_view import ResidentView, ViewRepository
from app.engine.executor import ExecutionData
from app.engine.responder import build_response
from app.models.plan import QueryPlan
from tests.conftest import FakeTranslator


class RecordingRepository(ViewRepository):
    """Captures the aggregation pipelines instead of running them."""

    def __init__(self):
        super().__init__(ResidentView())
        self.pipelines: list[list[dict]] = []

    async def _aggregate(self, pipeline):
        self.pipelines.append(pipeline)
        return [{"n": 425}] if pipeline[-1] == {"$count": "n"} else []


def stage_names(pipeline: list[dict]) -> list[str]:
    return [next(iter(s)) for s in pipeline]


async def test_id_lookup_narrows_before_any_join():
    repo = RecordingRepository()
    await repo.find_by_ids(["6a8829288c421a3a7f0e3166"])
    p = repo.pipelines[0]
    names = stage_names(p)
    # soft-delete filter, then the id match, and only then the joins and field mapping
    assert names[:2] == ["$match", "$match"] and names.index("$lookup") > 1
    id_match = p[1]["$match"]["$expr"]["$in"]
    assert id_match[1] == ["6A8829288C421A3A7F0E3166"]  # compared without case
    # the id expression is the schema's own residentId mapping (configuration, not hard-coded)
    assert id_match[0] == {"$toUpper": ResidentView().fields["residentId"]}


async def test_census_total_needs_no_joins():
    repo = RecordingRepository()
    assert await repo.total() == 425
    assert stage_names(repo.pipelines[0]) == ["$match", "$count"]


RESIDENT = {
    "residentId": "RES-1001", "firstName": "Eleanor", "lastName": "Vance", "status": "active",
    "careType": "memory_care", "gender": "Female", "age": 82, "roomNumber": "MC-101", "wing": "Garden Terrace",
    "primaryCaregiver": "Nurse Sarah Jenkins, RN", "admissionDate": "2026-01-15", "createdDate": "2026-01-10",
    "dischargeDate": None, "emergencyContact": {"name": "Thomas Vance", "relationship": "Son", "phone": "555-0192"},
}  # fmt: skip


def lookup_response(attributes: list[str]) -> dict:
    plan = QueryPlan.model_validate(
        {"action": "lookup", "lookup": {"residents": ["RES-1001"], "attributes": attributes}}
    )
    return build_response(plan, ExecutionData(total=88, matched=1, rows=[RESIDENT], looked_up=[RESIDENT]))


def test_attribute_answer_sends_only_the_asked_fact():
    r = lookup_response(["room_number"])
    assert r["headline"].startswith("MC-101") and r["attributeSpotlight"]["label"] == "Room"
    # no full record: no emergency contact, caregiver, age or admission date leaves the server
    assert "matchedResident" not in r and "residentsSample" not in r
    assert "555-0192" not in str(r) and "Sarah Jenkins" not in str(r)


def test_general_question_still_shows_the_profile():
    r = lookup_response([])
    assert r["matchedResident"]["residentId"] == "RES-1001"


async def test_room_question_end_to_end_returns_just_the_room(make_service):
    q = "What is the room number of Eleanor Vance?"
    fake = FakeTranslator(
        {"What is the room number of resident_1?": {"action": "lookup", "lookup": {"residents": ["resident_1"], "attributes": ["room_number"]}}}
    )
    service = await make_service(fake)
    result = await service.handle(q)
    assert result["response"]["headline"].startswith("MC-101")
    assert "matchedResident" not in result["response"] and "residentsSample" not in result["response"]


def test_resident_id_answer():
    r = lookup_response(["resident_id"])
    assert r["attributeSpotlight"]["label"] == "Resident ID" and r["attributeSpotlight"]["value"] == "RES-1001"
