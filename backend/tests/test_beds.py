"""Bed questions (app/engine/beds.py): the `beds` collection through snf_mcp for admins, no model; staff not supported."""

import pytest

from app.auth.principal import Principal
from app.data.memory_repository import MemoryResidentRepository
from app.data.repository import NotSupported
from app.engine import beds
from app.engine.service import QueryService
from tests.conftest import FakeTranslator

# 3 rooms, 6 beds: the shape of the Care App `beds` collection (PCC sync)
ROOMS = [
    {"roomId": 101, "updatedAt": {"$date": "2026-11-27T05:44:15.412Z"}, "beds": [
        {"id": 1, "description": "A", "isOccupied": True, "isMedicareCertified": True, "isMedicaidCertified": True},
        {"id": 2, "description": "B", "isOccupied": False, "isMedicareCertified": True, "isMedicaidCertified": False}]},
    {"roomId": 102, "updatedAt": {"$date": "2026-11-27T05:44:15.412Z"}, "beds": [
        {"id": 3, "description": "A", "isOccupied": True, "isMedicareCertified": False, "isMedicaidCertified": True},
        {"id": 4, "description": "B", "isOccupied": True, "isCompanionBed": True}]},
    {"roomId": 103, "updatedAt": {"$date": "2026-11-27T05:44:15.412Z"}, "beds": [
        {"id": 5, "description": "A", "isOccupied": False, "isMedicaidCertified": True},
        {"id": 6, "description": "B", "isOccupied": True}]},
]  # fmt: skip


def ask(question: str) -> dict:
    q = beds.parse(question)
    assert q is not None, question
    return beds.answer(q, ROOMS, "UTC")


def test_the_summary_matches_the_occupancy_api():
    r = ask("What is the bed occupancy?")  # GET /api/beds/occupancy: rooms 3, beds 6, occupied 4, available 2
    assert r["summaryText"].startswith("6 beds in 3 rooms: 4 occupied, 2 available, occupancy 66.7%")
    assert "as of the last PointClickCare sync, Nov 27, 2026 05:44" in r["summaryText"]


@pytest.mark.parametrize(
    ("question", "count", "text"),
    [
        ("How many beds are available?", 2, "2 available beds of 6 beds"),
        ("How many beds are occupied right now?", 4, "4 occupied beds of 6 beds"),
        ("How many vacant beds do we have?", 2, "2 available beds"),
        ("How many Medicare certified beds are available?", 1, "1 available Medicare-certified bed of 6 beds"),
        ("How many non-Medicaid beds are occupied?", 2, "2 occupied non-Medicaid-certified beds"),  # 102 B, 103 B
        ("How many companion beds are there?", 1, "1 companion bed"),
        ("How many beds are not occupied?", 2, "2 available beds"),
        ("How many beds in room 101 are free?", 1, "1 available bed in room 101 of 2 beds"),
        ("How many rooms do we have?", 3, "3 rooms"),
        ("How many rooms have available beds?", 2, "2 rooms with available beds"),
    ],
)
def test_counts(question, count, text):
    r = ask(question)
    assert r["count"] == count and r["summaryText"].startswith(text), r["summaryText"]


def test_rate_list_rooms_and_breakdown():
    assert ask("What is the occupancy rate?")["summaryText"].startswith("4 of 6 beds are occupied: 66.7%")
    assert ask("Occupancy rate of Medicare beds")["summaryText"].startswith("1 of 2 Medicare-certified beds are occupied: 50.0%")
    listed = ask("List all available beds")["summaryText"]
    assert listed.endswith("Room 101: B; Room 103: A")
    assert ask("Which rooms have available beds?")["summaryText"].endswith("Rooms: 101, 103")
    by_room = ask("Occupied beds by room")
    assert [(b["label"], b["count"]) for b in by_room["breakdown"]] == [("Room 101", 1), ("Room 102", 2), ("Room 103", 1)]
    missing = ask("How many beds are free in room 999?")
    assert missing["needsClarification"] and "Room 999 is not in the bed records" in missing["summaryText"]


@pytest.mark.parametrize("question", ["How many beds are in the memory care wing?", "Beds with more than 5 patients",
                                      "How many beds were occupied last month?"])
def test_a_bed_question_with_a_word_it_does_not_understand_is_never_answered_with_a_number(question):
    assert beds.parse(question) is None


@pytest.mark.parametrize(
    ("question", "named", "is_beds"),
    [
        ("What is the bed occupancy?", False, True),
        ("How many rooms do we have?", False, True),
        ("What bed is Eleanor Vance in?", True, False),  # a resident's room
        ("List residents with their room numbers", False, False),
        ("Who is in room 101?", False, False),
        ("How many residents are active?", False, False),
    ],
)
def test_only_questions_about_beds_take_the_bed_path(question, named, is_beds):
    assert beds.is_bed_question(question, named) is is_beds


@pytest.mark.parametrize(
    ("question", "person_name", "is_beds"),
    [
        ("In which room is Sagar staying?", "Sagar", False),  # an unknown person: their "not found" path
        ("What is Sagar Dodia's room number?", "Sagar Dodia", False),
        ("What is the bed occupancy?", "occupancy", True),  # the one-person grammar misreads these as names
        ("List all available beds", "available beds", True),
        ("Occupied beds by room", "Occupied beds by", True),
    ],
)
def test_a_name_that_is_not_bed_words_makes_it_a_person_question(question, person_name, is_beds):
    assert beds.is_bed_question(question, False, person_name) is is_beds


class BedRepo(MemoryResidentRepository):
    """The test residents plus the snf_mcp bed reader: admins get the rooms, staff are not supported."""

    staff = False

    async def bed_rooms(self):
        if self.staff:
            raise NotSupported("no snf_mcp tool for staff questions about beds", ("beds",))
        return ROOMS


ADMIN = Principal(kind="admin", id="a-1", facility_id="FAC-A")
STAFF = Principal(kind="staff", id="s-1", facility_id="FAC-A", scope_key="s-1",
                  modules=frozenset({"residents", "residents list"}))  # fmt: skip


async def test_admins_get_bed_answers_without_any_model_call(settings):
    fake = FakeTranslator({})  # any model call fails the test
    service = QueryService(BedRepo(), fake, settings)
    await service.refresh_vocabulary()
    r = await service.handle("How many beds are available?", principal=ADMIN)
    assert r["tier"] == "direct" and r["response"]["count"] == 2 and fake.calls == []
    assert r["interpretation"] == ["Beds", "Available"] and r["intent"]["category"] == "BED_QUERY"
    assert r["policy"]["policyCategory"] == "BED_INTENT"
    vague = await service.handle("How many beds are in the memory care wing?", principal=ADMIN)
    assert vague["response"]["needsClarification"] and "I can answer bed questions" in vague["response"]["summaryText"]


async def test_staff_bed_questions_are_not_supported(settings):
    repo = BedRepo()
    repo.staff = True
    service = QueryService(repo, FakeTranslator({}), settings)
    await service.refresh_vocabulary()
    r = await service.handle("What is the bed occupancy?", principal=STAFF)
    assert r["policy"]["policyCategory"] == "NOT_SUPPORTED" and r["response"]["isNotSupported"] is True
    assert r["response"]["summaryText"].startswith("Beds data isn't available to the assistant for your role")
