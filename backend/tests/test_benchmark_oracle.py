import pytest

from app.data.memory_repository import MemoryResidentRepository
from app.nlu.translator import ModelUsage, SchemaContext, Translation, TranslationRequest
from scripts.benchmark import ORACLE_RESIDENTS, CountingTranslator, oracle_signature, signature


@pytest.mark.asyncio
async def test_counting_translator_returns_successful_translation_and_records_usage():
    class Translator:
        provider = "test"
        model_name = "test-model"

        async def translate(self, request, timeout=None):
            return Translation(
                {"action": "count"},
                ModelUsage(calls=1, prompt_tokens=4, completion_tokens=2, inference_ms=12),
                self.model_name,
            )

    translator = CountingTranslator(Translator())

    request = TranslationRequest("question", "2026-11-09", SchemaContext([], [], []))
    result = await translator.translate(request)

    assert result.request_ir == {"action": "count"}
    assert translator.calls == 1
    assert translator.inference_ms == [12]
    assert translator.max_prompt_tokens == 4
    assert translator.max_completion_tokens == 2
    assert translator.max_total_tokens == 6
    assert translator.call_timings[0]["outcome"] == "succeeded"


@pytest.mark.asyncio
async def test_independent_oracle_scores_group_counts_and_resident_projections():
    plan = {
        "action": "breakdown",
        "filters": {
            "admission_date": {"start": "2026-08-02"},
            "status": {"include": ["active", "discharged"]},
        },
        "group_by": ["status"],
        "fields": ["room_number"],
    }

    expected = oracle_signature(plan)
    actual = await signature(plan, MemoryResidentRepository(ORACLE_RESIDENTS))

    assert expected == actual
    assert expected[1] == ((("active",), 28), (("discharged",), 1))
    members = dict(expected[2])
    assert len(members[("active",)]) == 28
    assert members[("active",)][0] == ("RES-1008", "Walter", "Bishop", "MC-125")
    assert members[("discharged",)] == (("RES-1071", "Benjamin", "Franklin", "DC-05"),)
    assert expected[3] is True


@pytest.mark.asyncio
async def test_independent_oracle_scores_list_projection_and_incomplete_page():
    plan = {
        "action": "list",
        "sort": {"field": "last_name", "direction": "desc"},
        "limit": 5,
        "fields": ["room_number"],
    }

    expected = oracle_signature(plan)
    actual = await signature(plan, MemoryResidentRepository(ORACLE_RESIDENTS))

    assert expected == actual
    assert expected[1] == len(ORACLE_RESIDENTS)
    assert expected[3] == 5
    assert expected[4] is False


@pytest.mark.asyncio
async def test_independent_oracle_scores_empty_requested_breakdown_groups():
    plan = {
        "action": "breakdown",
        "filters": {"status": {"include": ["active", "discharged"]}},
        "group_by": ["status"],
        "fields": ["room_number"],
    }
    residents = [
        {
            "residentId": "RES-TEST-1",
            "firstName": "Test",
            "lastName": "Resident",
            "status": "active",
            "roomNumber": "A-1",
        }
    ]

    expected = oracle_signature(plan, residents)
    actual = await signature(plan, MemoryResidentRepository(residents))

    assert expected == actual
    assert expected[1] == ((("active",), 1), (("discharged",), 0))
    assert dict(expected[2])[("discharged",)] == ()
    assert expected[3] is True


@pytest.mark.asyncio
async def test_independent_oracle_scores_grouped_aggregates():
    plan = {
        "action": "aggregate",
        "filters": {"status": {"include": ["active", "discharged"]}},
        "group_by": ["status"],
        "aggregate": {"op": "avg", "field": "age"},
    }

    expected = oracle_signature(plan)
    actual = await signature(plan, MemoryResidentRepository(ORACLE_RESIDENTS))

    assert expected == actual
