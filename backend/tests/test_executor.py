"""Execution correctness against the seed data (expected values computed independently from residents.json)."""

import pytest

from app.engine.executor import execute
from app.engine.responder import build_response
from app.models.plan import QueryPlan


async def run(repo, plan: dict):
    p = QueryPlan.model_validate(plan)
    return await execute(p, repo), p


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({}, 88),
        ({"status": {"include": ["active"]}}, 68),
        ({"status": {"exclude": ["active"]}}, 20),
        ({"status": {"include": ["pending", "inactive"]}}, 11),
        ({"care_type": {"include": ["memory_care"]}}, 19),
        ({"care_type": {"include": ["skilled_nursing", "memory_care"]}}, 35),
        ({"status": {"include": ["active"]}, "care_type": {"include": ["memory_care"]}}, 15),
        ({"status": {"include": ["active"]}, "care_type": {"exclude": ["memory_care"]}}, 53),
        ({"status": {"include": ["active"]}, "primary_caregiver": {"include": ["Nurse Sarah Jenkins, RN"]}}, 9),
        ({"gender": {"include": ["Male"]}}, 45),
        ({"age": {"min": 81}}, 39),
        ({"age": {"min": 80, "max": 89}}, 43),
        ({"wing": {"exclude": ["Garden Terrace"]}}, 61),
        ({"gender": {"include": ["Female"]}, "age": {"min": 86}, "wing": {"include": ["North Wing"]}}, 2),
        ({"admission_date": {"start": "2026-10-01", "end": "2026-10-31"}}, 12),
        ({"admission_date": {"start": "2026-03-01", "end": "2026-03-31"}}, 7),
        ({"admission_date": {"start": "2025-01-01", "end": "2025-12-31"}}, 8),
        ({"created_date": {"start": "2026-10-01", "end": "2026-10-31"}}, 16),
        (
            {
                "status": {"include": ["active"]},
                "care_type": {"include": ["memory_care"]},
                "admission_date": {"start": "2026-10-01", "end": "2026-10-31"},
            },
            3,
        ),
    ],
)
async def test_counts(repo, filters, expected):
    data, _ = await run(repo, {"action": "count", "filters": filters})
    assert data.matched == expected


async def test_breakdown_by_wing(repo):
    data, plan = await run(repo, {"action": "breakdown", "group_by": ["wing"]})
    assert {g.key[0]: g.count for g in data.groups} == {
        "Garden Terrace": 27,
        "North Wing": 27,
        "South Wing": 18,
        "East Wing": 16,
    }
    assert build_response(plan, data)["headline"] == "88 residents by Facility"


async def test_average_and_extremes(repo):
    data, _ = await run(repo, {"action": "aggregate", "aggregate": {"op": "avg", "field": "age"}})
    assert data.aggregate_value == 79.3
    data, plan = await run(repo, {"action": "list", "sort": {"field": "age", "direction": "desc"}, "limit": 1})
    assert data.rows[0]["lastName"] == "Kowalski"
    assert build_response(plan, data)["headline"].startswith("Dorothy Kowalski")


async def test_lookup_yes_no(repo):
    data, plan = await run(
        repo,
        {
            "action": "lookup",
            "lookup": {"residents": ["RES-1002"]},
            "filters": {"care_type": {"include": ["assisted_living"]}},
        },
    )
    assert build_response(plan, data)["headline"].startswith("No — Arthur Pendleton")


async def test_bulk_results_hide_emergency_contacts(repo):
    data, _ = await run(repo, {"action": "list", "filters": {"status": {"include": ["active"]}}})
    assert all(r["emergencyContact"]["phone"] == "" for r in data.rows)


async def test_single_lookup_includes_emergency_contact(repo):
    data, plan = await run(
        repo, {"action": "lookup", "lookup": {"residents": ["RES-1001"], "attributes": ["emergency_contact"]}}
    )
    assert "555-0192" in build_response(plan, data)["summaryText"]
