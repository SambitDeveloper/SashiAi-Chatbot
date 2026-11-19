"""Independently authored RequestIR fixtures for semantics and lossless lowering."""

import pytest

from app.data.memory_repository import MemoryResidentRepository
from app.nlu.request_ir import IRCompilationError, compile_request_ir
from scripts.benchmark import ORACLE_RESIDENTS, oracle_signature, signature

BREAKDOWN_QUESTION = (
    "Among residents admitted after August 1, 2026, how many are Active and how many are Discharged? "
    "Show the names and room numbers in each group."
)
BREAKDOWN_IR = {
    "action": "breakdown",
    "action_status": "supported",
    "action_evidence": "how many are Active and how many are Discharged",
    "where": {
        "kind": "and",
        "status": "supported",
        "evidence": "admitted after August 1, 2026, how many are Active and how many are Discharged",
        "children": [
            {
                "kind": "predicate",
                "status": "supported",
                "field": "admission_date",
                "operator": "gt",
                "value": "2026-08-01",
                "evidence": "admitted after August 1, 2026",
            },
            {
                "kind": "predicate",
                "status": "supported",
                "field": "status",
                "operator": "in",
                "value": ["active", "discharged"],
                "evidence": "Active and how many are Discharged",
            },
        ],
    },
    "group_by": [{"field": "status", "status": "supported", "evidence": "Active and how many are Discharged"}],
    "aggregate": None,
    "projection": [{"field": "room_number", "status": "supported", "evidence": "room numbers"}],
    "sort": None,
    "limit": None,
    "lookup": None,
    "message": None,
    "uses_previous": False,
    "unsupported": [],
    "unresolved": [],
    "ambiguous": [],
}


def ir(*, action="count", where=None, source="Find residents.", **overrides):
    base = {
        "action": action,
        "action_status": "supported",
        "action_evidence": source,
        "where": where,
        "group_by": [],
        "aggregate": None,
        "projection": [],
        "sort": None,
        "limit": None,
        "lookup": None,
        "message": None,
        "uses_previous": False,
        "unsupported": [],
        "unresolved": [],
        "ambiguous": [],
    }
    return {**base, **overrides}


def predicate(field, operator, value, evidence):
    return {
        "kind": "predicate",
        "status": "supported",
        "field": field,
        "operator": operator,
        "value": value,
        "evidence": evidence,
    }


def compile_ir(candidate, source):
    return compile_request_ir(candidate, source, ["active", "away", "discharged"], ["memory_care"])[1]


@pytest.mark.asyncio
async def test_composed_date_status_breakdown_and_projection_compile_and_execute():
    plan = compile_ir(BREAKDOWN_IR, BREAKDOWN_QUESTION)

    assert plan == {
        "action": "breakdown",
        "filters": {
            "admission_date": {"start": "2026-08-02"},
            "status": {"include": ["active", "discharged"]},
        },
        "group_by": ["status"],
        "fields": ["room_number"],
    }

    actual = await signature(plan, MemoryResidentRepository(ORACLE_RESIDENTS))
    expected = oracle_signature(plan, ORACLE_RESIDENTS)
    assert actual == expected
    assert actual[1] == ((("active",), 28), (("discharged",), 1))
    assert all(
        len(resident) == 4 and resident[1] and resident[2] and resident[3]
        for group in actual[2]
        for resident in group[1]
    )


def test_or_and_minimal_pair_preserves_distinct_boolean_meaning():
    source_or = "Find residents who are Active or Discharged."
    source_and = "Find residents who are Active and Discharged."
    disjunction = {
        "kind": "or",
        "status": "supported",
        "evidence": "Active or Discharged",
        "children": [
            predicate("status", "eq", "active", "Active"),
            predicate("status", "eq", "discharged", "Discharged"),
        ],
    }
    conjunction = {
        "kind": "and",
        "status": "supported",
        "evidence": "Active and Discharged",
        "children": [
            predicate("status", "eq", "active", "Active"),
            predicate("status", "eq", "discharged", "Discharged"),
        ],
    }

    assert compile_ir(ir(where=disjunction, source="Find residents who are Active or Discharged."), source_or)[
        "filters"
    ] == {"status": {"include": ["active", "discharged"]}}
    with pytest.raises(IRCompilationError, match="contradictory"):
        compile_ir(ir(where=conjunction, source=source_and), source_and)


@pytest.mark.parametrize(
    ("operator", "value", "expected"),
    [
        ("gt", "2026-08-01", {"admission_date": {"start": "2026-08-02"}}),
        ("gte", "2026-08-01", {"admission_date": {"start": "2026-08-01"}}),
    ],
)
def test_strict_and_inclusive_date_comparisons_have_distinct_boundaries(operator, value, expected):
    source = "admitted after August 1, 2026" if operator == "gt" else "admitted on or after August 1, 2026"
    evidence = "August 1, 2026"
    candidate = ir(
        where=predicate("admission_date", operator, value, evidence),
        source=source,
    )
    assert compile_ir(candidate, source)["filters"] == expected


@pytest.mark.parametrize(
    ("operator", "expected"),
    [("gt", {"age": {"min": 81}}), ("gte", {"age": {"min": 80}})],
)
def test_strict_and_inclusive_age_comparisons_have_distinct_boundaries(operator, expected):
    source = "residents over 80" if operator == "gt" else "residents age 80 or older"
    candidate = ir(where=predicate("age", operator, 80, "80"), source=source)
    assert compile_ir(candidate, source)["filters"] == expected


def test_negation_compiles_to_exclusion_without_changing_polarity():
    source = "Find residents who are not Active."
    candidate = ir(
        where={
            "kind": "not",
            "status": "supported",
            "evidence": "not Active",
            "child": predicate("status", "eq", "active", "Active"),
        },
        source=source,
    )
    assert compile_ir(candidate, source)["filters"] == {"status": {"exclude": ["active"]}}


def test_cross_field_or_fails_closed_instead_of_flattening():
    source = "Find residents who are Active or over 80."
    candidate = ir(
        where={
            "kind": "or",
            "status": "supported",
            "evidence": "Active or over 80",
            "children": [
                predicate("status", "eq", "active", "Active"),
                predicate("age", "gt", 80, "over 80"),
            ],
        },
        source=source,
    )
    with pytest.raises(IRCompilationError, match="cross-field OR"):
        compile_ir(candidate, source)


def test_grouping_is_separate_from_filtering():
    source = "Show each status group."
    grouped = ir(
        action="breakdown",
        source=source,
        group_by=[{"field": "status", "status": "supported", "evidence": "each status"}],
    )
    filtered = ir(
        source="Find Active residents.",
        where=predicate("status", "eq", "active", "Active"),
    )

    assert compile_ir(grouped, source)["group_by"] == ["status"]
    assert compile_ir(grouped, source)["filters"] == {}
    assert compile_ir(filtered, "Find Active residents.")["filters"] == {"status": {"include": ["active"]}}


def test_source_evidence_does_not_establish_semantic_equivalence():
    from app.nlu.request_ir import RequestIR, _validate_evidence

    source = "Find Active residents."
    candidate = ir(
        where=predicate("status", "eq", "discharged", "Active"),
        source=source,
    )
    parsed = RequestIR.model_validate(candidate)
    _validate_evidence(parsed, source)
    expected = predicate("status", "eq", "active", "Active")
    assert candidate["where"] != expected

    candidate["where"]["evidence"] = "therefore"
    with pytest.raises(IRCompilationError, match="source evidence"):
        compile_ir(candidate, source)


def test_non_supported_nodes_and_relationships_cannot_compile():
    source = "Find residents linked to a facility."
    unsupported_node = ir(
        source=source,
        where={
            "kind": "predicate",
            "status": "unsupported",
            "field": "status",
            "operator": "eq",
            "value": "active",
            "evidence": "linked to a facility",
        },
    )
    with pytest.raises(IRCompilationError, match="non-executable"):
        compile_ir(unsupported_node, source)

    relationship = ir(
        source=source,
        where={
            "kind": "relationship",
            "status": "supported",
            "relation": "resident_facility",
            "target": "facility",
            "evidence": "linked to a facility",
        },
    )
    with pytest.raises(IRCompilationError, match="relationship traversal"):
        compile_ir(relationship, source)

    unsupported_group = ir(
        action="breakdown",
        source="Show each status group.",
        group_by=[{"field": "status", "status": "unsupported", "evidence": "each status"}],
    )
    with pytest.raises(IRCompilationError, match="non-executable"):
        compile_ir(unsupported_group, "Show each status group.")


def test_ambiguous_interpretation_cannot_compile():
    source = "Find residents with a recent admission."
    candidate = ir(
        source=source,
        ambiguous=[{"reason": "recent has no defined date boundary", "evidence": "recent admission"}],
    )
    with pytest.raises(IRCompilationError, match="unsupported or unresolved"):
        compile_ir(candidate, source)


def test_limit_is_an_evidenced_semantic_node_and_lowers_to_queryplan():
    source = "List the 3 oldest residents."
    candidate = ir(
        action="list",
        source=source,
        limit={"value": 3, "status": "supported", "evidence": "3"},
        sort={"field": "age", "direction": "desc", "status": "supported", "evidence": "oldest"},
    )
    assert compile_ir(candidate, source)["limit"] == 3


def test_unresolved_material_and_invalid_query_shape_cannot_compile():
    source = "Find residents with an unsupported comparison."
    unresolved = ir(
        source=source,
        unresolved=[{"reason": "unresolved comparison", "evidence": "unsupported comparison"}],
    )
    with pytest.raises(IRCompilationError, match="unresolved"):
        compile_ir(unresolved, source)

    malformed = ir(action="breakdown", source=source)
    with pytest.raises(IRCompilationError, match="QueryPlan"):
        compile_ir(malformed, source)
