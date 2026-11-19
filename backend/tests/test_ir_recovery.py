"""Model-independent recovery: meaning-preserving normalisation, one retry with exact validator feedback, specific
messages for unsupported capabilities, exact errors in the trace — and strict validation throughout."""

import copy

import pytest

from app.nlu.request_ir import IRCompilationError, compile_request_ir, normalize_request_ir
from tests.conftest import FakeTranslator


def P(evidence, field, op, value, status="supported"):
    return {"kind": "predicate", "status": status, "evidence": evidence, "field": field, "operator": op, "value": value}


def IR(where, action="count", evidence="How many", **kw):
    return {"action": action, "action_status": "supported", "action_evidence": evidence, "where": where,
            "unsupported": [], "unresolved": [], "ambiguous": [], **kw}  # fmt: skip


def AND(*kids, status="supported"):
    return {"kind": "and", "status": status, "evidence": "residents", "children": list(kids)}


def OR(*kids, status="supported"):
    return {"kind": "or", "status": status, "evidence": "residents", "children": list(kids)}


ACTIVE = P("active", "status", "eq", "active")
AWAY = P("away", "status", "eq", "away")
OVER_80 = P("over 80", "age", "gt", 80)
Q = "How many active residents over 80 or away residents are there"


# ======================================================================== normaliser


def test_single_child_and_is_unwrapped_and_then_compiles():
    raw = IR(AND(ACTIVE))
    norm, applied = normalize_request_ir(raw)
    assert norm["where"] == ACTIVE and applied == ["unwrapped single-child and"]
    with pytest.raises(IRCompilationError):  # validation itself stays strict
        compile_request_ir(raw, Q)
    compile_request_ir(norm, Q)


def test_nested_same_kind_is_flattened():
    norm, applied = normalize_request_ir(IR(AND(ACTIVE, AND(OVER_80, ACTIVE))))
    assert norm["where"]["children"] == [ACTIVE, OVER_80, ACTIVE] and "flattened nested and" in applied
    norm, _ = normalize_request_ir(IR(OR(ACTIVE, OR(AWAY, ACTIVE))))
    assert len(norm["where"]["children"]) == 3


def test_double_negation_is_removed():
    neg = {"kind": "not", "status": "supported", "evidence": "not", "child": {
        "kind": "not", "status": "supported", "evidence": "not", "child": ACTIVE}}  # fmt: skip
    norm, applied = normalize_request_ir(IR(neg))
    assert norm["where"] == ACTIVE and applied == ["removed double negation"]


def test_null_facets_become_empty_lists():
    norm, applied = normalize_request_ir(IR(ACTIVE, group_by=None, projection=None))
    assert norm["group_by"] == [] and norm["projection"] == [] and len(applied) == 2


@pytest.mark.parametrize(
    "where",
    [
        AND(ACTIVE, status="ambiguous"),  # a status would be dropped: not provably equivalent
        AND(ACTIVE, AND(OVER_80, ACTIVE, status="ambiguous")),
        OR(ACTIVE, AND(OVER_80, AWAY)),  # different kinds: not associative
        AND(ACTIVE, OVER_80),  # already canonical
        P("x", "room_number", "like", "A%"),  # not a structural variant: left for the validator + retry
    ],
)
def test_anything_not_provably_equivalent_is_left_alone(where):
    raw = IR(where)
    norm, applied = normalize_request_ir(raw)
    assert applied == [] or (where["kind"] == "and" and where.get("status") != "supported" and norm == raw)
    assert norm == raw


def test_normaliser_never_mutates_its_input():
    raw = IR(AND(ACTIVE))
    before = copy.deepcopy(raw)
    normalize_request_ir(raw)
    assert raw == before


# ======================================================================== service: retry with validator feedback

Q_ACTIVE = "How many residents are active?"


def ir(where, **kw):
    return {"request_ir": IR(where, evidence="How many", **kw)}


async def test_compile_failure_is_retried_once_with_the_exact_error(make_service):
    bad = ir(P("active", "status", "like", "act%"))
    good = ir(P("active", "status", "eq", "active"))
    fake = FakeTranslator({Q_ACTIVE: [bad, good]})
    r = await (await make_service(fake)).handle(Q_ACTIVE)
    assert len(fake.calls) == 2 and r["tier"] == "llm" and r["response"]["count"] > 0
    hint = fake.calls[1]["hint"]
    assert "where.operator" in hint and "'like'" in hint and "starts_with" in hint  # exact validator output
    assert fake.calls[1]["retry_ir_json"]  # the rejected candidate is shown to the model
    first = r["modelPlans"][0]
    assert first["error"] == "unexecutable RequestIR" and first["compileError"]["code"] == "structure"
    assert "'like'" in first["compileError"]["details"][0]  # the exact error is in the trace
    assert r["modelCalls"] == 2 and r["tokenUsage"]["prompt_tokens"] == 1400  # each call counted once


async def test_normalised_plan_needs_no_retry_and_is_traced(make_service):
    fake = FakeTranslator({Q_ACTIVE: ir(AND(P("active", "status", "eq", "active")))})
    r = await (await make_service(fake)).handle(Q_ACTIVE)
    assert len(fake.calls) == 1 and r["response"]["count"] > 0
    assert r["modelPlans"][0]["normalized"] == ["unwrapped single-child and"]


async def test_never_more_than_one_retry_and_an_invalid_plan_never_executes(make_service):
    bad = ir(P("active", "status", "like", "act%"))
    fake = FakeTranslator({Q_ACTIVE: [bad, bad]})
    r = await (await make_service(fake)).handle(Q_ACTIVE)
    assert len(fake.calls) == 2
    assert r["response"]["needsClarification"] and r["mcpCall"] is None
    assert r["modelPlans"][1].get("identical_to_previous") is True


async def test_compile_retry_then_grounding_problem_does_not_retry_again(make_service):
    """The one retry is shared by every trigger: a compile retry that then fails grounding goes to the user."""
    q = "How many residents are active and in memory care?"
    bad = ir(P("active", "status", "like", "act%"))
    drops_care = ir(P("active", "status", "eq", "active"))  # compiles, but drops "memory care"
    fake = FakeTranslator({q: [bad, drops_care, drops_care]})
    r = await (await make_service(fake)).handle(q)
    assert len(fake.calls) == 2 and r["response"]["needsClarification"]


async def test_declared_unsupported_is_not_retried_and_names_the_part(make_service):
    q = "How many residents are on insulin?"
    declared = {"request_ir": {**IR(None, evidence="How many"),
                               "unsupported": [{"reason": "medication data", "evidence": "on insulin"}]}}  # fmt: skip
    fake = FakeTranslator({q: declared})
    r = await (await make_service(fake)).handle(q)
    assert len(fake.calls) == 1
    assert '"on insulin"' in r["response"]["summaryText"]
    assert r["modelPlans"][0]["compileError"]["code"] == "declared_unsupported"


async def test_declared_ambiguity_asks_about_that_part(make_service):
    q = "How many residents are in the new wing?"
    declared = {"request_ir": {**IR(None, evidence="How many"),
                               "ambiguous": [{"reason": "which facility", "evidence": "the new wing"}]}}  # fmt: skip
    r = await (await make_service(FakeTranslator({q: declared}))).handle(q)
    assert 'clarify what you mean by "the new wing"' in r["response"]["summaryText"]


async def test_unsupported_capability_is_named_after_the_retry(make_service):
    q = 'How many residents have a room that does not start with "A"?'
    neg = ir({"kind": "not", "status": "supported", "evidence": "does not start with",
              "child": P('start with "A"', "room_number", "starts_with", "A")})  # fmt: skip
    fake = FakeTranslator({q: [neg, neg]})
    r = await (await make_service(fake)).handle(q)
    assert len(fake.calls) == 2  # retried once: the model might have restructured it
    text = r["response"]["summaryText"]
    assert "excluding by a text match" in text and "isn't supported" in text and r["mcpCall"] is None


async def test_masked_resident_names_are_restored_in_quotes(make_service):
    q = "Which hobbies does Eleanor Vance enjoy?"  # not a policy-gated topic: it reaches the model
    declared = {"request_ir": {**IR(None, action="lookup", evidence="Which"),
                               "unsupported": [{"reason": "hobbies", "evidence": "hobbies does resident_1 enjoy"}]}}  # fmt: skip
    fake = FakeTranslator({"Which hobbies does resident_1 enjoy?": declared})
    r = await (await make_service(fake)).handle(q)
    text = r["response"]["summaryText"]
    assert len(fake.calls) == 1 and fake.calls[0]["question"] == "Which hobbies does resident_1 enjoy?"  # masked out
    assert '"hobbies does Eleanor Vance enjoy"' in text and "resident_1" not in text  # restored for the user only
