"""Compositional request interpretation and its conservative QueryPlan compiler."""

import copy
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.models.plan import TEXT_MATCH_FIELDS, QueryPlan, canonical_field

Action = Literal["count", "list", "breakdown", "aggregate", "lookup", "clarify", "unsupported", "help"]
PredicateOperator = Literal[
    "eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte", "between", "missing", "present", "starts_with", "contains"
]
NodeStatus = Literal["supported", "ambiguous", "unsupported", "unresolved"]
Scalar = str | int

# The ONE source of which fields and operators an interpretation may use. The prompt (nlu/prompt.py) and the output
# schema are generated from these, so what the model is told can never drift from what this compiler accepts.
SET_FIELDS: tuple[str, ...] = ("status", "care_type", "facility", "gender", "primary_caregiver", "room_number")
DATE_FIELDS: tuple[str, ...] = ("admission_date", "created_date", "discharge_date")
RANGE_FIELDS: tuple[str, ...] = ("age", *DATE_FIELDS)
CONDITION_FIELDS: tuple[str, ...] = (*SET_FIELDS, *RANGE_FIELDS)
GROUP_FIELDS: tuple[str, ...] = ("status", "care_type", "facility", "gender", "primary_caregiver")
SORT_FIELDS: tuple[str, ...] = ("admission_date", "created_date", "discharge_date", "age", "last_name")
ATTRIBUTES: tuple[str, ...] = (
    "primary_caregiver", "room_number", "facility", "admission_date", "created_date", "discharge_date",
    "emergency_contact", "resident_id", "status", "care_type", "age", "gender",
)  # fmt: skip
SET_OPERATORS: tuple[str, ...] = ("eq", "ne", "in", "not_in", "missing", "present")
TEXT_OPERATORS: tuple[str, ...] = ("starts_with", "contains")  # only on TEXT_MATCH_FIELDS (app/models/plan.py)
RANGE_OPERATORS: tuple[str, ...] = ("gt", "gte", "lt", "lte", "between", "missing", "present")
assert set(PredicateOperator.__args__) == {*SET_OPERATORS, *TEXT_OPERATORS, *RANGE_OPERATORS}  # type: ignore[attr-defined]
assert set(TEXT_MATCH_FIELDS) <= set(SET_FIELDS)


def operators_for(field: str) -> tuple[str, ...]:
    """The predicate operators the compiler accepts on one condition field."""
    if field in RANGE_FIELDS:
        return RANGE_OPERATORS
    if field in TEXT_MATCH_FIELDS:
        return (*SET_OPERATORS, *TEXT_OPERATORS)
    if field in SET_FIELDS:
        return SET_OPERATORS
    return ()


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class IRNode(_Strict):
    kind: Literal["predicate", "and", "or", "not", "relationship"]
    status: NodeStatus
    evidence: str = Field(min_length=1, max_length=300)
    field: str | None = None
    operator: PredicateOperator | None = None
    value: Scalar | list[Scalar] | None = None
    relation: str | None = None
    target: str | None = None
    children: list["IRNode"] | None = Field(default=None, min_length=2, max_length=12)
    child: "IRNode | None" = None

    @model_validator(mode="after")
    def _shape(self):
        if self.kind == "predicate":
            if (
                not self.field
                or self.operator is None
                or self.children is not None
                or self.child is not None
                or self.relation is not None
                or self.target is not None
            ):
                raise ValueError("predicate needs a field and operator and cannot have children")
            if self.operator in ("missing", "present"):
                if self.value is not None:
                    raise ValueError("missingness predicates do not take a value")
            elif self.value is None:
                raise ValueError("comparison predicate needs a value")
            if self.operator == "between" and (not isinstance(self.value, list) or len(self.value) != 2):
                raise ValueError("between needs exactly two values")
            if self.operator in ("in", "not_in") and (not isinstance(self.value, list) or not self.value):
                raise ValueError("set membership needs a non-empty value list")
            if self.operator not in ("in", "not_in", "between") and isinstance(self.value, list):
                raise ValueError("this operator accepts one value")
        elif self.kind in ("and", "or"):
            if (
                self.field is not None
                or self.operator is not None
                or self.value is not None
                or self.child is not None
                or self.relation is not None
                or self.target is not None
            ):
                raise ValueError("boolean nodes only take children")
            if self.children is None or len(self.children) < 2:
                raise ValueError("boolean nodes need at least two children")
        elif self.kind == "not":
            if (
                self.field is not None
                or self.operator is not None
                or self.value is not None
                or self.children is not None
                or self.child is None
                or self.relation is not None
                or self.target is not None
            ):
                raise ValueError("not needs exactly one child")
        elif self.kind == "relationship":
            if (
                not self.relation
                or not self.target
                or self.field is not None
                or self.operator is not None
                or self.value is not None
                or self.children is not None
                or self.child is not None
            ):
                raise ValueError("relationship needs a relation and target")
        return self


IRNode.model_rebuild()


class IRField(_Strict):
    field: str
    status: NodeStatus
    evidence: str = Field(min_length=1, max_length=300)


class IRAggregate(_Strict):
    op: Literal["avg", "min", "max"]
    field: str = "age"
    status: NodeStatus
    evidence: str = Field(min_length=1, max_length=300)


class IRSort(_Strict):
    field: str
    direction: Literal["asc", "desc"]
    status: NodeStatus
    evidence: str = Field(min_length=1, max_length=300)


class IRLookup(_Strict):
    residents: list[str] = Field(min_length=1, max_length=5)
    attributes: list[str] = Field(default_factory=list, max_length=12)
    status: NodeStatus
    evidence: str = Field(min_length=1, max_length=300)


class IRLimit(_Strict):
    value: int = Field(ge=1, le=200)
    status: NodeStatus
    evidence: str = Field(min_length=1, max_length=300)


class IRDiagnostic(_Strict):
    reason: str = Field(min_length=1, max_length=200)
    evidence: str = Field(min_length=1, max_length=300)


class RequestIR(_Strict):
    action: Action
    action_status: NodeStatus
    action_evidence: str = Field(min_length=1, max_length=300)
    where: IRNode | None
    group_by: list[IRField] = Field(default_factory=list, max_length=2)
    aggregate: IRAggregate | None = None
    projection: list[IRField] = Field(default_factory=list, max_length=12)
    sort: IRSort | None = None
    limit: IRLimit | None = None
    lookup: IRLookup | None = None
    message: str | None = Field(default=None, max_length=500)
    uses_previous: bool = False
    unsupported: list[IRDiagnostic] = Field(max_length=8)
    unresolved: list[IRDiagnostic] = Field(max_length=8)
    ambiguous: list[IRDiagnostic] = Field(max_length=8)

    @model_validator(mode="after")
    def _unique_facets(self):
        for name, items in (("group_by", self.group_by), ("projection", self.projection)):
            fields = [canonical_field(item.field) for item in items]
            if len(fields) != len(set(fields)):
                raise ValueError(f"{name} fields must be unique")
        return self


class IRCompilationError(ValueError):
    """The candidate IR is malformed, unsupported, or cannot be lowered without loss.

    code        machine-readable category (structure, evidence, field, value, capability, declared, plan, ...)
    details     exact problems (validator paths, allowed values) — model retry feedback and the trace, never the user
    retryable   a model could plausibly fix it with feedback (False: the model itself declared it unsupported/ambiguous)
    capability  what the backend cannot do yet, in user-facing words ("NOT of a text match"), when that is the cause
    quotes      for declared_*: the exact question phrases (masked) the model marked unsupported/ambiguous
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid",
        details: list[str] | None = None,
        retryable: bool = True,
        capability: str | None = None,
        quotes: list[str] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.details = details or [message]
        self.retryable = retryable
        self.capability = capability
        self.quotes = quotes or []


def _validation_details(exc: ValidationError, limit: int = 6) -> list[str]:
    """Pydantic errors as 'path: message (got: value)' lines — precise feedback a model can act on."""
    out = []
    for err in exc.errors()[:limit]:
        path = ".".join(str(p) for p in err["loc"])
        got = err.get("input")
        shown = f" (got: {got!r})" if isinstance(got, (str, int, float, bool)) or got is None else ""
        out.append(f"{path or 'RequestIR'}: {err['msg']}{shown}")
    return out


_SET_FIELDS = set(SET_FIELDS)
_DATE_FIELDS = set(DATE_FIELDS)
_GROUP_FIELDS = set(GROUP_FIELDS)
_ATTRIBUTES = set(ATTRIBUTES)
_SORT_FIELDS = set(SORT_FIELDS)
_TEXT_FIELDS = set(TEXT_MATCH_FIELDS)
_STATUS_VALUES = {"active", "away", "inactive", "pending", "admitted", "discharged"}
_CARE_VALUES = {"assisted_living", "memory_care", "independent_living", "skilled_nursing", "respite_care"}
_GENDER_VALUES = {"Female", "Male", "Other"}


_BOOLEAN_ONLY_KEYS = ("field", "operator", "value", "relation", "target", "child")
_MAX_CHILDREN = 12  # IRNode.children max_length


def normalize_request_ir(raw: dict) -> tuple[dict, list[str]]:
    """Rewrite PROVABLY meaning-preserving structural variants of a candidate IR; validation stays strict afterwards.

    Applied only where every node involved has the same status (no status, i.e. no diagnostic, is ever dropped):
      • and/or with exactly one child            → that child          (AND/OR of one term is the term)
      • and inside and, or inside or             → one flat node       (associativity), up to 12 children
      • not(not(x))                              → x                   (double negation)
      • "group_by"/"projection": null            → []                  (both fields default to an empty list)
    Anything else (unknown operators, wrong field names, SQL-style patterns) is left for the validator and a retry.
    Returns (normalised copy, list of the rewrites applied, for the trace).
    """
    if not isinstance(raw, dict):
        return raw, []
    applied: list[str] = []
    out = copy.deepcopy(raw)

    def is_bool(node, kind=None) -> bool:
        return (
            isinstance(node, dict)
            and node.get("kind") in ((kind,) if kind else ("and", "or"))
            and isinstance(node.get("children"), list)
            and all(node.get(k) is None for k in _BOOLEAN_ONLY_KEYS)
        )

    def visit(node):
        if not isinstance(node, dict):
            return node
        if isinstance(node.get("children"), list):
            node["children"] = [visit(child) for child in node["children"]]
        if isinstance(node.get("child"), dict):
            node["child"] = visit(node["child"])
        if is_bool(node):
            kids = node["children"]
            flat: list = []
            for kid in kids:
                if is_bool(kid, node["kind"]) and kid.get("status") == node.get("status"):
                    flat.extend(kid["children"])
                else:
                    flat.append(kid)
            if len(flat) != len(kids) and len(flat) <= _MAX_CHILDREN:
                node["children"] = flat
                applied.append(f"flattened nested {node['kind']}")
            if len(node["children"]) == 1 and isinstance(node["children"][0], dict) and (
                node["children"][0].get("status") == node.get("status")
            ):
                applied.append(f"unwrapped single-child {node['kind']}")
                return node["children"][0]
        if (
            node.get("kind") == "not"
            and isinstance(node.get("child"), dict)
            and node["child"].get("kind") == "not"
            and isinstance(node["child"].get("child"), dict)
            and node.get("status") == node["child"].get("status") == node["child"]["child"].get("status")
        ):
            applied.append("removed double negation")
            return node["child"]["child"]
        return node

    if isinstance(out.get("where"), dict):
        out["where"] = visit(out["where"])
    for key in ("group_by", "projection"):
        if key in out and out[key] is None:
            out[key] = []
            applied.append(f"{key}: null → []")
    return out, applied


def compile_request_ir(
    raw: dict,
    source_text: str,
    allowed_statuses: list[str] | None = None,
    allowed_care_types: list[str] | None = None,
) -> tuple[RequestIR, dict]:
    """Validate evidence and compile one IR to the existing executable QueryPlan shape."""
    try:
        ir = RequestIR.model_validate(raw)
    except ValidationError as exc:
        raise IRCompilationError(
            "invalid RequestIR structure", code="structure", details=_validation_details(exc)
        ) from exc

    _validate_evidence(ir, source_text)
    if ir.unsupported or ir.unresolved or ir.ambiguous:
        # the model itself says part of the question cannot be represented: no retry can change the data model
        kind = "ambiguous" if ir.ambiguous and not (ir.unsupported or ir.unresolved) else "unsupported"
        quotes = [d.evidence for d in (ir.ambiguous if kind == "ambiguous" else [*ir.unsupported, *ir.unresolved])]
        raise IRCompilationError(
            "the interpretation contains unsupported or unresolved material",
            code=f"declared_{kind}",
            details=[f"{d.reason} ({d.evidence!r})" for d in [*ir.unsupported, *ir.unresolved, *ir.ambiguous]],
            retryable=False,
            quotes=quotes,
        )
    facets = [
        (ir.action_status, ir.action_evidence),
        *((item.status, item.evidence) for item in ir.group_by),
        *((item.status, item.evidence) for item in ir.projection),
        *([(ir.aggregate.status, ir.aggregate.evidence)] if ir.aggregate else []),
        *([(ir.sort.status, ir.sort.evidence)] if ir.sort else []),
        *([(ir.limit.status, ir.limit.evidence)] if ir.limit else []),
        *([(ir.lookup.status, ir.lookup.evidence)] if ir.lookup else []),
    ]
    flagged = [(status, evidence) for status, evidence in facets if status != "supported"]
    if flagged:
        raise _declared("the interpretation contains a non-executable semantic node", flagged)
    _validate_node_status(ir.where)

    filters = _compile_where(ir.where, allowed_statuses, allowed_care_types) if ir.where else {}
    group_by = [_known_field(item.field, _GROUP_FIELDS, "grouping") for item in ir.group_by]
    projection = [_known_field(item.field, _ATTRIBUTES, "projection") for item in ir.projection]
    aggregate = None
    if ir.aggregate:
        if ir.aggregate.field != "age":
            raise IRCompilationError(
                "unsupported aggregate field", code="field",
                details=[f"aggregate.field {ir.aggregate.field!r} is not allowed; allowed: age"],
                capability=f"averages/minimums/maximums of {ir.aggregate.field.replace('_', ' ')}",
            )  # fmt: skip
        aggregate = {"op": ir.aggregate.op, "field": "age"}
    sort = None
    if ir.sort:
        sort = {
            "field": _known_field(ir.sort.field, _SORT_FIELDS, "sort"),
            "direction": ir.sort.direction,
        }
    lookup = None
    if ir.lookup:
        attributes = [_known_field(v, _ATTRIBUTES, "lookup attribute") for v in ir.lookup.attributes]
        lookup = {"residents": list(ir.lookup.residents), "attributes": attributes}

    plan = {
        "action": ir.action,
        "filters": filters,
        "group_by": group_by or None,
        "aggregate": aggregate,
        "sort": sort,
        "limit": ir.limit.value if ir.limit else None,
        "lookup": lookup,
        "fields": projection or None,
        "message": ir.message,
        "uses_previous": ir.uses_previous,
    }
    try:
        validated = QueryPlan.model_validate({k: v for k, v in plan.items() if k != "uses_previous"})
    except ValidationError as exc:
        raise IRCompilationError(
            "the interpretation does not meet QueryPlan requirements", code="plan", details=_validation_details(exc)
        ) from exc
    return ir, {
        **validated.model_dump(mode="json", exclude_none=True),
        **({"uses_previous": True} if ir.uses_previous else {}),
    }


def request_ir_from_plan(plan: dict, evidence: str) -> dict:
    """Encode an existing deterministic/test plan as IR without changing its represented operations."""
    action = plan.get("action", "clarify")
    filters = plan.get("filters") or {}
    where_terms: list[dict] = []
    for raw_field, body in filters.items():
        field = canonical_field(raw_field)
        if not isinstance(body, dict):
            raise IRCompilationError("invalid filter body")
        if field in _SET_FIELDS:
            parts: list[dict] = []
            include = body.get("include")
            exclude = body.get("exclude")
            if include:
                parts.append(
                    _predicate(
                        field,
                        "eq" if len(include) == 1 else "in",
                        include[0] if len(include) == 1 else include,
                        evidence,
                    )
                )
            if exclude:
                parts.append(
                    _predicate(
                        field,
                        "ne" if len(exclude) == 1 else "not_in",
                        exclude[0] if len(exclude) == 1 else exclude,
                        evidence,
                    )
                )
            for op in TEXT_OPERATORS:  # needles in one list are OR-ed; the lists are AND-ed with the rest
                needles = body.get(op) or []
                alternatives = [_predicate(field, op, needle, evidence) for needle in needles]
                if len(alternatives) == 1:
                    parts.append(alternatives[0])
                elif alternatives:
                    parts.append({"kind": "or", "status": "supported", "evidence": evidence, "children": alternatives})
            if body.get("missing") is True:
                missing = _predicate(field, "missing", None, evidence)
                parts = (
                    [{"kind": "or", "status": "supported", "evidence": evidence, "children": [*parts, missing]}]
                    if parts
                    else [missing]
                )
            elif body.get("missing") is False:
                present = _predicate(field, "present", None, evidence)
                parts.append(present)
            if len(parts) == 1:
                where_terms.append(parts[0])
            elif parts:
                where_terms.append({"kind": "and", "status": "supported", "evidence": evidence, "children": parts})
            continue
        if field == "age":
            bounds = []
            if body.get("min") is not None:
                bounds.append(_predicate(field, "gte", body["min"], evidence))
            if body.get("max") is not None:
                bounds.append(_predicate(field, "lte", body["max"], evidence))
            if body.get("missing") is True:
                missing = _predicate(field, "missing", None, evidence)
                bounds = (
                    [{"kind": "or", "status": "supported", "evidence": evidence, "children": [*bounds, missing]}]
                    if bounds
                    else [missing]
                )
            elif body.get("missing") is False:
                bounds.append(_predicate(field, "present", None, evidence))
            where_terms.extend(
                bounds
                if len(bounds) <= 1
                else [{"kind": "and", "status": "supported", "evidence": evidence, "children": bounds}]
            )
            continue
        if field in _DATE_FIELDS:
            bounds = []
            if body.get("start") is not None:
                bounds.append(_predicate(field, "gte", str(body["start"]), evidence))
            if body.get("end") is not None:
                bounds.append(_predicate(field, "lte", str(body["end"]), evidence))
            if body.get("missing") is True:
                missing = _predicate(field, "missing", None, evidence)
                bounds = (
                    [{"kind": "or", "status": "supported", "evidence": evidence, "children": [*bounds, missing]}]
                    if bounds
                    else [missing]
                )
            elif body.get("missing") is False:
                bounds.append(_predicate(field, "present", None, evidence))
            where_terms.extend(
                bounds
                if len(bounds) <= 1
                else [{"kind": "and", "status": "supported", "evidence": evidence, "children": bounds}]
            )
            continue
        raise IRCompilationError("unsupported filter field")
    where = None
    if len(where_terms) == 1:
        where = where_terms[0]
    elif where_terms:
        where = {"kind": "and", "status": "supported", "evidence": evidence, "children": where_terms}

    lookup_raw = plan.get("lookup")
    lookup = None
    if lookup_raw:
        lookup = {
            "residents": list(lookup_raw.get("residents") or []),
            "attributes": list(lookup_raw.get("attributes") or []),
            "status": "supported",
            "evidence": evidence,
        }
    return {
        "action": action,
        "action_status": "supported",
        "action_evidence": evidence,
        "where": where,
        "group_by": [
            {"field": canonical_field(v), "status": "supported", "evidence": evidence}
            for v in plan.get("group_by") or []
        ],
        "aggregate": (
            {
                **plan["aggregate"],
                "field": plan["aggregate"].get("field", "age"),
                "status": "supported",
                "evidence": evidence,
            }
            if plan.get("aggregate")
            else None
        ),
        "projection": [
            {"field": canonical_field(v), "status": "supported", "evidence": evidence} for v in plan.get("fields") or []
        ],
        "sort": ({**plan["sort"], "status": "supported", "evidence": evidence} if plan.get("sort") else None),
        "limit": (
            {"value": plan["limit"], "status": "supported", "evidence": evidence}
            if plan.get("limit") is not None
            else None
        ),
        "lookup": lookup,
        "message": plan.get("message"),
        "uses_previous": bool(plan.get("uses_previous")),
        "unsupported": [],
        "unresolved": [],
        "ambiguous": [],
    }


def _predicate(field: str, operator: str, value, evidence: str) -> dict:
    return {
        "kind": "predicate",
        "status": "supported",
        "field": field,
        "operator": operator,
        "value": value,
        "evidence": evidence,
    }


def _validate_evidence(ir: RequestIR, source_text: str) -> None:
    source = source_text.casefold()

    def check(value: str) -> None:
        if value.strip().casefold() not in source:
            raise IRCompilationError(
                "source evidence is not present in the masked question", code="evidence",
                details=[f"evidence {value!r} is not an exact quote of the Question; copy the words exactly"],
            )  # fmt: skip

    def visit(node: IRNode | None) -> None:
        if node is None:
            return
        check(node.evidence)
        for child in node.children or []:
            visit(child)
        visit(node.child)

    check(ir.action_evidence)
    visit(ir.where)
    for item in [*ir.group_by, *ir.projection]:
        check(item.evidence)
    if ir.aggregate:
        check(ir.aggregate.evidence)
    if ir.sort:
        check(ir.sort.evidence)
    if ir.limit:
        check(ir.limit.evidence)
    if ir.lookup:
        check(ir.lookup.evidence)
    for item in [*ir.unsupported, *ir.unresolved]:
        check(item.evidence)
    for item in ir.ambiguous:
        check(item.evidence)


def _validate_node_status(node: IRNode | None) -> None:
    if node is None:
        return
    if node.status != "supported":
        raise _declared("the expression contains a non-executable semantic node", [(node.status, node.evidence)])
    if node.kind == "relationship":
        raise IRCompilationError(
            "relationship traversal is not supported by QueryPlan", code="capability", retryable=False,
            capability="questions that follow a relationship between records",
        )  # fmt: skip
    for child in node.children or []:
        _validate_node_status(child)
    _validate_node_status(node.child)


def _known_field(field: str, allowed: set[str], purpose: str) -> str:
    canonical = canonical_field(field)
    if canonical not in allowed:
        ordered = [f for f in (*CONDITION_FIELDS, *ATTRIBUTES, *SORT_FIELDS) if f in allowed]
        raise IRCompilationError(
            f"unsupported {purpose} field", code="field",
            details=[f"{purpose} field {field!r} is not allowed; allowed: {', '.join(dict.fromkeys(ordered))}"],
        )  # fmt: skip
    return canonical


def _compile_where(
    root: IRNode,
    allowed_statuses: list[str] | None,
    allowed_care_types: list[str] | None,
) -> dict:
    terms: list[IRNode] = []

    def flatten_and(node: IRNode) -> None:
        if node.kind == "and":
            for child in node.children or []:
                flatten_and(child)
        else:
            terms.append(node)

    flatten_and(root)
    per_field: dict[str, list[IRNode]] = {}
    for term in terms:
        fields = _term_fields(term)
        if len(fields) != 1:
            raise IRCompilationError(
                "cross-field OR or unsupported boolean scope", code="capability",
                details=["an OR/NOT node mixes different fields; only same-field OR is executable "
                         f"(fields: {', '.join(sorted(fields))}). If the question really needs it, list it in unsupported."],
                capability="combining conditions on different fields with OR",
            )  # fmt: skip
        field = _known_field(next(iter(fields)), _SET_FIELDS | _DATE_FIELDS | {"age"}, "filter")
        per_field.setdefault(field, []).append(term)

    filters: dict[str, dict] = {}
    for field, field_terms in per_field.items():
        condition = _compile_field_terms(field, field_terms)
        if condition is None:
            continue
        filters[field] = condition
        if field == "status" and allowed_statuses is not None:
            values = _filter_values(filters[field])
            if any(value not in allowed_statuses for value in values):
                raise IRCompilationError(
                    "status value is outside the live vocabulary", code="value",
                    details=[f"status values {values} must be among: {', '.join(allowed_statuses)}"],
                )  # fmt: skip
        if field == "care_type" and allowed_care_types is not None:
            values = _filter_values(filters[field])
            if any(value not in allowed_care_types for value in values):
                raise IRCompilationError(
                    "care type is outside the live vocabulary", code="value",
                    details=[f"care_type values {values} must be among: {', '.join(allowed_care_types)}"],
                )  # fmt: skip
    return filters


def _term_fields(node: IRNode) -> set[str]:
    if node.kind == "predicate":
        return {canonical_field(node.field or "")}
    if node.kind == "not":
        return _term_fields(node.child) if node.child else set()
    fields: set[str] = set()
    for child in node.children or []:
        fields |= _term_fields(child)
    return fields


def _compile_field_terms(field: str, terms: list[IRNode]) -> dict | None:
    if len(terms) == 1 and terms[0].kind == "or":
        return _compile_same_field_or(field, terms[0].children or [])
    terms = [_negated_predicate(term) if term.kind == "not" else term for term in terms]
    # An AND term that is an OR of ONE text operator on this field ("starts with A or B") is exactly a needle list.
    text_or: dict[str, list[str]] = {}
    if field in _TEXT_FIELDS:
        kept: list[IRNode] = []
        for term in terms:
            kids = term.children or []
            ops = {kid.operator for kid in kids}
            if (
                term.kind == "or" and len(ops) == 1 and ops <= set(TEXT_OPERATORS)
                and all(kid.kind == "predicate" and canonical_field(kid.field or "") == field for kid in kids)
            ):  # fmt: skip
                op = next(iter(ops))
                if op in text_or:
                    raise IRCompilationError(
                        f"several {op} conditions combined with AND", code="capability",
                        details=[f"{field}: two {op} groups under AND cannot be executed"],
                        capability=f"requiring several \"{op.replace('_', ' ')}\" texts at the same time",
                    )  # fmt: skip
                text_or[op] = list(dict.fromkeys(_text_value(kid) for kid in kids))
            else:
                kept.append(term)
        terms = kept
        if text_or and not terms:
            return text_or
    if any(term.kind != "predicate" for term in terms):
        raise IRCompilationError(
            "nested boolean expression is outside the executable subset", code="capability",
            details=[f"conditions on {field} nest AND/OR in a way QueryPlan cannot run; use one OR of predicates "
                     "or an AND of predicates per field"],
            capability="nested AND/OR combinations on one field",
        )  # fmt: skip

    operators = [term.operator for term in terms]
    _check_operators(field, operators)
    if field in _SET_FIELDS:
        if any(op in ("gt", "gte", "lt", "lte", "between") for op in operators):
            raise IRCompilationError("numeric comparison on a set field")
        if any(op == "missing" for op in operators) and (len(terms) > 1 or text_or):
            raise IRCompilationError("missingness combined with another predicate cannot be represented losslessly")
        out = _compile_set_and(terms)
        for op, needles in text_or.items():
            if op in out:
                raise IRCompilationError(
                    f"several {op} conditions combined with AND", code="capability",
                    details=[f"{field}: a {op} predicate and a {op} OR-group under AND cannot be executed"],
                    capability=f"requiring several \"{op.replace('_', ' ')}\" texts at the same time",
                )  # fmt: skip
            out[op] = needles
        if text_or and out.get("missing") is False:
            out.pop("missing")  # a text match implies a recorded value
        return out
    if field == "age" or field in _DATE_FIELDS:
        if any(op in ("eq", "ne", "in", "not_in") for op in operators):
            raise IRCompilationError("set comparison on a range field")
        if any(op == "missing" for op in operators) and len(terms) > 1:
            raise IRCompilationError("missingness combined with another predicate cannot be represented losslessly")
        return _compile_range_and(field, terms)
    raise IRCompilationError("unsupported filter field")


def _compile_same_field_or(field: str, terms: list[IRNode]) -> dict | None:
    if not terms or any(term.kind != "predicate" or canonical_field(term.field or "") != field for term in terms):
        raise IRCompilationError(
            "OR must contain supported predicates on one field", code="capability",
            details=[f"an OR on {field} must contain only predicates on {field}"],
            capability="nested AND/OR combinations on one field",
        )  # fmt: skip
    operators = [term.operator for term in terms]
    _check_operators(field, operators)
    if field in _TEXT_FIELDS and set(operators) & set(TEXT_OPERATORS):
        if len(set(operators)) == 1:  # "starts with A or B": OR of one text operator
            return {operators[0]: list(dict.fromkeys(_text_value(term) for term in terms))}
        raise IRCompilationError(
            "OR mixes a text match with other conditions", code="capability",
            details=[f"an OR on {field} mixes {', '.join(sorted(set(operators)))}; a text-match OR may only contain "
                     "one text operator"],
            capability="combining a text match with other values using OR",
        )  # fmt: skip
    if "missing" in operators and "present" in operators:
        return None
    if field in _SET_FIELDS and set(operators) <= {"missing", "present"}:
        return {"missing": operators[0] == "missing"}
    if field in _SET_FIELDS and "present" in operators and set(operators) <= {"eq", "in", "present"}:
        return {"missing": False}
    if field in _SET_FIELDS and all(op in ("eq", "in") for op in operators):
        values = list(dict.fromkeys(value for term in terms for value in _as_values(term)))
        return {"include": values}
    if field in _SET_FIELDS and all(op in ("eq", "in", "missing") for op in operators):
        values = list(
            dict.fromkeys(value for term in terms if term.operator != "missing" for value in _as_values(term))
        )
        result = {"missing": True}
        if values:
            result["include"] = values
        return result
    if field == "age" or field in _DATE_FIELDS:
        if set(operators) <= {"missing", "present"}:
            return {"missing": operators[0] == "missing"}
        range_terms = [term for term in terms if term.operator != "missing"]
        missing_terms = [term for term in terms if term.operator == "missing"]
        if len(range_terms) == 1 and len(missing_terms) == 1:
            return {**_compile_range_and(field, range_terms), "missing": True}
    raise IRCompilationError(
        "OR expression is not representable by QueryPlan", code="capability",
        details=[f"an OR on {field} with operators {', '.join(map(str, operators))} cannot be executed"],
        capability=f"this combination of OR conditions on {field.replace('_', ' ')}",
    )  # fmt: skip


def _negated_predicate(node: IRNode) -> IRNode:
    if node.kind != "not" or node.child is None or node.child.kind != "predicate":
        raise IRCompilationError("NOT is executable only for one supported predicate")
    inverse = {
        "eq": "ne",
        "ne": "eq",
        "in": "not_in",
        "not_in": "in",
        "gt": "lte",
        "gte": "lt",
        "lt": "gte",
        "lte": "gt",
        "missing": "present",
        "present": "missing",
    }
    operator = inverse.get(node.child.operator)
    if operator is None:
        text = node.child.operator in TEXT_OPERATORS
        raise IRCompilationError(
            "NOT operator is not supported", code="capability",
            details=[f"not({node.child.operator}) cannot be executed"],
            capability="excluding by a text match (\"does not start with\" / \"does not contain\")" if text
            else f"negating a {node.child.operator} condition",
        )  # fmt: skip
    return IRNode(
        kind="predicate",
        status="supported",
        field=node.child.field,
        operator=operator,
        value=node.child.value,
        evidence=node.evidence,
    )


def _compile_set_and(terms: list[IRNode]) -> dict:
    includes: list[set[str]] = []
    excludes: set[str] = set()
    missing: bool | None = None
    text: dict[str, list[str]] = {}
    for term in terms:
        op = term.operator
        if op in TEXT_OPERATORS:
            if op in text:  # "starts with A AND starts with B": the plan's needle list means OR, so it cannot carry this
                raise IRCompilationError(
                    f"several {op} conditions combined with AND", code="capability",
                    details=[f"{term.field}: two {op} predicates under AND cannot be executed; use one, or OR them"],
                    capability=f"requiring several \"{op.replace('_', ' ')}\" texts at the same time",
                )  # fmt: skip
            text[op] = [_text_value(term)]
            continue
        if op in ("eq", "in"):
            includes.append(set(_as_values(term)))
        elif op in ("ne", "not_in"):
            excludes.update(_as_values(term))
        elif op == "missing":
            if missing is False:
                raise IRCompilationError("contradictory missingness predicates")
            missing = True
        elif op == "present":
            if missing is True:
                raise IRCompilationError("contradictory missingness predicates")
            missing = False
        else:
            raise IRCompilationError("unsupported set-field comparison")
    include = set.intersection(*includes) if includes else None
    if include is not None:
        include -= excludes
        if not include:
            raise IRCompilationError("contradictory set predicates")
        if missing is False:
            missing = None
    if missing is True and (include is not None or excludes):
        raise IRCompilationError("conjunctive missingness is not representable")
    if text and missing is True:
        raise IRCompilationError(
            "a text match cannot hold for a value that is not recorded", code="contradiction",
            details=["missing AND starts_with/contains on one field can never match; remove one of them"],
        )  # fmt: skip
    if text and missing is False:
        missing = None  # a text match implies a recorded value
    out: dict = {}
    if include is not None:
        out["include"] = sorted(include)
    if excludes:
        out["exclude"] = sorted(excludes)
    if missing is not None:
        out["missing"] = missing
    out.update(text)
    if not out:
        raise IRCompilationError("empty set predicate")
    return out


def _compile_range_and(field: str, terms: list[IRNode]) -> dict:
    lower: int | date | None = None
    upper: int | date | None = None
    missing: bool | None = None
    is_date = field in _DATE_FIELDS
    for term in terms:
        op = term.operator
        if op == "missing":
            if missing is False:
                raise IRCompilationError("contradictory missingness predicates")
            missing = True
            continue
        if op == "present":
            if missing is True:
                raise IRCompilationError("contradictory missingness predicates")
            missing = False
            continue
        if op == "between":
            values = term.value
            assert isinstance(values, list) and len(values) == 2
            low, high = _range_value(values[0], is_date), _range_value(values[1], is_date)
            lower = _max_bound(lower, low)
            upper = _min_bound(upper, high)
            continue
        value = _range_value(term.value, is_date)
        if op == "gt":
            value = _shift_boundary(value, 1, is_date)
            lower = _max_bound(lower, value)
        elif op == "gte":
            lower = _max_bound(lower, value)
        elif op == "lt":
            value = _shift_boundary(value, -1, is_date)
            upper = _min_bound(upper, value)
        elif op == "lte":
            upper = _min_bound(upper, value)
        else:
            raise IRCompilationError("unsupported range comparison")
    if lower is not None and upper is not None and lower > upper:
        raise IRCompilationError("contradictory range predicates")
    if missing is False and (lower is not None or upper is not None):
        missing = None
    out: dict = {}
    if is_date:
        if lower is not None:
            out["start"] = lower.isoformat()
        if upper is not None:
            out["end"] = upper.isoformat()
    else:
        if lower is not None:
            out["min"] = lower
        if upper is not None:
            out["max"] = upper
    if missing is not None:
        out["missing"] = missing
    if not out:
        raise IRCompilationError("empty range predicate")
    return out


def _range_value(value: Scalar | list[Scalar] | None, is_date: bool):
    if is_date:
        if not isinstance(value, str):
            raise IRCompilationError("date comparison needs an ISO date")
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise IRCompilationError("date comparison needs an ISO date") from exc
    if not isinstance(value, int) or isinstance(value, bool):
        raise IRCompilationError("age comparison needs an integer")
    return value


def _max_bound(a, b):
    return b if a is None or b > a else a


def _min_bound(a, b):
    return b if a is None or b < a else a


def _shift_boundary(value, days: int, is_date: bool):
    try:
        return value + timedelta(days=days) if is_date else value + days
    except OverflowError as exc:
        raise IRCompilationError("comparison boundary is outside the supported range") from exc


def _declared(message: str, flagged: list[tuple[str, str]]) -> IRCompilationError:
    """The model marked part of the request ambiguous/unsupported/unresolved: not retried, reported to the user."""
    kind = "ambiguous" if all(status == "ambiguous" for status, _ in flagged) else "unsupported"
    return IRCompilationError(
        message, code=f"declared_{kind}", retryable=False,
        details=[f"{status}: {evidence!r}" for status, evidence in flagged], quotes=[e for _, e in flagged],
    )  # fmt: skip


def _check_operators(field: str, operators: list) -> None:
    allowed = operators_for(field)
    for op in operators:
        if op in allowed:
            continue
        text = op in TEXT_OPERATORS
        raise IRCompilationError(
            f"operator {op} is not allowed on {field}", code="operator",
            details=[f"{field}: operator {op!r} is not allowed; allowed on {field}: {', '.join(allowed)}"
                     + (f". Text matching works only on {', '.join(TEXT_MATCH_FIELDS)}" if text else "")],
            capability=f"text matching on {field.replace('_', ' ')}" if text else None,
        )  # fmt: skip


def _text_value(node: IRNode) -> str:
    """The literal text of a starts_with/contains predicate. An integer (room 1) is its decimal text."""
    value = node.value
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        raise IRCompilationError(
            "text match needs a non-empty text value", code="value",
            details=[f"{node.field}: {node.operator} needs one non-empty text value (got: {node.value!r})"],
        )  # fmt: skip
    return value.strip()


def _as_values(node: IRNode) -> list[str]:
    value = node.value
    values = value if isinstance(value, list) else [value]
    if not all(isinstance(item, str) for item in values):
        raise IRCompilationError("set values must be strings")
    return list(values)


def _filter_values(condition: dict) -> list[str]:
    return list(condition.get("include") or []) + list(condition.get("exclude") or [])
