"""Prompt, few-shot examples and constrained-output schema for question → RequestIR interpretation.

Application-level definitions (actions, which fields exist, their operators, security limits) are code.
Database-specific values (statuses, care types, facilities and caregivers) come from the live vocabulary as a
SchemaContext. The IR schema constrains structure; backend compilation and entity resolution enforce actual values.

The prompt is short and example-led because the model is small. It is rebuilt only when the vocabulary
changes, so llama.cpp can keep the system prompt + examples in its prompt (KV) cache between requests.
"""

import json

from app.models.plan import canonical_field
from app.nlu.request_ir import ATTRIBUTES as ATTRIBUTES_IR
from app.nlu.request_ir import CONDITION_FIELDS as CONDITION_FIELDS_IR
from app.nlu.request_ir import GROUP_FIELDS as GROUP_FIELDS_IR
from app.nlu.request_ir import RANGE_FIELDS as RANGE_FIELDS_IR
from app.nlu.request_ir import SET_FIELDS as SET_FIELDS_IR
from app.nlu.request_ir import SORT_FIELDS as SORT_FIELDS_IR
from app.nlu.request_ir import TEXT_OPERATORS, RequestIR, operators_for, request_ir_from_plan
from app.nlu.translator import SchemaContext

# Field lists come from the compiler (app/nlu/request_ir.py), so the prompt and schema never drift from what it accepts.
SET_FIELDS = list(SET_FIELDS_IR)
RANGE_FIELDS = list(RANGE_FIELDS_IR)
CONDITION_FIELDS = list(CONDITION_FIELDS_IR)
GROUP_FIELDS = list(GROUP_FIELDS_IR)
SORT_FIELDS = list(SORT_FIELDS_IR)
# Longest evidence quote in a few-shot EXAMPLE. Short example quotes teach short evidence (copying the whole question
# into every node made a 4-condition plan exceed 256 tokens, 2026-11-17). This is deliberately NOT enforced in the
# output grammar: a grammar cap forced the quote closed mid-phrase and both Qwen3-4B and Phi-4-mini then closed the
# AND node with no conditions (Q31/Q32/Q36 benchmark, 2026-11-17). The RequestIR model's own max_length still applies.
EVIDENCE_MAX_CHARS = 60

# llama.cpp (b12017 common/json-schema-to-grammar.cpp, _build_object_rule) emits an object's REQUIRED properties
# first, then the optional ones, whatever order the schema lists them in. Examples must use that same order: an
# example with "group_by" before "unsupported" teaches a key the grammar forbids at that point, and once the model has
# written the last required key ("ambiguous") every example it has seen ends there — so group_by, projection,
# aggregate, sort and limit became unreachable (benchmark 2026-11-17: 0 of 20 first attempts wrote either).
_IR_SCHEMA = RequestIR.model_json_schema()
_IR_REQUIRED = [k for k in _IR_SCHEMA["properties"] if k in _IR_SCHEMA.get("required", [])]
GRAMMAR_KEY_ORDER = _IR_REQUIRED + [k for k in _IR_SCHEMA["properties"] if k not in _IR_REQUIRED]
ATTRIBUTES = list(ATTRIBUTES_IR)
# staff fields are described in their own section (Bedrock prompt only); the resident tables stay as they were
_STAFF_PREFIX = "staff_"
RESIDENT_CONDITION_FIELDS = [n for n in CONDITION_FIELDS if not n.startswith(_STAFF_PREFIX)]


def _family_rules(ctx: SchemaContext) -> str:
    """The family relation (familymembers), for models without an output grammar. Field names come from the compiler;
    the relation words are the live database's labels (son, daughter, ...), never a person's data."""
    relations = " | ".join(ctx.family_relations) if ctx.family_relations else "son | daughter | spouse | ..."
    return (
        "\n- Family (the resident's family members; a condition holds when ANY of them matches):"
        f"\n  family_relation: {relations}. eq/in = has such a family member; ne/not_in = has none;"
        " a word like spouse, children or parents may be used as written."
        "\n  family_type: family | emergency · family_flags: portal_access | app_access | health_access | guarantor |"
        " has_email | has_phone."
        ' "a family member with portal access" is family_flags eq "portal_access"; "a family member whose email is'
        ' registered / on file" is family_flags eq "has_email" (never missing/present with a value).'
        "\n  family_count: number of family members (0 = none recorded); sort by family_count for most / fewest."
        "\n  To SHOW family: projection family_members (names + relations; for ONE named resident also phone and"
        " email) or family_count. A family question about one named resident is a lookup with attribute"
        " family_members. Never filter by a family member's own name."
        + (
            "\n- People: resident_N = a resident; family_N = a family member; person_N = a name that may be a staff member"
            " or a family member (the backend decides). Use family_N / person_N ONLY as a family_member value (eq/in),"
            " or person_N as a primary_caregiver value when the question is about a caregiver; never in lookup.residents."
            "\n  How family_N is related to resident_N: lookup resident_N, attributes [\"family_relationship\"], where"
            " family_member eq family_N. Which resident family_N is related to: list where family_member eq family_N."
            " \"resident_N's daughters\": lookup resident_N, attributes [\"family_members\"], where family_relation eq daughter."
            if ctx.family_names else ""
        )
    )


def _staff_rules() -> str:
    """Staff (the staffs collection; app/data/staff.py), for models without an output grammar. Field names come from
    the compiler; role words are mapped to roles by the backend (resident_schema.yaml), never by the model."""
    from app.models.plan import STAFF_ATTRIBUTES, STAFF_GROUP_FIELDS, STAFF_SINGLE_RECORD, STAFF_SORT_FIELDS

    staff_fields = [n for n in CONDITION_FIELDS if n.startswith(_STAFF_PREFIX)]
    ops: dict[tuple[str, ...], list[str]] = {}
    for name in staff_fields:
        ops.setdefault(operators_for(name), []).append(name)
    shown = [a for a in STAFF_ATTRIBUTES if a not in STAFF_SINGLE_RECORD]
    return (
        "\n- Staff members (care team, physicians, case managers, nurses, therapists, drivers, ...). Staff questions are"
        " in scope: never answer unsupported because no resident_N is named. person_N is a named person (the Analysis"
        ' says which kinds it can be): a staff member → "lookup": {"staff": ["person_N"], ...} with entity staff (or'
        " staff_name / primary_caregiver person_N); a resident → lookup.residents [\"person_N\"]; never drop it."
        "\n  staff_role: the role word as written (physician, doctor, case manager, nurse practitioner, social worker,"
        " dietitian, therapist, ...); staff_designation: a designation exactly as written. In a RESIDENT question they"
        " mean ANY of the resident's active assigned staff: \"residents without a case manager\" is staff_role ne"
        " \"case manager\"; staff_count = number of active assigned staff. \"caregiver\" stays primary_caregiver."
        "\n  Show a resident's staff: care_team (assigned staff with designations; inactive marked) or primary_physician"
        " (only for the words \"primary physician\"). \"resident_N's physician\" = lookup resident_N, attributes"
        " [\"care_team\"], where staff_role eq \"physician\"; \"resident_N's care team\" = attributes [\"care_team\"]."
        "\n  When the rows are STAFF MEMBERS (\"how many physicians\", \"list case managers\", \"staff with more than 20"
        ' residents"), add "entity": "staff" right after "action". Staff fields: '
        + "; ".join(f"{', '.join(names)}: {' | '.join(o)}" for o, names in ops.items())
        + ". staff_status: active | inactive (omit = active staff only; \"including inactive\" = in [active, inactive]);"
        " staff_type: provider | facility_staff | external_user; staff_name: a staff name as written;"
        " staff_resident_count: number of assigned residents."
        "\n  In a staff question every resident field is a condition on the staff member's assigned residents"
        ' ("staff caring for memory care residents" = entity staff, care_type eq memory_care).'
        f"\n  Staff group_by: {', '.join(STAFF_GROUP_FIELDS)} · sort: {', '.join(STAFF_SORT_FIELDS)} · projection:"
        f" {', '.join(shown)}."
        '\n  One named staff member: action lookup, entity staff, "lookup": {"staff": ["name as written"], "attributes":'
        f" [...], ...}}; {', '.join(STAFF_SINGLE_RECORD)} only in such a lookup (staff_phone includes the country code)."
        ' "residents assigned to <staff name>" is a RESIDENT list with primary_caregiver eq the name. "how many residents'
        ' does each physician have" = entity staff, list, staff_role eq physician, projection staff_resident_count, sort'
        " staff_resident_count desc."
    )


def _operator_rules(full: bool) -> str:
    """Predicate operators, generated from the compiler's definitions (request_ir.operators_for).

    full=False (grammar-constrained models, e.g. Qwen on llama.cpp): the output grammar already allows only the
    operators in PredicateOperator, so one compact line says which fields take text matching (keeps the prompt within
    its CPU size budget). full=True (no output grammar, e.g. Bedrock): the whole per-field table."""
    text_fields = [n for n in RESIDENT_CONDITION_FIELDS if set(TEXT_OPERATORS) <= set(operators_for(n))]
    if not full:
        return f"- {'/'.join(TEXT_OPERATORS)}: only {', '.join(text_fields)}"
    groups: dict[tuple[str, ...], list[str]] = {}
    for name in RESIDENT_CONDITION_FIELDS:
        groups.setdefault(operators_for(name), []).append(name)
    lines = ["- Predicate operators — use ONLY these, per field:"]
    lines += [f"  {', '.join(names)}: {' | '.join(ops)}" for ops, names in groups.items()]
    lines.append(
        f"  {' / '.join(TEXT_OPERATORS)}: case-insensitive match on literal text; value is the text only (no %, * or "
        'regex), e.g. "room starts with B" is room_number starts_with "B". Anything needing another operator goes '
        "in unsupported."
    )
    return "\n".join(lines)


def build_system_prompt(ctx: SchemaContext, operator_table: bool = False) -> str:
    statuses = " | ".join(ctx.statuses)
    care = " | ".join(ctx.care_types)
    return f"""Interpret care-home questions as one compositional RequestIR JSON object. Never answer or invent values;
data determines counts.

Fields:
- status: {statuses}
- care_type: {care}
- gender: Female | Male | Other
- facility: the resident's facility (names are listed in the request)
- primary_caregiver: staff name (nurse, aide, caretaker) exactly as written in the question
- room_number, age (years)
- admission_date, discharge_date, created_date (record registered)

Actions:
- count: how many residents match.
- list: which residents; with sort + limit for oldest, youngest, latest, top N.
- breakdown: counts per field (group_by), e.g. "by status", "per facility", "each care type". If the question also
  asks to show residents in each group, set "projection" for those rows just as for a list; resident identity is included.
- aggregate: average/min/max age.
- lookup: facts about a resident named in the current question. Use only a resident_N placeholder present in Question;
  never invent resident placeholders or look up a resident when none is named.
- clarify: the question is missing something essential; ask one short question in message.
- unsupported: needs data not listed above (diagnosis, medication, diet, billing, visitors ...) or is not about \
residents; give a short reason in message.
- help: greetings, thanks, what can you do.

RequestIR fields:
- Evidence is the shortest exact phrase from Question that supports that part (a few words, e.g. "over 85"), never
  the whole question.
- action/action_status/action_evidence: operation, its status, and exact supporting quote.
- where: predicate/and/or/not/relationship AST. Every node has status (supported, ambiguous, unsupported, unresolved)
  and exact evidence. Predicates use field/operator/value. Relationship nodes use relation/target and are rejected.
{_operator_rules(operator_table)}{_family_rules(ctx) if operator_table and ctx.family_rules else ''}{_staff_rules() if operator_table and ctx.staff_rules else ''}
- group_by and projection: arrays of {{field, status, evidence}}; grouping is not filtering. Projection is what to SHOW,
  separate from where (for example "show room numbers" is projection, not a room filter).
- aggregate, sort, limit, lookup: each has status and exact evidence.
- A lookup MUST include "lookup": {{"residents": ["resident_N"], "attributes": [...], "status": ..., "evidence": ...}}
  after "ambiguous". attributes = exactly what is asked: "which room is resident_N in" → ["room_number"],
  "who looks after resident_N" → ["primary_caregiver"]; an empty list means the whole profile.
- ambiguous/unsupported/unresolved: list uncertain or unrepresentable material with evidence; any non-empty list
  blocks execution. Status/completeness claims are diagnostic; backend allowlists authorize compilation.

Rules:
- Preserve all conditions and their exact boolean scope. "Active or Discharged" is OR; "Active and Discharged" is AND.
  Do not convert one into the other. Same-field OR values may use an OR node or one in predicate.
- Strict and inclusive comparisons differ: "after X" is gt; "on or after X" is gte; "before X" is lt;
  "on or before X" is lte. Copy normalized dates exactly from the Dates line. Admitted/joined → admission_date;
  discharged/left → discharge_date plus status discharged; registered/added → created_date.
- Negation is explicit: "not Active" is not(eq(status, active)); never silently turn it into inclusion.
- The result's name/identity is implicit. "show their room numbers" adds room_number to projection, including for
  each resident in a breakdown.
- If the question asks separate counts for multiple values of one groupable field, use breakdown and group_by.
- Age is an integer in years: "over 80" is gt 80; "80 or older" is gte 80.
- If Question contains a resident_N placeholder, copy it unchanged into lookup.residents; never create one.
- An "Analysis" block is only a hint; the original Question is authoritative.
- For a follow-up, use the previous RequestIR only as context and set uses_previous true; preserve earlier semantics
  unless the question changes them. On retry, repair only issues in Check, preserve valid components, and honor the
  original Question.
- Source evidence proves only that a quote occurs in Question, not that its meaning was interpreted correctly.
- Output only the RequestIR JSON object."""


def _examples(ctx: SchemaContext, today: str) -> list[tuple[str, dict, dict[str, str]]]:
    """Few-shot pairs built from real vocabulary values, so no example names a value the database lacks.

    Each example carries short evidence quotes (see _short_evidence): "action", "where" (a node combining different
    fields), "aggregate", and one per field; the examples teach the model what evidence should look like.
    """
    st = ctx.statuses
    ct = ctx.care_types
    ex: list[tuple[str, dict, dict[str, str]]] = []

    def req(q: str, dates: list[str] | None = None, wings: bool = False) -> str:
        return build_user_message(q, today, [], None, dates or [])

    if "active" in st and "away" in st:
        ex.append(
            (
                req(
                    "List residents who are either Active or Away and were admitted after January 1, 2025.",
                    ["'January 1, 2025' = 2025-01-01"],
                ),
                {
                    "action": "list",
                    "filters": {
                        "status": {"include": ["active", "away"]},
                        "admission_date": {"start": "2025-01-02"},
                    },
                },
                {
                    "action": "List residents",
                    "where": "Active or Away and were admitted",
                    "status": "Active or Away",
                    "admission_date": "admitted after January 1, 2025",
                },
            )
        )
    if "active" in st and "memory_care" in ct:
        ex.append(
            (
                req("Which residents are not Active and over 85?"),
                {
                    "action": "list",
                    "filters": {"status": {"exclude": ["active"]}, "age": {"min": 86}},
                },
                {"action": "Which residents", "where": "not Active and over 85", "status": "not Active", "age": "over 85"},
            )
        )
    if "memory_care" in ct:
        ex.append(
            (
                req(
                    "Count September 2026 admissions per care type; show names and rooms.",
                    ["'September 2026' = 2026-09-01 to 2026-09-30"],
                ),
                {
                    "action": "breakdown",
                    "filters": {"admission_date": {"start": "2026-09-01", "end": "2026-09-30"}},
                    "group_by": ["care_type"],
                    "fields": ["room_number"],
                },
                {
                    "action": "Count",
                    "admission_date": "September 2026 admissions",
                    "care_type": "per care type",
                    "room_number": "rooms",
                },
            )
        )
    ex.extend(
        [
            (
                req("Average age per care type."),
                {
                    "action": "aggregate",
                    "aggregate": {"op": "avg", "field": "age"},
                    "group_by": ["care_type"],
                },
                {"action": "Average age", "aggregate": "Average age", "care_type": "per care type"},
            ),
            (
                req("Diabetes?"),
                {"action": "unsupported", "message": "Diagnosis data is not recorded."},
                {"action": "Diabetes?"},
            ),
        ]
    )
    return ex


def _short_evidence(ir: dict, quotes: dict[str, str], question: str) -> dict:
    """Replace the whole-question evidence request_ir_from_plan writes with each part's short quote (in place).

    Predicates, projection and group_by items use the quote for their field; a boolean node whose predicates all share
    one field uses that field's quote, otherwise quotes["where"]. A missing or non-verbatim quote raises at prompt build,
    so a broken example can never reach the model.
    """

    def quote(key: str) -> str:
        text = quotes[key]
        if text.casefold() not in question.casefold() or len(text) > EVIDENCE_MAX_CHARS:
            raise ValueError(f"example evidence {text!r} is not a short quote from {question!r}")
        return text

    def fields(node: dict) -> set[str]:
        if node.get("kind") == "predicate":
            return {node["field"]}
        kids = [*(node.get("children") or []), *([node["child"]] if node.get("child") else [])]
        return set().union(*(fields(k) for k in kids)) if kids else set()

    def visit(node: dict | None) -> None:
        if not node:
            return
        names = fields(node)
        node["evidence"] = quote(next(iter(names)) if len(names) == 1 else "where")
        for kid in [*(node.get("children") or []), *([node["child"]] if node.get("child") else [])]:
            visit(kid)

    ir["action_evidence"] = quote("action")
    visit(ir.get("where"))
    for item in [*(ir.get("group_by") or []), *(ir.get("projection") or [])]:
        item["evidence"] = quote(item["field"])
    for key in ("aggregate", "sort", "limit", "lookup"):
        if ir.get(key):
            ir[key]["evidence"] = quote(key)
    return ir


def build_messages(
    ctx: SchemaContext, user_message: str, today_for_examples: str = "2026-11-09", operator_table: bool = False
) -> list[dict]:
    """Chat messages: static system prompt + few-shot turns (cacheable prefix), then the real request.
    operator_table: spell out every allowed operator per field (for models without a constrained output grammar)."""
    messages: list[dict] = [{"role": "system", "content": build_system_prompt(ctx, operator_table)}]
    for q, plan, quotes in _examples(ctx, today_for_examples):
        messages.append({"role": "user", "content": q})
        example = dict(plan)
        if isinstance(example.get("conditions"), list):
            example["filters"] = conditions_to_filters(example.pop("conditions"))
        question = q.split("Question: ", 1)[-1]
        raw = _short_evidence(request_ir_from_plan(example, question), quotes, question)
        ir = RequestIR.model_validate(raw).model_dump(mode="json", exclude_defaults=True)
        ir = {key: ir[key] for key in GRAMMAR_KEY_ORDER if key in ir}  # the order the output grammar allows
        messages.append({"role": "assistant", "content": json.dumps(ir, separators=(",", ":"))})
    messages.append({"role": "user", "content": user_message})
    return messages


def build_user_message(
    masked_question: str,
    today: str,
    wings: list[str],
    previous_ir_json: str | None,
    date_hints: list[str] | None = None,
    hint: str | None = None,
    brief: str | None = None,
    retry_ir_json: str | None = None,
) -> str:
    parts = [f"Today: {today}"]
    if wings:
        parts.append(f"Wings: {', '.join(wings)}")
    if date_hints:
        parts.append(f"Dates: {'; '.join(date_hints)}")
    if previous_ir_json:
        parts.append(f"Previous RequestIR (use only if the question refers back to it): {previous_ir_json}")
    if retry_ir_json:
        parts.append(f"Retry candidate RequestIR (same question): {retry_ir_json}")
    parts.append(f"Question: {masked_question}")
    if brief:
        parts.append(brief)
    if hint:
        parts.append(f"Check: {hint}")
    return "\n".join(parts)


def build_output_schema(ctx: SchemaContext) -> dict:
    """Schema-constrained RequestIR; backend compilation remains the execution boundary."""
    schema = RequestIR.model_json_schema()
    defs = schema["$defs"]
    _enum_schema(defs["IRNode"]["properties"]["field"], CONDITION_FIELDS)
    _enum_schema(defs["IRField"]["properties"]["field"], GROUP_FIELDS + ATTRIBUTES)
    _enum_schema(defs["IRSort"]["properties"]["field"], SORT_FIELDS)
    _enum_schema(defs["IRLookup"]["properties"]["attributes"]["items"], ATTRIBUTES)
    _enum_schema(defs["IRAggregate"]["properties"]["field"], ["age"])
    return schema


def _enum_schema(schema: dict, values: list[str]) -> None:
    if "anyOf" in schema:
        target = next((item for item in schema["anyOf"] if item.get("type") == "string"), None)
        if target is not None:
            target.clear()
            target["enum"] = values
    else:
        schema.clear()
        schema["enum"] = values


def conditions_to_filters(conditions: list) -> dict:
    """Model output (list of conditions) → QueryPlan.filters. Unknown shapes are kept so validation rejects them."""
    filters: dict = {}
    for cond in conditions:
        if not isinstance(cond, dict) or "field" not in cond:
            filters.setdefault("_invalid", []).append(cond)
            continue
        field = canonical_field(cond["field"])  # "wing" (legacy) → "facility"
        body = {k: v for k, v in cond.items() if k != "field"}
        if field in SET_FIELDS:
            entry = filters.setdefault(field, {})
            for key in ("include", "exclude"):
                if body.get(key):
                    entry[key] = list(dict.fromkeys([*entry.get(key, []), *body.pop(key)]))
                else:
                    body.pop(key, None)
            entry.update(body)  # anything else is invalid for a set field and fails validation
        elif field in RANGE_FIELDS:
            filters.setdefault(field, {}).update(body)
        else:
            filters.setdefault("_invalid", []).append(cond)
    return filters


def filters_to_conditions(filters: dict) -> list:
    """QueryPlan.filters → the model's list format (used to show the previous plan to the model)."""
    return [{"field": field, **body} for field, body in filters.items()]
