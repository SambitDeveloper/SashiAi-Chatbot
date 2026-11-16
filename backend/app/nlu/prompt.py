"""Prompt, few-shot examples and constrained-output schema for question → query-plan translation.

Application-level definitions (actions, which fields exist, their operators, security limits) are code.
Database-specific knowledge (which statuses and care types exist, wing/facility names, caregiver names) is NOT:
it comes from the live vocabulary as a SchemaContext, so the prompt and the JSON grammar always match the data and
the model cannot emit a status, care type or wing that the database does not contain.

The prompt is short and example-led because the model is small (0.5B). It is rebuilt only when the vocabulary
changes, so llama.cpp can keep the system prompt + examples in its prompt (KV) cache between requests.
"""

import json

from app.models.plan import canonical_field
from app.nlu.translator import SchemaContext

SET_FIELDS = ["status", "care_type", "facility", "gender", "primary_caregiver", "room_number"]
RANGE_FIELDS = {
    "age": ("min", "max"),
    "admission_date": ("start", "end"),
    "created_date": ("start", "end"),
    "discharge_date": ("start", "end"),
}
CONDITION_FIELDS = SET_FIELDS + list(RANGE_FIELDS)
GROUP_FIELDS = ["status", "care_type", "facility", "gender", "primary_caregiver"]
SORT_FIELDS = ["admission_date", "created_date", "discharge_date", "age", "last_name"]
ATTRIBUTES = [
    "primary_caregiver",
    "room_number",
    "facility",
    "admission_date",
    "created_date",
    "discharge_date",
    "emergency_contact",
    "resident_id",
    "status",
    "care_type",
    "age",
    "gender",
]
ACTIONS = ["count", "list", "breakdown", "aggregate", "lookup", "clarify", "unsupported", "help"]
GENDERS = ["Female", "Male", "Other"]

# Grammar-size guard: beyond this many wings the wing value is a free string resolved server-side.
MAX_ENUM_WINGS = 150


def build_system_prompt(ctx: SchemaContext) -> str:
    statuses = " | ".join(ctx.statuses)
    care = " | ".join(ctx.care_types)
    return f"""You convert a care-home staff question into ONE JSON query plan. Never answer the question. \
Never invent values. The database computes every number.

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
- breakdown: counts per field (group_by), e.g. "by status", "per facility", "each care type".
- aggregate: average/min/max age.
- lookup: facts about a named resident (resident_1, resident_2 ...).
- clarify: the question is missing something essential; ask one short question in message.
- unsupported: needs data not listed above (diagnosis, medication, diet, billing, visitors ...) or is not about \
residents; give a short reason in message.
- help: greetings, thanks, what can you do.

Rules:
- One condition per field mentioned in the question. Add nothing the question does not say.
- "not", "except", "excluding" go in exclude. Several values of one field go together in include.
- What to SHOW goes in "fields" (list/lookup), separate from what to filter by: "show their room numbers" → \
"fields": ["room_number"]. One attribute can be both ("in room 101 … show their rooms").
- No value recorded ("no caregiver", "without a room", "admission date missing", "not assigned") → \
{{"field": F, "missing": true}}. "has a caregiver assigned" → "missing": false.
- Copy dates exactly from the "Dates" line of the request. Admitted/joined → admission_date. \
Discharged/left → discharge_date plus status discharged. Registered/added → created_date.
- age uses min/max: "over 80" → min 81, "80 or older" → min 80, "under 70" → max 69.
- Copy resident_N placeholders unchanged into lookup.residents.
- If the question refers to the previous plan ("them", "those", "of these", "what about"), start from its \
conditions, apply the change and set uses_previous true.
- Output only the JSON plan."""


def _examples(ctx: SchemaContext, today: str) -> list[tuple[str, dict]]:
    """Few-shot pairs built from real vocabulary values, so no example names a value the database lacks."""
    has = lambda xs, v: v in xs  # noqa: E731
    st = ctx.statuses
    ct = ctx.care_types
    # Fixed example values (not this question's wings), so the examples are identical on every request and stay in
    # llama.cpp's prompt cache.
    wing = ctx.example_wing or (ctx.wings[0] if ctx.wings else None)
    cg = _short_caregiver(ctx.caregivers[0]) if ctx.caregivers else None
    ex: list[tuple[str, dict]] = []

    def req(q: str, dates: list[str] | None = None, wings: bool = False) -> str:
        return build_user_message(q, today, [wing] if wings and wing else [], None, dates or [])

    if has(st, "active") and has(ct, "memory_care"):
        ex.append(
            (
                req("How many active female residents are in memory care?"),
                {
                    "action": "count",
                    "conditions": [
                        {"field": "status", "include": ["active"]},
                        {"field": "gender", "include": ["Female"]},
                        {"field": "care_type", "include": ["memory_care"]},
                    ],
                },
            )
        )
    if has(st, "active"):
        ex.append(
            (
                req("which residents are not active and over 85"),
                {
                    "action": "list",
                    "conditions": [{"field": "status", "exclude": ["active"]}, {"field": "age", "min": 86}],
                },
            )
        )
    if wing and cg:
        ex.append(
            (
                req(
                    f"residents of {cg} in {wing} admitted last month",
                    ["'last month' = 2026-10-01 to 2026-10-31"],
                    True,
                ),
                {
                    "action": "list",
                    "conditions": [
                        {"field": "primary_caregiver", "include": [cg]},
                        {"field": "facility", "include": [wing]},
                        {"field": "admission_date", "start": "2026-10-01", "end": "2026-10-31"},
                    ],
                },
            )
        )
    if has(ct, "skilled_nursing") and has(ct, "assisted_living"):
        ex.append(
            (
                req("breakdown by gender for skilled nursing or assisted living"),
                {
                    "action": "breakdown",
                    "group_by": ["gender"],
                    "conditions": [{"field": "care_type", "include": ["skilled_nursing", "assisted_living"]}],
                },
            )
        )
    if has(st, "discharged"):
        ex.append(
            (
                req("who was discharged since March", ["'since March' = from 2026-03-01 (no end)"]),
                {
                    "action": "list",
                    "conditions": [
                        {"field": "status", "include": ["discharged"]},
                        {"field": "discharge_date", "start": "2026-03-01"},
                    ],
                },
            )
        )
    ex.append(
        (
            req("the 3 youngest residents"),
            {"action": "list", "sort": {"field": "age", "direction": "asc"}, "limit": 3},
        )
    )
    ex.append(
        (
            req("average age per care type"),
            {"action": "aggregate", "aggregate": {"op": "avg", "field": "age"}, "group_by": ["care_type"]},
        )
    )
    ex.append(
        (
            req("what is resident_1's room and caregiver"),
            {
                "action": "lookup",
                "lookup": {"residents": ["resident_1"], "attributes": ["room_number", "primary_caregiver"]},
            },
        )
    )
    if has(st, "active") and has(ct, "memory_care"):  # only values this database has
        ex.append(
            (
                req("list active residents in memory care and show their rooms and caregivers"),
                {
                    "action": "list",
                    "conditions": [
                        {"field": "status", "include": ["active"]},
                        {"field": "care_type", "include": ["memory_care"]},
                    ],
                    "fields": ["room_number", "primary_caregiver"],
                },
            )
        )
    ex.append(
        (
            req("how many residents have no caregiver assigned"),
            {"action": "count", "conditions": [{"field": "primary_caregiver", "missing": True}]},
        )
    )
    ex.append(
        (
            req("which residents have diabetes"),
            {"action": "unsupported", "message": "Diagnoses are not available in the resident records."},
        )
    )
    return ex


def _short_caregiver(value: str) -> str:
    """'Nurse Sarah Jenkins, RN' → 'Sarah Jenkins' (how staff are usually named in questions)."""
    name = value.split(",")[0].split()
    titles = {"nurse", "head", "care", "aide", "coordinator", "wellness", "senior", "charge", "dr", "dr."}
    while len(name) > 2 and name[0].lower() in titles:
        name = name[1:]
    return " ".join(name)


def build_messages(ctx: SchemaContext, user_message: str, today_for_examples: str = "2026-11-09") -> list[dict]:
    """Chat messages: static system prompt + few-shot turns (cacheable prefix), then the real request."""
    messages: list[dict] = [{"role": "system", "content": build_system_prompt(ctx)}]
    for q, plan in _examples(ctx, today_for_examples):
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": json.dumps(plan, separators=(",", ":"))})
    messages.append({"role": "user", "content": user_message})
    return messages


def build_user_message(
    masked_question: str,
    today: str,
    wings: list[str],
    previous_plan_json: str | None,
    date_hints: list[str] | None = None,
    hint: str | None = None,
) -> str:
    parts = [f"Today: {today}"]
    if wings:
        parts.append(f"Wings: {', '.join(wings)}")
    if date_hints:
        parts.append(f"Dates: {'; '.join(date_hints)}")
    if previous_plan_json:
        parts.append(f"Previous plan (use only if the question refers back to it): {previous_plan_json}")
    parts.append(f"Question: {masked_question}")
    if hint:
        parts.append(f"Check: {hint}")
    return "\n".join(parts)


def _set_condition(field: str, values: dict | None) -> dict:
    item = {"type": "array", "items": values or {"type": "string", "maxLength": 80}, "minItems": 1, "maxItems": 6}
    return {
        "type": "object",
        "properties": {"field": {"const": field}, "include": item, "exclude": item, "missing": {"type": "boolean"}},
        "required": ["field"],
        "additionalProperties": False,
    }


def build_output_schema(ctx: SchemaContext) -> dict:
    """JSON Schema for grammar-constrained generation (llama.cpp converts it to a GBNF grammar).

    Value enums come from the live vocabulary. Valid JSON is still not proof of a correct plan: the resolver,
    Pydantic models and grounding checks validate every candidate afterwards.
    """
    wing_values = {"enum": ctx.wings} if 0 < len(ctx.wings) <= MAX_ENUM_WINGS else None
    date_cond = {
        "type": "object",
        "properties": {
            "field": {"enum": ["admission_date", "created_date", "discharge_date"]},
            "start": {"type": "string", "format": "date"},
            "end": {"type": "string", "format": "date"},
            "missing": {"type": "boolean"},
        },
        "required": ["field"],
        "additionalProperties": False,
    }
    age_cond = {
        "type": "object",
        "properties": {
            "field": {"const": "age"},
            "min": {"type": "integer", "minimum": 0, "maximum": 130},
            "max": {"type": "integer", "minimum": 0, "maximum": 130},
            "missing": {"type": "boolean"},
        },
        "required": ["field"],
        "additionalProperties": False,
    }
    conditions = {
        "type": "array",
        "maxItems": 8,
        "items": {
            "anyOf": [
                _set_condition("status", {"enum": ctx.statuses}),
                _set_condition("care_type", {"enum": ctx.care_types}),
                _set_condition("gender", {"enum": GENDERS}),
                _set_condition("facility", wing_values),
                _set_condition("primary_caregiver", None),
                _set_condition("room_number", None),
                age_cond,
                date_cond,
            ]
        },
    }
    return {
        "type": "object",
        "properties": {
            "action": {"enum": ACTIONS},
            "conditions": conditions,
            "group_by": {"type": "array", "items": {"enum": GROUP_FIELDS}, "minItems": 1, "maxItems": 2},
            "aggregate": {
                "type": "object",
                "properties": {"op": {"enum": ["avg", "min", "max"]}, "field": {"const": "age"}},
                "required": ["op"],
                "additionalProperties": False,
            },
            "sort": {
                "type": "object",
                "properties": {"field": {"enum": SORT_FIELDS}, "direction": {"enum": ["asc", "desc"]}},
                "required": ["field", "direction"],
                "additionalProperties": False,
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            # projection (what to show), independent of the conditions
            "fields": {"type": "array", "items": {"enum": ATTRIBUTES}, "minItems": 1, "maxItems": len(ATTRIBUTES)},
            "lookup": {
                "type": "object",
                "properties": {
                    "residents": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 60},
                        "minItems": 1,
                        "maxItems": 5,
                    },
                    "attributes": {"type": "array", "items": {"enum": ATTRIBUTES}, "maxItems": len(ATTRIBUTES)},
                },
                "required": ["residents"],
                "additionalProperties": False,
            },
            "message": {"type": "string", "maxLength": 300},
            "uses_previous": {"type": "boolean"},
        },
        "required": ["action"],
        "additionalProperties": False,
    }


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
