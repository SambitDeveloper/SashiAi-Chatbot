"""System prompt and output schema for question → QueryPlan translation.

SYSTEM_PROMPT is fully static (no dates, no data) so it is served from the prompt cache on every
request. Per-request context (today's date, wing names, previous plan, the question) goes in the
user message, after the cached prefix.
"""

SYSTEM_PROMPT = """You translate questions from care-home staff into ONE JSON query plan for a resident database. \
You never answer the question yourself and never invent data; the database computes every answer from your plan.

Resident fields you can use in conditions, group by or ask about:
- status: active | inactive | pending (waitlist, awaiting intake) | admitted (recently admitted) | discharged (left, moved out)
- care_type: assisted_living | memory_care (dementia, Alzheimer's) | independent_living | skilled_nursing (nursing, rehab, SNF) | respite_care (short stay)
- wing (building wing), room_number, gender (Female | Male | Other), age (years)
- primary_caregiver (the assigned nurse / aide / coordinator / caretaker)
- admission_date, created_date (record registered/created), discharge_date
- emergency_contact (only for a specific resident)

Actions:
- count: how many residents match the conditions.
- list: which residents match; use sort + limit for "oldest", "youngest", "most recently admitted", "top 5".
- breakdown: counts grouped by one or two fields (group_by), e.g. "by status", "per wing", "each care type".
- aggregate: average/min/max age (aggregate.op), optionally with group_by.
- lookup: facts about specific residents (lookup.residents + lookup.attributes). With conditions it is a yes/no \
check, e.g. "is resident_1 in memory care?" → lookup + condition {"field":"care_type","include":["memory_care"]}.
- clarify: the question is ambiguous or missing something essential; put one short question in message.
- unsupported: the question needs data not listed above (diagnoses, medications, doctors, diet, allergies, \
billing, birthdays, visitors, staff payroll, etc.) or is not about residents; put a one-sentence reason in message.
- help: greetings, thanks, or "what can you do".

Rules:
- Put every condition in the question into "conditions": one entry per field, in any order. Fields status, \
care_type, wing, gender, primary_caregiver and room_number use include/exclude lists; age uses min/max; \
admission_date, created_date and discharge_date use start/end.
- Add only conditions the question states. Never add a date condition for a date the question does not mention: \
"joined", "came in" or "admitted in X" is admission_date only.
- Negation ("not", "excluding", "other than", "except") goes in exclude; alternatives ("A or B", "A and B" for \
the same field) go together in include.
- Dates are YYYY-MM-DD. Resolve relative dates ("last month", "this year", "since March", "past 30 days") \
against Today from the user message. A month without a year means its most recent occurrence that is not \
in the future. "Admitted in X" is an admission_date condition; "discharged in X" is a discharge_date condition \
plus status discharged.
- Words like resident_1 are placeholders for specific residents: use them unchanged in lookup.residents. \
Any other person named as a resident also goes in lookup.residents exactly as written. A staff name (nurse, \
aide, coordinator, caretaker) is a primary_caregiver condition, written exactly as in the question.
- Use only wing names from the provided list (copy the one meant). Copy room numbers as written.
- If a previous plan is provided and the new question refers back to it ("them", "those", "of these", \
"what about ...", "and how many ..."), start from the previous plan's conditions, apply the change and set uses_previous to true. Otherwise ignore it and omit uses_previous.
- Never use group_by on a field that the question pins to a single value.
- Omit keys you do not need. Output only the JSON plan."""


SET_FIELDS = ["status", "care_type", "wing", "gender", "primary_caregiver", "room_number"]
RANGE_FIELDS = {
    "age": ("min", "max"),
    "admission_date": ("start", "end"),
    "created_date": ("start", "end"),
    "discharge_date": ("start", "end"),
}
CONDITION_FIELDS = SET_FIELDS + list(RANGE_FIELDS)
GROUP_FIELDS = ["status", "care_type", "wing", "gender", "primary_caregiver"]
SORT_FIELDS = ["admission_date", "created_date", "discharge_date", "age", "last_name"]
ATTRIBUTES = [
    "primary_caregiver",
    "room_number",
    "wing",
    "admission_date",
    "created_date",
    "discharge_date",
    "emergency_contact",
    "status",
    "care_type",
    "age",
    "gender",
]

# JSON schema for the model's structured output. Same as app.models.plan.QueryPlan except that filters are a
# list of conditions (see below); the resolver converts them (tested in test_plan_and_prompt.py). Value enums and
# numeric/length limits are enforced by Pydantic after the response.
OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["count", "list", "breakdown", "aggregate", "lookup", "clarify", "unsupported", "help"],
        },
        # A list, not an object with fixed keys: models generate object keys in schema order, so a condition
        # mentioned late in the sentence (e.g. a caregiver after a date) could no longer be written and got
        # dropped or misplaced. List entries can come in any order.
        "conditions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "enum": CONDITION_FIELDS},
                    "include": {"type": "array", "items": {"type": "string"}},
                    "exclude": {"type": "array", "items": {"type": "string"}},
                    "min": {"type": "integer"},
                    "max": {"type": "integer"},
                    "start": {"type": "string", "format": "date"},
                    "end": {"type": "string", "format": "date"},
                },
                "required": ["field"],
                "additionalProperties": False,
            },
        },
        "group_by": {"type": "array", "items": {"type": "string", "enum": GROUP_FIELDS}},
        "aggregate": {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["avg", "min", "max"]},
                "field": {"type": "string", "enum": ["age"]},
            },
            "required": ["op"],
            "additionalProperties": False,
        },
        "sort": {
            "type": "object",
            "properties": {
                "field": {"type": "string", "enum": SORT_FIELDS},
                "direction": {"type": "string", "enum": ["asc", "desc"]},
            },
            "required": ["field", "direction"],
            "additionalProperties": False,
        },
        "limit": {"type": "integer"},
        "lookup": {
            "type": "object",
            "properties": {
                "residents": {"type": "array", "items": {"type": "string"}},
                "attributes": {"type": "array", "items": {"type": "string", "enum": ATTRIBUTES}},
            },
            "required": ["residents"],
            "additionalProperties": False,
        },
        "message": {"type": "string"},
        "uses_previous": {"type": "boolean"},
    },
    "required": ["action"],
    "additionalProperties": False,
}


def build_user_message(
    masked_question: str, today: str, wings: list[str], previous_plan_json: str | None, hint: str | None = None
) -> str:
    parts = [f"Today: {today}", f"Wings: {', '.join(wings)}"]
    if previous_plan_json:
        parts.append(f"Previous plan (use only if the question refers back to it): {previous_plan_json}")
    parts.append(f"Question: {masked_question}")
    if hint:
        parts.append(f"Check: {hint}")
    return "\n".join(parts)


def conditions_to_filters(conditions: list) -> dict:
    """Model output (list of conditions) → QueryPlan.filters. Unknown shapes are kept so validation rejects them."""
    filters: dict = {}
    for cond in conditions:
        if not isinstance(cond, dict) or "field" not in cond:
            filters.setdefault("_invalid", []).append(cond)
            continue
        field = cond["field"]
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
