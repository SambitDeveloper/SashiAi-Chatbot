"""System prompt and output schema for question → QueryPlan translation.

SYSTEM_PROMPT is fully static (no dates, no data) so it is served from the prompt cache on every
request. Per-request context (today's date, wing names, previous plan, the question) goes in the
user message, after the cached prefix.
"""

from app.models.plan import CARE_TYPES, STATUSES

SYSTEM_PROMPT = """You translate questions from care-home staff into ONE JSON query plan for a resident database. \
You never answer the question yourself and never invent data; the database computes every answer from your plan.

Resident fields you can filter, group or ask about:
- status: active | inactive | pending (waitlist, awaiting intake) | admitted (recently admitted) | discharged (left, moved out)
- care_type: assisted_living | memory_care (dementia, Alzheimer's) | independent_living | skilled_nursing (nursing, rehab, SNF) | respite_care (short stay)
- wing (building wing), room_number, gender (Female | Male | Other), age (years)
- primary_caregiver (the assigned nurse / aide / coordinator / caretaker)
- admission_date, created_date (record registered/created), discharge_date
- emergency_contact (only for a specific resident)

Actions:
- count: how many residents match the filters.
- list: which residents match; use sort + limit for "oldest", "youngest", "most recently admitted", "top 5".
- breakdown: counts grouped by one or two fields (group_by), e.g. "by status", "per wing", "each care type".
- aggregate: average/min/max age (aggregate.op), optionally with group_by.
- lookup: facts about specific residents (lookup.residents + lookup.attributes). With filters it is a yes/no \
check, e.g. "is resident_1 in memory care?" → lookup + filters.care_type.include=["memory_care"].
- clarify: the question is ambiguous or missing something essential; put one short question in message.
- unsupported: the question needs data not listed above (diagnoses, medications, doctors, diet, allergies, \
billing, birthdays, visitors, staff payroll, etc.) or is not about residents; put a one-sentence reason in message.
- help: greetings, thanks, or "what can you do".

Rules:
- Combine every condition in the question into filters. Negation ("not", "excluding", "other than", "except") \
goes in exclude; alternatives ("A or B", "A and B" for the same field) go together in include.
- Dates are YYYY-MM-DD. Resolve relative dates ("last month", "this year", "since March", "past 30 days") \
against Today from the user message. A month without a year means its most recent occurrence that is not \
in the future. "Admitted in X" filters admission_date; "discharged in X" filters discharge_date and status.
- Words like resident_1 are placeholders for specific residents: use them unchanged in lookup.residents. \
Any other person named as a resident also goes in lookup.residents exactly as written. Staff names go in \
primary_caregiver.include exactly as written (without needing titles).
- Use only wing names from the provided list (copy the one meant). Copy room numbers as written.
- If a previous plan is provided and the new question refers back to it ("them", "those", "of these", \
"what about ...", "and how many ..."), start from the previous plan, apply the change and set uses_previous to true. Otherwise ignore it and omit uses_previous.
- Never use group_by on a field that the question pins to a single value.
- Omit keys you do not need. Output only the JSON plan."""


def _set_filter(item: dict) -> dict:
    return {
        "type": "object",
        "properties": {"include": {"type": "array", "items": item}, "exclude": {"type": "array", "items": item}},
        "additionalProperties": False,
    }


_DATE_RANGE = {
    "type": "object",
    "properties": {"start": {"type": "string", "format": "date"}, "end": {"type": "string", "format": "date"}},
    "additionalProperties": False,
}

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

# JSON schema for Claude structured outputs. Mirrors app.models.plan.QueryPlan (tested in test_prompt.py);
# numeric/length limits are enforced by Pydantic after the response, as the API schema does not support them.
OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["count", "list", "breakdown", "aggregate", "lookup", "clarify", "unsupported", "help"],
        },
        "filters": {
            "type": "object",
            "properties": {
                "status": _set_filter({"type": "string", "enum": list(STATUSES)}),
                "care_type": _set_filter({"type": "string", "enum": list(CARE_TYPES)}),
                "wing": _set_filter({"type": "string"}),
                "gender": _set_filter({"type": "string", "enum": ["Female", "Male", "Other"]}),
                "primary_caregiver": _set_filter({"type": "string"}),
                "room_number": _set_filter({"type": "string"}),
                "age": {
                    "type": "object",
                    "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}},
                    "additionalProperties": False,
                },
                "admission_date": _DATE_RANGE,
                "created_date": _DATE_RANGE,
                "discharge_date": _DATE_RANGE,
            },
            "additionalProperties": False,
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


def build_user_message(masked_question: str, today: str, wings: list[str], previous_plan_json: str | None) -> str:
    parts = [f"Today: {today}", f"Wings: {', '.join(wings)}"]
    if previous_plan_json:
        parts.append(f"Previous plan (use only if the question refers back to it): {previous_plan_json}")
    parts.append(f"Question: {masked_question}")
    return "\n".join(parts)
