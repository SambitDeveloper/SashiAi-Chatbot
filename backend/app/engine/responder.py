"""Builds the answer the UI renders from a plan and its execution data. Templates only: every number
comes from the database results, never from the language model."""

from typing import Any

from app.data.relations import FAMILY_VIEW_FIELDS
from app.data.repository import Resident
from app.data.schema_config import output_fields
from app.engine.describe import (
    ATTRIBUTE_LABELS,
    CARE_TYPE_LABELS,
    GROUP_LABELS,
    STATUS_LABELS,
    describe_filters,
    label_value,
)
from app.engine.executor import ExecutionData
from app.models.plan import QueryPlan

HELP_OPTIONS = [
    {"label": "How many residents are active?", "query": "How many residents are active?"},
    {"label": "Active residents by care type", "query": "Active residents by care type"},
    {"label": "Admitted in the last 3 months", "query": "Residents admitted in the last 3 months"},
    {"label": "Residents over 85 in memory care", "query": "Residents over 85 in memory care"},
    {"label": "Who is Eleanor Vance's caregiver?", "query": "Who is Eleanor Vance's caregiver?"},
]


def status_label(value) -> str:
    """Display text for a status; a missing or malformed value (not text, "", "unknown") reads "not recorded"."""
    if not isinstance(value, str) or value in ("", "unknown"):
        return NOT_RECORDED
    return STATUS_LABELS.get(value, value.title())


def care_label(value) -> str:
    if not isinstance(value, str) or value in ("", "unknown"):
        return NOT_RECORDED
    return CARE_TYPE_LABELS.get(value, value.replace("_", " ").title())


def _name(r: Resident) -> str:
    name = f"{r.get('firstName') or ''} {r.get('lastName') or ''}".strip()
    return name or f"Resident {r.get('residentId', '')}".strip()


def _plural(n: int, word: str = "resident") -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


NOT_RECORDED = "not recorded"


def _readable(r: Resident) -> Resident:
    """Copy for sentences: fields the Care App left empty read "not recorded" instead of "None"."""
    out = dict(r)
    for key in ("age", "roomNumber", "wing", "primaryCaregiver", "admissionDate", "createdDate"):
        if out.get(key) in (None, "", "Unknown"):
            out[key] = NOT_RECORDED
    contact = out.get("emergencyContact")
    out["emergencyContact"] = dict(contact) if isinstance(contact, dict) else {}  # malformed → empty, never a crash
    return out


def _public(r: Resident) -> Resident:
    """A resident row as it may appear in a response: family details only through the answer text (and only what was
    asked), never record ids or contact lists in the raw row. The number of family members is not personal data."""
    return {k: v for k, v in r.items() if k not in FAMILY_VIEW_FIELDS or k == "familyCount"}


def _contacts(r: Resident) -> list[dict]:
    value = r.get("familyContacts")
    return [m for m in value if isinstance(m, dict)] if isinstance(value, list) else []


def _contact_text(m: dict) -> str:
    """One family member for ONE resident's answer: name (relation), phone, email — whatever is recorded."""
    parts = [f"{m.get('name') or NOT_RECORDED}" + (f" ({m['relation']})" if m.get("relation") else "")]
    parts += [m[k] for k in ("phone", "email") if m.get(k)]
    if m.get("type") == "emergency":
        parts.append("emergency contact")
    return " · ".join(parts)


def spotlight(r: Resident, attribute: str) -> dict[str, Any]:
    """One asked-for fact about one resident. The answer states ONLY that fact (data minimisation, 2026-11-19: a
    caregiver question used to also show the room and facility); the supporting line just says whose fact it is."""
    r = _readable(r)
    name = _name(r)

    def fact(label: str, value: Any, detail: str | None = None) -> dict[str, Any]:
        text = str(value).strip() if value is not None else NOT_RECORDED
        return {"attribute": attribute, "label": label, "value": text, "supportingDetail": detail or f"{label} of {name}."}

    match attribute:
        case "primary_caregiver":
            return fact("Primary Caregiver", r["primaryCaregiver"])
        case "room_number":
            return fact("Room", r["roomNumber"])
        case "facility":
            return fact("Facility", r["wing"])
        case "admission_date":
            return fact("Admission Date", r["admissionDate"])
        case "created_date":
            return fact("Registration Date", r["createdDate"])
        case "discharge_date":
            return fact("Discharge Date", r.get("dischargeDate") or "Not discharged")
        case "resident_id":
            return fact("Resident ID", r["residentId"])
        case "emergency_contact":
            c = r["emergencyContact"]
            relation = f" ({c['relationship']})" if c.get("relationship") else ""
            phone = f"Phone: {c['phone']}." if c.get("phone") else None  # part of the contact that was asked for
            # both sources, labelled: the resident record's own contact AND family members recorded as emergency
            family = [m for m in _contacts(r) if m.get("type") == "emergency"]
            if family:
                extra = "Family emergency contacts: " + "; ".join(_contact_text(m) for m in family) + "."
                phone = f"{phone} {extra}" if phone else extra
            return fact("Emergency Contact", f"{c.get('name') or NOT_RECORDED}{relation}", phone)
        case "family_members":
            people = _contacts(r)
            if not people:
                return fact("Family Members", "No family members recorded")
            return fact("Family Members", "; ".join(_contact_text(m) for m in people),
                        f"{len(people)} family member{'s' if len(people) != 1 else ''} of {name}.")  # fmt: skip
        case "family_count":
            return fact("Number of Family Members", r.get("familyCount", 0))
        case "family_relationship":
            people = _contacts(r)  # already narrowed to the selected family member(s) by the executor
            if not people:
                return fact("Relationship", f"No matching family member is recorded for {name}")
            sentences = []
            for m in people:
                relation = (m.get("relation") or "").strip()
                if not relation or relation.lower() == "other":
                    sentences.append(f"{m.get('name')} is listed as a family contact of {name}" + (f" ({relation})" if relation else "") + ".")
                else:
                    sentences.append(f"{m.get('name')} is {name}'s {relation.lower()}.")
            return fact("Relationship", " ".join(sentences), " ".join(sentences))
        case "status":
            return fact("Status", status_label(r.get("status")))
        case "care_type":
            return fact("Care Type", care_label(r.get("careType")))
        case "age":
            return fact("Age", f"{r['age']} years" if r["age"] != NOT_RECORDED else NOT_RECORDED)
        case "gender":
            return fact("Gender", r.get("gender") or NOT_RECORDED)
    return {"attribute": "fullProfile", "label": "Overview", "value": name}


def _checked_values(plan: QueryPlan, r: Resident) -> str:
    f = plan.filters
    r = _readable(r)  # empty fields read "not recorded"; every part is text, whatever type the data had
    parts = []
    if f.status:
        parts.append(status_label(r.get("status")))
    if f.care_type:
        parts.append(care_label(r.get("careType")))
    if f.facility:
        parts.append(str(r.get("wing")))
    if f.gender:
        parts.append(str(r.get("gender") or NOT_RECORDED))
    if f.primary_caregiver:
        parts.append(str(r.get("primaryCaregiver")))
    if f.room_number:
        parts.append(f"Room {r.get('roomNumber')}")
    if f.age:
        parts.append(f"{r.get('age')} years" if r.get("age") != NOT_RECORDED else f"age {NOT_RECORDED}")
    if f.admission_date:
        parts.append(f"admitted {r.get('admissionDate')}")
    return " · ".join(parts)


def build_response(plan: QueryPlan, data: ExecutionData) -> dict[str, Any]:
    filter_text = " · ".join(describe_filters(plan.filters))
    scope = f" matching {filter_text}" if filter_text else ""

    if plan.action == "help":
        return {
            "headline": "What I can help with",
            "summaryText": "Ask about residents in plain language: counts, status, care type, facility, room, caregiver, gender, age, "
            "and admission, registration or discharge dates. You can combine conditions and ask follow-up questions.",
            "count": data.total,
            "clarificationOptions": HELP_OPTIONS,
        }
    if plan.action == "clarify":
        return {
            "headline": "Need a bit more detail",
            "summaryText": plan.message or "Could you rephrase the question?",
            "count": 0,
            "needsClarification": True,
            "clarificationOptions": [o.model_dump() for o in plan.options] if plan.options else None,
        }
    if plan.action == "unsupported":
        return {
            "headline": "Not available",
            "summaryText": plan.message or "That is outside the resident data this assistant can access.",
            "count": 0,
            "isPolicyViolation": True,
            "clarificationOptions": HELP_OPTIONS[:3],
        }

    if plan.action == "lookup":
        people = data.looked_up or []
        if not people:
            return {
                "headline": "Resident not found",
                "summaryText": "No matching resident record was found.",
                "count": 0,
                "needsClarification": True,
            }
        attrs = plan.output_fields  # explicit `fields` and legacy `lookup.attributes`, normalised
        if data.checks is not None and len(people) == 1:
            r, ok = people[0], data.checks[0]
            actual = _checked_values(plan, r)
            return {
                "headline": f"{'Yes' if ok else 'No'} — {_name(r)}" + (f" · {actual}" if actual else ""),
                "summaryText": f"{_name(r)} ({r['residentId']}) {'matches' if ok else 'does not match'} {filter_text}. "
                f"Current record: {status_label(r.get('status'))}, {care_label(r.get('careType'))}, room {_readable(r)['roomNumber']} ({_readable(r)['wing']}).",
                "count": 1,
                "matchedResident": _public(r),
                "residentsSample": [_public(r)],
            }
        if len(people) > 1:
            shown = attrs or ["status", "care_type", "room_number", "primary_caregiver"]
            multi = [
                {**spotlight(r, a), "label": f"{_name(r)} · {spotlight(r, a)['label']}"} for r in people for a in shown
            ]
            out = {
                "headline": f"Comparing {len(people)} residents",
                "summaryText": " | ".join(
                    f"{_name(r)}: {', '.join(str(spotlight(r, a)['value']) for a in shown)}" for r in people
                ),
                "count": len(people),
                "multiSpotlight": multi,
            }
            if not attrs:  # a general comparison shows the profiles; asked-for facts send only those facts
                out |= {"matchedResidents": [_public(p) for p in people], "residentsSample": [_public(p) for p in people]}
            return out
        r = people[0]
        if not attrs:
            d = _readable(r)
            return {
                "headline": f"{_name(r)} — {status_label(r.get("status"))} · {care_label(r.get("careType"))}",
                "summaryText": f"{_name(d)} ({d['residentId']}), age {d['age']}, room {d['roomNumber']} ({d['wing']}). "
                f"Caregivers: {d['primaryCaregiver']}. Admitted: {d['admissionDate']}.",
                "count": 1,
                "matchedResident": _public(r),
                "residentsSample": [_public(r)],
            }
        # Specific facts were asked for: answer with exactly those, not the resident's whole record (data minimisation).
        spots = [spotlight(r, a) for a in attrs]
        return {
            "headline": f"{spots[0]['value']} — {spots[0]['label']}"
            if len(spots) == 1
            else " · ".join(str(s["value"]) for s in spots),
            "summaryText": (spots[0].get("supportingDetail") or f"{_name(r)}.")
            if len(spots) == 1
            else f"{_name(r)}: " + "; ".join(f"{s['label']}: {s['value']}" for s in spots) + ".",
            "count": 1,
            "attributeSpotlight": spots[0],
            "multiSpotlight": spots,
        }

    if plan.action == "breakdown":
        groups = data.groups or []
        denom = data.matched or 1
        label = " × ".join(GROUP_LABELS[g] for g in plan.group_by)
        if plan.output_fields and data.group_details_complete is False:
            return {
                "headline": f"{_plural(data.matched)} by {label}",
                "summaryText": (
                    "The group counts are available, but all requested resident details exceed the result limit. "
                    "Narrow the filters to retrieve the complete names and requested fields."
                ),
                "count": data.matched,
                "needsClarification": True,
                "breakdown": [
                    {
                        "key": "|".join(g.key),
                        "label": " · ".join(label_value(f, k) for f, k in zip(plan.group_by, g.key, strict=True)),
                        "count": g.count,
                        "percentage": round(g.count / denom * 100),
                    }
                    for g in groups
                ],
            }
        return {
            "headline": f"{_plural(data.matched)} by {label}",
            "summaryText": f"Breakdown of {_plural(data.matched)}{scope} (of {data.total} in the census).",
            "count": data.matched,
            "breakdown": [
                {
                    "key": "|".join(g.key),
                    "label": " · ".join(label_value(f, k) for f, k in zip(plan.group_by, g.key, strict=True)),
                    "count": g.count,
                    "percentage": round(g.count / denom * 100),
                    **(
                        {"residents": (data.group_residents or {}).get(g.key, [])}
                        if plan.output_fields
                        else {}
                    ),
                }
                for g in groups
            ],
            **({"columns": _columns(plan)} if plan.output_fields else {"residentsSample": data.rows}),
        }

    if plan.action == "aggregate":
        op = {"avg": "Average", "max": "Highest", "min": "Lowest"}[plan.aggregate.op]
        if data.group_values is not None:

            def shown(g) -> str:
                # missing-value rule: residents without a recorded age are counted in the group but not averaged
                if g.value is None:
                    return "age not recorded"
                if g.n_values < g.count:
                    return f"{g.value} (age recorded for {g.n_values} of {g.count})"
                return str(g.value)

            rows = [
                (" · ".join(label_value(f, k) for f, k in zip(plan.group_by, g.key, strict=True)), g.count, shown(g))
                for g in data.group_values
            ]
            return {
                "headline": f"{op} age by " + " × ".join(GROUP_LABELS[g] for g in plan.group_by),
                "summaryText": " | ".join(f"{lbl}: {v} ({_plural(c)})" for lbl, c, v in rows),
                "count": data.matched,
                "breakdown": [
                    {
                        "key": lbl,
                        "label": f"{lbl} — {op.lower()} {v}",
                        "count": c,
                        "percentage": round(c / (data.matched or 1) * 100),
                    }
                    for lbl, c, v in rows
                ],
            }
        if data.aggregate_value is None and data.matched:
            # residents match, but none has a recorded age: say so — "no matching residents" would be false
            return {
                "headline": f"{op} age: not recorded",
                "summaryText": f"Age is not recorded for any of the {_plural(data.matched)}{scope}.",
                "count": data.matched,
            }
        if data.aggregate_value is None:
            return {
                "headline": "No matching residents",
                "summaryText": f"No residents{scope}, so there is no {op.lower()} age.",
                "count": 0,
            }
        return {
            "headline": f"{op} age: {data.aggregate_value}",
            "summaryText": f"{op} age across {_plural(data.matched)}{scope}.",
            "count": data.matched,
            "aggregateValue": data.aggregate_value,
        }

    # count / list
    if plan.sort and plan.limit == 1 and data.rows:
        # Rows can be incomplete (a projection returns only the asked-for fields; the Care App leaves fields empty):
        # every field is read with .get and shown as "not recorded" when absent, never a KeyError.
        r = _readable(data.rows[0])
        if plan.sort.field == "age":
            detail = f"{r['age']} years" if r.get("age") not in (None, NOT_RECORDED) else f"age {NOT_RECORDED}"
        else:
            detail = {
                "discharge_date": f"discharged {r.get('dischargeDate') or NOT_RECORDED}",
                "created_date": f"registered {r.get('createdDate') or NOT_RECORDED}",
                "last_name": r.get("lastName") or NOT_RECORDED,
            }.get(plan.sort.field, f"admitted {r.get('admissionDate') or NOT_RECORDED}")
        facts = [f"{_name(r)} ({r.get('residentId', '')})"]
        if r.get("status"):
            facts.append(status_label(r.get("status")))
        if r.get("careType"):
            facts.append(care_label(r.get("careType")))
        if r.get("roomNumber") is not None:
            facts.append(f"room {r['roomNumber']}")
        return {
            "headline": f"{_name(r)} — {detail}",
            "summaryText": ", ".join(facts) + f". Chosen from {_plural(data.matched)}{scope}.",
            "count": 1,
            "residentsSample": data.rows,
        }
    # A count fetches no rows; a list fetches only the current page. Report total matches and page size honestly.
    returned = len(data.rows)
    has_more = returned < data.matched
    if plan.action == "count":
        summary = f"{data.matched} of {data.total} residents{scope or ' in the census'}."
    elif has_more:
        summary = (
            f"{data.matched} of {data.total} residents{scope or ' in the census'}. "
            f"Showing {returned} of {data.matched}; use pagination to view the rest."
        )
    else:
        summary = f"{data.matched} of {data.total} residents{scope or ' in the census'}."
    return {
        "headline": "No matching residents" if data.matched == 0 else _plural(data.matched),
        "summaryText": summary,
        "count": data.matched,
        "totalMatched": data.matched,
        "returnedCount": returned,
        "hasMore": has_more,
        "residentsSample": data.rows,
        # explicit projection: the columns to show (view field + label), in the order asked; absent → default columns
        **({"columns": _columns(plan)} if plan.action == "list" and plan.output_fields else {}),
    }


def _columns(plan: QueryPlan) -> list[dict[str, str]]:
    mapping = output_fields()
    return [
        {"key": mapping[a], "label": ATTRIBUTE_LABELS.get(a, a.replace("_", " ").title()), "attribute": a}
        for a in plan.output_fields
        if a in mapping
    ]
