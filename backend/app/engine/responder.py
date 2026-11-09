"""Builds the answer the UI renders from a plan and its execution data. Templates only: every number
comes from the database results, never from the language model."""

from typing import Any

from app.data.repository import Resident
from app.engine.describe import CARE_TYPE_LABELS, GROUP_LABELS, STATUS_LABELS, describe_filters, label_value
from app.engine.executor import ExecutionData
from app.models.plan import QueryPlan

HELP_OPTIONS = [
    {"label": "How many residents are active?", "query": "How many residents are active?"},
    {"label": "Active residents by care type", "query": "Active residents by care type"},
    {"label": "Admitted in the last 3 months", "query": "Residents admitted in the last 3 months"},
    {"label": "Residents over 85 in memory care", "query": "Residents over 85 in memory care"},
    {"label": "Who is Eleanor Vance's caregiver?", "query": "Who is Eleanor Vance's caregiver?"},
]


def _name(r: Resident) -> str:
    return f"{r['firstName']} {r['lastName']}"


def _plural(n: int, word: str = "resident") -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def spotlight(r: Resident, attribute: str) -> dict[str, Any]:
    name = _name(r)
    match attribute:
        case "primary_caregiver":
            return {
                "attribute": attribute,
                "label": "Primary Caregiver",
                "value": r["primaryCaregiver"],
                "supportingDetail": f"Assigned to {name}, room {r['roomNumber']} ({r['wing']}).",
            }
        case "room_number":
            return {
                "attribute": attribute,
                "label": "Room",
                "value": f"{r['roomNumber']} · {r['wing']}",
                "supportingDetail": f"{name} is in room {r['roomNumber']}, {r['wing']}.",
            }
        case "wing":
            return {
                "attribute": attribute,
                "label": "Wing",
                "value": r["wing"],
                "supportingDetail": f"Room {r['roomNumber']}.",
            }
        case "admission_date":
            return {
                "attribute": attribute,
                "label": "Admission Date",
                "value": r["admissionDate"],
                "supportingDetail": f"{name} was admitted on {r['admissionDate']}.",
            }
        case "created_date":
            return {"attribute": attribute, "label": "Registration Date", "value": r["createdDate"]}
        case "discharge_date":
            return {
                "attribute": attribute,
                "label": "Discharge Date",
                "value": r.get("dischargeDate") or "Not discharged",
            }
        case "emergency_contact":
            c = r["emergencyContact"]
            return {
                "attribute": attribute,
                "label": "Emergency Contact",
                "value": f"{c['name']} ({c['relationship']})",
                "supportingDetail": f"Phone: {c['phone']}",
            }
        case "status":
            return {
                "attribute": attribute,
                "label": "Status",
                "value": STATUS_LABELS.get(r["status"], r["status"]),
                "supportingDetail": f"Discharged on {r['dischargeDate']}." if r.get("dischargeDate") else None,
            }
        case "care_type":
            return {
                "attribute": attribute,
                "label": "Care Type",
                "value": CARE_TYPE_LABELS.get(r["careType"], r["careType"]),
            }
        case "age":
            return {
                "attribute": attribute,
                "label": "Age",
                "value": f"{r['age']} years",
                "supportingDetail": r["gender"],
            }
        case "gender":
            return {"attribute": attribute, "label": "Gender", "value": r["gender"]}
    return {"attribute": "fullProfile", "label": "Overview", "value": name}


def _checked_values(plan: QueryPlan, r: Resident) -> str:
    f = plan.filters
    parts = []
    if f.status:
        parts.append(STATUS_LABELS[r["status"]])
    if f.care_type:
        parts.append(CARE_TYPE_LABELS[r["careType"]])
    if f.wing:
        parts.append(r["wing"])
    if f.gender:
        parts.append(r["gender"])
    if f.primary_caregiver:
        parts.append(r["primaryCaregiver"])
    if f.room_number:
        parts.append(f"Room {r['roomNumber']}")
    if f.age:
        parts.append(f"{r['age']} years")
    if f.admission_date:
        parts.append(f"admitted {r['admissionDate']}")
    return " · ".join(parts)


def build_response(plan: QueryPlan, data: ExecutionData) -> dict[str, Any]:
    filter_text = " · ".join(describe_filters(plan.filters))
    scope = f" matching {filter_text}" if filter_text else ""

    if plan.action == "help":
        return {
            "headline": "What I can help with",
            "summaryText": "Ask about residents in plain language: counts, status, care type, wing, room, caregiver, gender, age, "
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
        attrs = list(plan.lookup.attributes)
        if data.checks is not None and len(people) == 1:
            r, ok = people[0], data.checks[0]
            actual = _checked_values(plan, r)
            return {
                "headline": f"{'Yes' if ok else 'No'} — {_name(r)}" + (f" · {actual}" if actual else ""),
                "summaryText": f"{_name(r)} ({r['residentId']}) {'matches' if ok else 'does not match'} {filter_text}. "
                f"Current record: {STATUS_LABELS[r['status']]}, {CARE_TYPE_LABELS[r['careType']]}, room {r['roomNumber']} ({r['wing']}).",
                "count": 1,
                "matchedResident": r,
                "residentsSample": [r],
            }
        if len(people) > 1:
            shown = attrs or ["status", "care_type", "room_number", "primary_caregiver"]
            multi = [
                {**spotlight(r, a), "label": f"{_name(r)} · {spotlight(r, a)['label']}"} for r in people for a in shown
            ]
            return {
                "headline": f"Comparing {len(people)} residents",
                "summaryText": " | ".join(
                    f"{_name(r)}: {', '.join(str(spotlight(r, a)['value']) for a in shown)}" for r in people
                ),
                "count": len(people),
                "matchedResidents": people,
                "multiSpotlight": multi,
                "residentsSample": people,
            }
        r = people[0]
        if not attrs:
            return {
                "headline": f"{_name(r)} — {STATUS_LABELS[r['status']]} · {CARE_TYPE_LABELS[r['careType']]}",
                "summaryText": f"{_name(r)} ({r['residentId']}), {r['age']}, room {r['roomNumber']} ({r['wing']}). "
                f"Primary caregiver: {r['primaryCaregiver']}. Admitted {r['admissionDate']}.",
                "count": 1,
                "matchedResident": r,
                "residentsSample": [r],
            }
        spots = [spotlight(r, a) for a in attrs]
        return {
            "headline": f"{spots[0]['value']} — {spots[0]['label']}"
            if len(spots) == 1
            else " · ".join(str(s["value"]) for s in spots),
            "summaryText": (spots[0].get("supportingDetail") or f"{_name(r)} ({r['residentId']}).")
            if len(spots) == 1
            else f"{_name(r)} ({r['residentId']}): " + "; ".join(f"{s['label']}: {s['value']}" for s in spots) + ".",
            "count": 1,
            "matchedResident": r,
            "attributeSpotlight": spots[0],
            "multiSpotlight": spots,
            "residentsSample": [r],
        }

    if plan.action == "breakdown":
        groups = data.groups or []
        denom = data.matched or 1
        label = " × ".join(GROUP_LABELS[g] for g in plan.group_by)
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
                }
                for g in groups
            ],
            "residentsSample": data.rows,
        }

    if plan.action == "aggregate":
        op = {"avg": "Average", "max": "Highest", "min": "Lowest"}[plan.aggregate.op]
        if data.group_values is not None:
            rows = [
                (" · ".join(label_value(f, k) for f, k in zip(plan.group_by, g.key, strict=True)), g.count, v)
                for g, v in data.group_values
            ]
            return {
                "headline": f"{op} age by " + " × ".join(GROUP_LABELS[g] for g in plan.group_by),
                "summaryText": " | ".join(f"{lbl}: {v if v is not None else '—'} ({_plural(c)})" for lbl, c, v in rows),
                "count": data.matched,
                "breakdown": [
                    {
                        "key": lbl,
                        "label": f"{lbl} — {op.lower()} {v if v is not None else '—'}",
                        "count": c,
                        "percentage": round(c / (data.matched or 1) * 100),
                    }
                    for lbl, c, v in rows
                ],
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
        r = data.rows[0]
        detail = {
            "age": f"{r['age']} years",
            "discharge_date": f"discharged {r.get('dischargeDate')}",
            "created_date": f"registered {r['createdDate']}",
            "last_name": r["lastName"],
        }.get(plan.sort.field, f"admitted {r['admissionDate']}")
        return {
            "headline": f"{_name(r)} — {detail}",
            "summaryText": f"{_name(r)} ({r['residentId']}), {STATUS_LABELS[r['status']]}, {CARE_TYPE_LABELS[r['careType']]}, "
            f"room {r['roomNumber']}. Chosen from {_plural(data.matched)}{scope}.",
            "count": 1,
            "residentsSample": data.rows,
        }
    shown = f" Showing the first {len(data.rows)}." if len(data.rows) < data.matched else ""
    return {
        "headline": "No matching residents" if data.matched == 0 else _plural(data.matched),
        "summaryText": f"{data.matched} of {data.total} residents{scope or ' in the census'}.{shown}",
        "count": data.matched,
        "residentsSample": data.rows,
    }
