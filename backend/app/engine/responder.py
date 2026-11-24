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
    STAFF_TYPE_LABELS,
    STATUS_LABELS,
    describe_filters,
    label_value,
    role_label,
)
from app.engine.executor import ExecutionData
from app.models.plan import STAFF_FIELD_MAP, QueryPlan

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
    return {k: v for k, v in r.items() if (k not in FAMILY_VIEW_FIELDS or k == "familyCount") and k != "careTeamMembers"}


def _team(r: Resident) -> list[dict]:
    value = r.get("careTeamMembers")
    return [m for m in value if isinstance(m, dict)] if isinstance(value, list) else []


def _team_text(members: list[dict]) -> str:
    """One resident's care team grouped by designation as recorded: "Case Manager: Jane Okafor · Nurse: A, B (inactive)"."""
    groups: dict[str, list[str]] = {}
    for m in members:
        who = (m.get("name") or NOT_RECORDED) + (" (inactive)" if m.get("status") == "inactive" else "")
        groups.setdefault(m.get("designation") or "No designation", []).append(who)
    return " · ".join(f"{d}: {', '.join(names)}" for d, names in sorted(groups.items()))


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
        case "care_team":
            members = _team(r)  # narrowed to the asked-for role(s) by the executor ("resident_1's physician")
            if not members:
                return fact("Care Team", "No matching staff member is assigned")
            active = sum(1 for m in members if m.get("status") == "active")
            detail = f"{len(members)} assigned staff member{'s' if len(members) != 1 else ''} of {name}" + (
                f" ({len(members) - active} inactive)." if active < len(members) else ".")
            return fact("Care Team", _team_text(members), detail)
        case "primary_physician":
            return fact("Primary Physician", r.get("primaryPhysician") or "No primary physician recorded")
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
    if f.staff_role or f.staff_designation:
        roles = [role_label(x) for x in r.get("staffRoles") or [] if isinstance(x, str)]
        parts.append("staff: " + (", ".join(roles) if roles else "no active staff with a role"))
    return " · ".join(parts)


def build_response(plan: QueryPlan, data: ExecutionData) -> dict[str, Any]:
    filter_text = " · ".join(describe_filters(plan.filters))
    scope = f" matching {filter_text}" if filter_text else ""
    if plan.entity == "staff" and plan.is_data_query:
        return _staff_response(plan, data, filter_text)

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
            f = plan.filters
            if f.staff_name is not None and f.staff_name.include and f.model_dump(exclude_none=True).keys() == {"staff_name"}:
                # "Is Jone part of Sagar Dodia's care team?": said as membership, not "matches Staff member: Jone"
                wanted = set(f.staff_name.include)
                on_team = [m for m in r.get("careTeamMembers") or [] if m.get("name") in wanted]
                if ok and on_team:  # name who was found, with the designation: "Jone (Transport Driver)"
                    who = ", ".join(dict.fromkeys(m["name"] + (f" ({m['designation']})" if m.get("designation") else "")
                                                  for m in on_team))  # fmt: skip
                else:
                    who = " or ".join(dict.fromkeys({n.lower(): n for n in f.staff_name.include}.values()))
                return {
                    "headline": f"{'Yes' if ok else 'No'} — {who}",
                    "summaryText": f"{who} {'is' if ok else 'is not'} on {_name(r)}'s care team ({r['residentId']}).",
                    "count": 1,
                    "matchedResident": _public(r),
                    "residentsSample": [_public(r)],
                }
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


# ---------------------------------------------------------------------- staff answers (plan.entity == "staff")
def _staff_value(row: dict, attribute: str) -> str:
    value = row.get(STAFF_FIELD_MAP[attribute])
    if attribute == "staff_role":
        return role_label(value) if isinstance(value, str) and value else "No role"
    if attribute == "staff_status":
        return str(value or NOT_RECORDED).title()
    if attribute == "staff_type":
        return STAFF_TYPE_LABELS.get(value, NOT_RECORDED) if value else NOT_RECORDED
    if attribute == "staff_residents":
        return ", ".join(value) if value else "No residents assigned"
    if attribute == "staff_resident_count":
        return str(value if value is not None else 0)
    return str(value) if value not in (None, "") else NOT_RECORDED


def _staff_row(row: dict) -> dict:
    """A staff member as a result row the UI renders like a resident row (name + the requested columns). The record
    key (staffs.cName) never leaves the backend."""
    out = {k: v for k, v in row.items() if k not in ("staffId", "name")}
    return {"firstName": row.get("name") or NOT_RECORDED, "lastName": "", "residentId": "", **out}


def _staff_columns(plan: QueryPlan) -> list[dict[str, str]]:
    return [{"key": STAFF_FIELD_MAP[a], "label": ATTRIBUTE_LABELS.get(a, a), "attribute": a} for a in plan.output_fields]


def _staff_word(n: int) -> str:
    return f"{n} staff member{'' if n == 1 else 's'}"


def _staff_response(plan: QueryPlan, data: ExecutionData, filter_text: str) -> dict[str, Any]:
    scope = f" matching {filter_text}" if filter_text else ""
    active_note = " Only active staff are included; ask for inactive staff to include them." if data.staff_default_active else ""
    who = "active staff members" if data.staff_default_active else "staff members"

    if plan.action == "lookup":
        rows = data.staff_rows or []
        if not rows:
            return {"headline": "Staff member not found", "summaryText": "No matching staff record was found.", "count": 0,
                    "needsClarification": True}  # fmt: skip
        attrs = plan.output_fields or ["staff_designation", "staff_status", "staff_facility", "staff_resident_count"]
        if data.staff_contacts_withheld:
            attrs = [a for a in attrs if a not in ("staff_email", "staff_phone")]
        if data.checks is not None and len(rows) == 1:
            ok = data.checks[0]
            r = rows[0]
            return {
                "headline": f"{'Yes' if ok else 'No'} — {r.get('name')}",
                "summaryText": f"{r.get('name')} {'matches' if ok else 'does not match'} {filter_text}. Designation: "
                f"{_staff_value(r, 'staff_designation')}, {_staff_value(r, 'staff_status').lower()}.",
                "count": 1,
            }
        spots = [
            {"attribute": a, "label": f"{r.get('name')} · {ATTRIBUTE_LABELS.get(a, a)}" if len(rows) > 1 else ATTRIBUTE_LABELS.get(a, a),
             "value": _staff_value(r, a), "supportingDetail": f"{ATTRIBUTE_LABELS.get(a, a)} of {r.get('name')}."}
            for r in rows for a in attrs
        ]  # fmt: skip
        note = " Contact details are shown for one staff member at a time." if data.staff_contacts_withheld else ""
        if len(spots) == 1:
            return {"headline": f"{spots[0]['value']} — {spots[0]['label']}", "summaryText": spots[0]["supportingDetail"] + note,
                    "count": 1, "attributeSpotlight": spots[0], "multiSpotlight": spots}  # fmt: skip
        return {
            "headline": " · ".join(str(s["value"]) for s in spots[: len(attrs)]) if len(rows) == 1 else f"{len(rows)} staff members",
            "summaryText": " | ".join(f"{r.get('name')}: " + "; ".join(f"{ATTRIBUTE_LABELS.get(a, a)}: {_staff_value(r, a)}" for a in attrs)
                                      for r in rows) + "." + note,
            "count": len(rows),
            "multiSpotlight": spots,
        }  # fmt: skip

    if plan.action == "breakdown":
        groups = data.groups or []
        denom = data.matched or 1
        label = " × ".join(GROUP_LABELS[g] for g in plan.group_by)
        return {
            "headline": f"{_staff_word(data.matched)} by {label}",
            "summaryText": f"Breakdown of {data.matched} {who}{scope} ({data.total} staff in total).{active_note}",
            "count": data.matched,
            "breakdown": [
                {"key": "|".join(g.key), "label": " · ".join(label_value(f, k) for f, k in zip(plan.group_by, g.key, strict=True)),
                 "count": g.count, "percentage": round(g.count / denom * 100)}
                for g in groups
            ],
        }  # fmt: skip

    if plan.action == "count":
        return {
            "headline": f"{_staff_word(data.matched)}" if data.matched else "No matching staff members",
            "summaryText": f"{data.matched} {who}{scope} ({data.total} staff in total).{active_note}",
            "count": data.matched,
        }
    rows = data.staff_rows or []
    has_more = len(rows) < data.matched
    if plan.sort and plan.limit == 1 and rows:
        r = rows[0]
        detail = f"{_staff_value(r, plan.sort.field)} {ATTRIBUTE_LABELS.get(plan.sort.field, '').lower()}".strip()
        return {"headline": f"{r.get('name')} — {detail}", "summaryText": f"Chosen from {data.matched} {who}{scope}.{active_note}",
                "count": 1, "residentsSample": [_staff_row(r)], "columns": _staff_columns(plan)}  # fmt: skip
    summary = f"{data.matched} {who}{scope}." + (f" Showing {len(rows)} of {data.matched}." if has_more else "") + active_note
    return {
        "headline": "No matching staff members" if not data.matched else _staff_word(data.matched),
        "summaryText": summary,
        "count": data.matched,
        "totalMatched": data.matched,
        "returnedCount": len(rows),
        "hasMore": has_more,
        "entity": "staff",
        "residentsSample": [_staff_row(r) for r in rows],
        "columns": _staff_columns(plan),
    }
