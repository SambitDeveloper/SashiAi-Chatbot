"""Human-readable "Interpreted as" chips, so staff can see exactly how a question was understood."""

from collections.abc import Callable
from datetime import date

from app.data.repository import MISSING_GROUP
from app.models.plan import DateRange, Filters, QueryPlan

STATUS_LABELS = {
    "active": "Active",
    "away": "Away",
    "inactive": "Inactive",
    "pending": "Pending Intake",
    "admitted": "Recently Admitted",
    "discharged": "Discharged",
}
CARE_TYPE_LABELS = {
    "assisted_living": "Assisted Living",
    "memory_care": "Memory Care",
    "independent_living": "Independent Living",
    "skilled_nursing": "Skilled Nursing",
    "respite_care": "Respite Care",
}
GROUP_LABELS = {
    "status": "Status",
    "care_type": "Care Type",
    "facility": "Facility",
    "gender": "Gender",
    "primary_caregiver": "Caregiver",
}
ATTRIBUTE_LABELS = {
    "primary_caregiver": "Primary Caregiver",
    "room_number": "Room",
    "facility": "Facility",
    "admission_date": "Admission Date",
    "created_date": "Registration Date",
    "discharge_date": "Discharge Date",
    "emergency_contact": "Emergency Contact",
    "resident_id": "Resident ID",
    "status": "Status",
    "care_type": "Care Type",
    "age": "Age",
    "gender": "Gender",
    "family_members": "Family Members",
    "family_count": "Number of Family Members",
}
FAMILY_FLAG_LABELS = {"portal_access": "portal access", "app_access": "app access", "health_access": "health access",
                      "guarantor": "guarantor"}  # fmt: skip
_SET_FIELDS = [
    ("status", "Status"),
    ("care_type", "Care Type"),
    ("facility", "Facility"),
    ("gender", "Gender"),
    ("primary_caregiver", "Caregiver"),
    ("room_number", "Room"),
    ("family_relation", "Family"),
    ("family_type", "Family contact"),
    ("family_flags", "Family with"),
]


def label_value(field_name: str, value: str) -> str:
    if value == MISSING_GROUP:  # the one group of residents whose field is not recorded
        return MISSING_GROUP
    if field_name == "status":
        return STATUS_LABELS.get(value, value)
    if field_name == "care_type":
        return CARE_TYPE_LABELS.get(value, value)
    if field_name == "family_flags":
        return FAMILY_FLAG_LABELS.get(value, value)
    if field_name in ("family_relation", "family_type"):
        return value.replace("_", " ")
    return value


def fmt_date(d: date | str) -> str:
    d = date.fromisoformat(d) if isinstance(d, str) else d
    return f"{d.strftime('%b')} {d.day}, {d.year}"


def _range_text(r: DateRange) -> str:
    if r.start and r.end:
        return f"on {fmt_date(r.start)}" if r.start == r.end else f"{fmt_date(r.start)} – {fmt_date(r.end)}"
    return f"since {fmt_date(r.start)}" if r.start else f"up to {fmt_date(r.end)}"


def describe_filters(f: Filters) -> list[str]:
    chips: list[str] = []
    for field_name, label in _SET_FIELDS:
        sf = getattr(f, field_name)
        if sf is None:
            continue
        if sf.include or sf.missing is True:
            shown = [label_value(field_name, v) for v in sf.include or []]
            if sf.missing is True:
                shown.append("not recorded")
            chips.append(f"{label}: {' or '.join(shown)}")
        if sf.missing is False:
            chips.append(f"{label}: recorded")
        if sf.exclude:
            chips.append(f"{label}: not {' / '.join(label_value(field_name, v) for v in sf.exclude)}")
        for words, needles in (("starts with", sf.starts_with), ("contains", sf.contains)):
            if needles:  # literal text as the user wrote it (never a vocabulary label)
                chips.append(f"{label}: {words} " + " or ".join(f'"{n}"' for n in needles))
    if f.family_count:
        lo, hi = f.family_count.min, f.family_count.max
        if hi == 0:
            chips.append("Family: none recorded")
        elif lo is not None and lo == hi:
            chips.append(f"Family members: exactly {lo}")
        elif lo is not None and hi is not None:
            chips.append(f"Family members: {lo} to {hi}")
        else:
            chips.append(f"Family members: at least {lo}" if lo is not None else f"Family members: at most {hi}")
    if f.age:
        lo, hi = f.age.min, f.age.max
        if not f.age.has_range:
            chips.append("Age: not recorded" if f.age.missing else "Age: recorded")
        else:
            text = f"Age: {lo}" if lo == hi else f"Age: {lo}–{hi}" if lo is not None and hi is not None else (
                f"Age: {lo}+" if lo is not None else f"Age: {hi} or younger"
            )
            chips.append(text + (" or not recorded" if f.age.missing else ""))
    for field_name, label in (("admission_date", "Admitted"), ("created_date", "Registered"), ("discharge_date", "Discharged")):
        r = getattr(f, field_name)
        if r is None:
            continue
        if not r.has_range:
            chips.append(f"{label}: {'date not recorded' if r.missing else 'date recorded'}")
        else:
            chips.append(f"{label}: {_range_text(r)}" + (" or date not recorded" if r.missing else ""))
    return chips


def describe_plan(plan: QueryPlan, name_of: Callable[[str], str]) -> list[str]:
    chips: list[str] = []
    match plan.action:
        case "count":
            chips.append("Count")
        case "list":
            chips.append("List")
            if plan.output_fields:
                chips.append("Show: " + ", ".join(ATTRIBUTE_LABELS.get(a, a) for a in plan.output_fields))
        case "breakdown":
            chips.append("Breakdown by " + " × ".join(GROUP_LABELS[g] for g in plan.group_by))
            if plan.output_fields:
                chips.append("Show: " + ", ".join(ATTRIBUTE_LABELS.get(a, a) for a in plan.output_fields))
        case "aggregate":
            op = {"avg": "Average", "max": "Highest", "min": "Lowest"}[plan.aggregate.op]
            chips.append(
                f"{op} age" + (" by " + " × ".join(GROUP_LABELS[g] for g in plan.group_by) if plan.group_by else "")
            )
        case "lookup":
            chips.extend(f"Resident: {name_of(i)} ({i})" for i in plan.lookup.residents)
            chips.extend(f"Asked: {ATTRIBUTE_LABELS.get(a, a)}" for a in plan.output_fields)
    chips.extend(describe_filters(plan.filters))
    if plan.sort:
        s = plan.sort
        if s.field == "age":
            what = "Oldest first" if s.direction == "desc" else "Youngest first"
        else:
            what = f"{'Most recent' if s.direction == 'desc' else 'Earliest'} {ATTRIBUTE_LABELS.get(s.field, s.field).lower()}"
        if s.field == "family_count":
            what = "Most family members first" if s.direction == "desc" else "Fewest family members first"
        chips.append(f"{what} · top {plan.limit}" if plan.limit else what)
    return chips
