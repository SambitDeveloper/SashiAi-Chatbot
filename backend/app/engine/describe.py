"""Human-readable "Interpreted as" chips, so staff can see exactly how a question was understood."""

from collections.abc import Callable
from datetime import date

from app.models.plan import DateRange, Filters, QueryPlan

STATUS_LABELS = {
    "active": "Active",
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
    "wing": "Wing",
    "gender": "Gender",
    "primary_caregiver": "Caregiver",
}
ATTRIBUTE_LABELS = {
    "primary_caregiver": "Primary Caregiver",
    "room_number": "Room",
    "wing": "Wing",
    "admission_date": "Admission Date",
    "created_date": "Registration Date",
    "discharge_date": "Discharge Date",
    "emergency_contact": "Emergency Contact",
    "status": "Status",
    "care_type": "Care Type",
    "age": "Age",
    "gender": "Gender",
}
_SET_FIELDS = [
    ("status", "Status"),
    ("care_type", "Care Type"),
    ("wing", "Wing"),
    ("gender", "Gender"),
    ("primary_caregiver", "Caregiver"),
    ("room_number", "Room"),
]


def label_value(field_name: str, value: str) -> str:
    if field_name == "status":
        return STATUS_LABELS.get(value, value)
    if field_name == "care_type":
        return CARE_TYPE_LABELS.get(value, value)
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
        if sf.include:
            chips.append(f"{label}: {' or '.join(label_value(field_name, v) for v in sf.include)}")
        if sf.exclude:
            chips.append(f"{label}: not {' / '.join(label_value(field_name, v) for v in sf.exclude)}")
    if f.age:
        lo, hi = f.age.min, f.age.max
        if lo is not None and hi is not None:
            chips.append(f"Age: {lo}" if lo == hi else f"Age: {lo}–{hi}")
        elif lo is not None:
            chips.append(f"Age: {lo}+")
        else:
            chips.append(f"Age: {hi} or younger")
    if f.admission_date:
        chips.append(f"Admitted: {_range_text(f.admission_date)}")
    if f.created_date:
        chips.append(f"Registered: {_range_text(f.created_date)}")
    if f.discharge_date:
        chips.append(f"Discharged: {_range_text(f.discharge_date)}")
    return chips


def describe_plan(plan: QueryPlan, name_of: Callable[[str], str]) -> list[str]:
    chips: list[str] = []
    match plan.action:
        case "count":
            chips.append("Count")
        case "list":
            chips.append("List")
        case "breakdown":
            chips.append("Breakdown by " + " × ".join(GROUP_LABELS[g] for g in plan.group_by))
        case "aggregate":
            op = {"avg": "Average", "max": "Highest", "min": "Lowest"}[plan.aggregate.op]
            chips.append(
                f"{op} age" + (" by " + " × ".join(GROUP_LABELS[g] for g in plan.group_by) if plan.group_by else "")
            )
        case "lookup":
            chips.extend(f"Resident: {name_of(i)} ({i})" for i in plan.lookup.residents)
            chips.extend(f"Asked: {ATTRIBUTE_LABELS.get(a, a)}" for a in plan.lookup.attributes)
    chips.extend(describe_filters(plan.filters))
    if plan.sort:
        s = plan.sort
        if s.field == "age":
            what = "Oldest first" if s.direction == "desc" else "Youngest first"
        else:
            what = f"{'Most recent' if s.direction == 'desc' else 'Earliest'} {ATTRIBUTE_LABELS.get(s.field, s.field).lower()}"
        chips.append(f"{what} · top {plan.limit}" if plan.limit else what)
    return chips
