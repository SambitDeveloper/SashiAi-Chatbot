"""Runs a validated QueryPlan against the repository. Pure data access: no wording, no guessing."""

from dataclasses import dataclass, field
from itertools import product

from app.auth.guard import check_plan as authorize_plan
from app.data.relations import FAMILY_SELECTORS
from app.data.repository import (
    FIELD_MAP,
    MISSING_GROUP,
    GroupAggregateRow,
    GroupRow,
    Resident,
    ResidentRepository,
    group_key,
    group_keys,
    list_view,
    matches,
    project,
)
from app.data.schema_config import output_fields
from app.data.staff import staff_matches
from app.engine.caller import current_caller
from app.models.plan import STAFF_FIELD_MAP, Filters, QueryPlan

# staff filters that, on a ONE-resident lookup showing the care team, select which staff members are shown
STAFF_SELECTORS = ("staff_role", "staff_designation")

SAMPLE_SIZE = 25


@dataclass
class ExecutionData:
    total: int
    matched: int = 0
    rows: list[Resident] = field(default_factory=list)
    groups: list[GroupRow] | None = None
    group_residents: dict[tuple[str, ...], list[Resident]] | None = None
    group_details_complete: bool | None = None
    group_values: list[GroupAggregateRow] | None = None  # grouped aggregate: count + value per group
    aggregate_value: float | None = None
    looked_up: list[Resident] | None = None
    checks: list[bool] | None = None  # lookup + filters → does each resident satisfy the filters?
    # staff plans (plan.entity == "staff"): staff rows instead of resident rows
    staff_rows: list[dict] | None = None
    staff_default_active: bool = False  # no staff_status condition was given: active staff only
    staff_contacts_withheld: bool = False  # contact details asked for several staff members: not shown


def projected_view_fields(plan: QueryPlan) -> list[str] | None:
    """The view fields requested for resident rows, or None for the default result shape."""
    if plan.action not in ("list", "breakdown") or not plan.output_fields:
        return None
    mapping = output_fields()
    return [mapping[a] for a in plan.output_fields if a in mapping]


async def execute(plan: QueryPlan, repo: ResidentRepository) -> ExecutionData:
    if not plan.is_data_query:
        return ExecutionData(total=0)
    # C (app/auth/guard.py): every plan — fast path, template, follow-up, model — needs only what the caller may read;
    # raises AccessDenied. The rows themselves are limited by the request's row scope (app/data/scope.py).
    authorize_plan(plan)
    if hasattr(repo, "for_plan"):  # snf_mcp: admins → run_query; staff → the API tool for this plan (or not supported)
        repo = await repo.for_plan(plan)
    if plan.entity == "staff":
        return await _execute_staff(plan, repo)
    total = await repo.total()
    f = plan.filters

    if plan.action == "lookup":
        ids = plan.lookup.residents
        caller = current_caller()
        shown = set(plan.output_fields)
        # family phone / email are read only when the answer shows them AND the caller may see them
        contacts = caller.can_read_family_contacts and bool(shown & {"family_members", "emergency_contact"})
        found = await repo.find_by_ids(ids, contacts=contacts)
        order = {rid.upper(): i for i, rid in enumerate(ids)}
        found.sort(key=lambda r: order.get(r["residentId"].upper(), 0))
        # family filters on a lookup that SHOWS family select which members are shown; every other filter (and family
        # filters when no family detail is asked: "is X a family member of R?") stays the yes/no check
        select = {n: getattr(f, n) for n in FAMILY_SELECTORS if getattr(f, n) is not None}
        if select and shown & {"family_members", "family_relationship"}:
            found = [{**r, "familyContacts": [c for c in r.get("familyContacts") or [] if _member_selected(c, select)]}
                     for r in found]  # fmt: skip
            f = f.model_copy(update=dict.fromkeys(select))
        # the same for the care team: "resident_1's physician" shows only the physicians (active or not, marked)
        team_select = {n: getattr(f, n) for n in STAFF_SELECTORS if getattr(f, n) is not None}
        if team_select and "care_team" in shown:
            found = [_select_team(r, team_select) for r in found]
            f = f.model_copy(update=dict.fromkeys(team_select))
        checks = [matches(r, f) for r in found] if not f.is_empty() else None
        return ExecutionData(total=total, matched=len(found), rows=found, looked_up=found, checks=checks)

    if plan.action == "breakdown":
        groups = await repo.group_count(f, plan.group_by)
        groups = _include_empty_groups(groups, plan)
        projection = projected_view_fields(plan)
        if projection is not None:
            group_fields = [FIELD_MAP[name] for name in plan.group_by]
            fetch_fields = list(dict.fromkeys([*projection, *group_fields]))
            found = await repo.find(f, fields=fetch_fields)
            members: dict[tuple[str, ...], list[Resident]] = {g.key: [] for g in groups}
            membership_count = 0
            for resident in found:
                visible = project(resident, projection)
                for key in group_keys(resident, plan.group_by):
                    members.setdefault(key, []).append(visible)
                    membership_count += 1
            expected_memberships = sum(g.count for g in groups)
            return ExecutionData(
                total=total,
                matched=sum(g.count for g in groups),
                groups=groups,
                group_residents=members,
                group_details_complete=membership_count == expected_memberships,
            )
        rows = [list_view(r) for r in await repo.find(f, limit=SAMPLE_SIZE)]
        return ExecutionData(total=total, matched=sum(g.count for g in groups), rows=rows, groups=groups)

    if plan.action == "aggregate":
        matched = await repo.count(f)
        if plan.group_by:
            # One grouped query: every group's count and aggregate together. A group key is never turned back into
            # a filter, so a "not recorded" or unexpected value (status "unknown", no facility) cannot fail
            # validation or silently match nothing (see repository.MISSING_GROUP).
            rows = await repo.group_aggregate(f, plan.group_by, plan.aggregate.op, plan.aggregate.field)
            return ExecutionData(total=total, matched=matched, group_values=rows)
        value = await repo.aggregate(f, plan.aggregate.op, plan.aggregate.field)
        return ExecutionData(total=total, matched=matched, aggregate_value=value)

    if plan.action == "count":
        # A count returns only the number: no resident documents, caregivers, contacts or samples are fetched.
        return ExecutionData(total=total, matched=await repo.count(f))

    if plan.action == "list":
        matched = await repo.count(f)
        projection = projected_view_fields(plan)
        found = await repo.find(f, sort=plan.sort, limit=plan.limit or SAMPLE_SIZE, fields=projection)
        # a projection returns only identity + the requested fields; the default shape keeps list_view's privacy rule
        rows = found if projection is not None else [list_view(r) for r in found]
        return ExecutionData(total=total, matched=matched, rows=rows)

    return ExecutionData(total=total)


def _select_team(r: Resident, select: dict) -> Resident:
    members = r.get("careTeamMembers") or []
    labels = r.get("careTeam") or []
    keep = [i for i, m in enumerate(members) if _staff_selected(m, select)]
    return {**r, "careTeamMembers": [members[i] for i in keep],
            "careTeam": [labels[i] for i in keep if i < len(labels)]}  # fmt: skip


def _staff_selected(member: dict, select: dict) -> bool:
    for name, cond in select.items():
        value = member.get(STAFF_FIELD_MAP[name])
        if (cond.include and value not in cond.include) or (cond.exclude and value in cond.exclude):
            return False
        if cond.has_text_match and not staff_matches(member, Filters(**{name: cond})):
            return False
    return True


async def _execute_staff(plan: QueryPlan, repo: ResidentRepository) -> ExecutionData:
    """A staff plan (app/data/staff.py): the same actions over the staff view. Without a staff_status condition only
    active staff are counted and listed; a lookup by name shows the person whatever their status."""
    total = await repo.staff_total()
    f = plan.filters
    default_active = f.staff_status is None and plan.action != "lookup"
    if default_active:
        f = Filters.model_validate({**f.model_dump(exclude_none=True), "staff_status": {"include": ["active"]}})
    shown = plan.output_fields

    if plan.action == "lookup":
        names = list(plan.lookup.staff)
        rows = await repo.staff_by_names(names)
        withheld = False
        if set(shown) & {"staff_email", "staff_phone"}:
            if len(rows) == 1:  # contact details for exactly one staff member
                rows = await repo.staff_by_names(names, contacts=True)
            else:
                withheld = True
        checks = None
        if not f.is_empty():  # "is Priya Raman a physician?" — the plan's conditions, checked per person
            checks = []
            for r in rows:
                one = Filters.model_validate({**f.model_dump(exclude_none=True), "staff_name": {"include": [r["name"]]}})
                checks.append(await repo.staff_count(one) > 0)
        return ExecutionData(total=total, matched=len(rows), staff_rows=rows, checks=checks, staff_contacts_withheld=withheld)

    if plan.action == "breakdown":
        groups = await repo.staff_group_count(f, list(plan.group_by))
        groups = _include_empty_groups(groups, plan)
        return ExecutionData(total=total, matched=sum(g.count for g in groups), groups=groups, staff_default_active=default_active)

    matched = await repo.staff_count(f)
    if plan.action == "count":
        return ExecutionData(total=total, matched=matched, staff_default_active=default_active)
    fields = [STAFF_FIELD_MAP[a] for a in shown]
    rows = await repo.staff_find(f, sort=plan.sort, limit=plan.limit or SAMPLE_SIZE, fields=fields)
    return ExecutionData(total=total, matched=matched, staff_rows=rows, staff_default_active=default_active)


def _member_selected(contact: dict, select: dict) -> bool:
    """One family member against the lookup's family selectors (same normalised values as the filters)."""
    for name, cond in select.items():
        value = {"family_member": contact.get("id"), "family_relation": contact.get("rel"), "family_type": contact.get("type")}[name]
        if cond.include and value not in cond.include:
            return False
        if cond.exclude and value in cond.exclude:
            return False
    return True


def _include_empty_groups(groups: list[GroupRow], plan: QueryPlan) -> list[GroupRow]:
    """Represent explicitly requested group values with zero counts instead of omitting them."""
    values = []
    for field_name in plan.group_by:
        condition = getattr(plan.filters, field_name)
        if condition is None or (not condition.include and condition.missing is not True):
            return groups
        candidates = [group_key(value, FIELD_MAP.get(field_name, field_name)) for value in condition.include or []]
        if condition.missing is True:
            candidates.append(MISSING_GROUP)
        values.append(list(dict.fromkeys(candidates)))
    existing = {group.key for group in groups}
    for key in product(*values):
        if key not in existing:
            groups.append(GroupRow(key=key, count=0))
    return groups
