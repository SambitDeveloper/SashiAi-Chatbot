"""Runs a validated QueryPlan against the repository. Pure data access: no wording, no guessing."""

from dataclasses import dataclass, field
from itertools import product

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
from app.engine.caller import current_caller
from app.models.plan import QueryPlan

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


def projected_view_fields(plan: QueryPlan) -> list[str] | None:
    """The view fields requested for resident rows, or None for the default result shape."""
    if plan.action not in ("list", "breakdown") or not plan.output_fields:
        return None
    mapping = output_fields()
    return [mapping[a] for a in plan.output_fields if a in mapping]


async def execute(plan: QueryPlan, repo: ResidentRepository) -> ExecutionData:
    if not plan.is_data_query:
        return ExecutionData(total=0)
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
        candidates = [group_key(value, FIELD_MAP[field_name]) for value in condition.include or []]
        if condition.missing is True:
            candidates.append(MISSING_GROUP)
        values.append(list(dict.fromkeys(candidates)))
    existing = {group.key for group in groups}
    for key in product(*values):
        if key not in existing:
            groups.append(GroupRow(key=key, count=0))
    return groups
