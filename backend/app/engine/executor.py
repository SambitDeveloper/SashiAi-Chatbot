"""Runs a validated QueryPlan against the repository. Pure data access: no wording, no guessing."""

from dataclasses import dataclass, field

from app.data.repository import GroupAggregateRow, GroupRow, Resident, ResidentRepository, list_view, matches
from app.data.schema_config import output_fields
from app.models.plan import QueryPlan

SAMPLE_SIZE = 25


@dataclass
class ExecutionData:
    total: int
    matched: int = 0
    rows: list[Resident] = field(default_factory=list)
    groups: list[GroupRow] | None = None
    group_values: list[GroupAggregateRow] | None = None  # grouped aggregate: count + value per group
    aggregate_value: float | None = None
    looked_up: list[Resident] | None = None
    checks: list[bool] | None = None  # lookup + filters → does each resident satisfy the filters?


def projected_view_fields(plan: QueryPlan) -> list[str] | None:
    """The view fields a list should return (schema `output_fields`), or None for the default result shape."""
    if plan.action != "list" or not plan.output_fields:
        return None
    mapping = output_fields()
    return [mapping[a] for a in plan.output_fields if a in mapping]


async def execute(plan: QueryPlan, repo: ResidentRepository) -> ExecutionData:
    total = await repo.total()
    f = plan.filters

    if plan.action == "lookup":
        ids = plan.lookup.residents
        found = await repo.find_by_ids(ids)
        order = {rid.upper(): i for i, rid in enumerate(ids)}
        found.sort(key=lambda r: order.get(r["residentId"].upper(), 0))
        checks = [matches(r, f) for r in found] if not f.is_empty() else None
        return ExecutionData(total=total, matched=len(found), rows=found, looked_up=found, checks=checks)

    if plan.action == "breakdown":
        groups = await repo.group_count(f, plan.group_by)
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
