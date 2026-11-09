"""Runs a validated QueryPlan against the repository. Pure data access: no wording, no guessing."""

from dataclasses import dataclass, field

from app.data.repository import GroupRow, Resident, ResidentRepository, list_view, matches
from app.models.plan import Filters, QueryPlan, SetFilter

SAMPLE_SIZE = 25


@dataclass
class ExecutionData:
    total: int
    matched: int = 0
    rows: list[Resident] = field(default_factory=list)
    groups: list[GroupRow] | None = None
    group_values: list[tuple[GroupRow, float | None]] | None = None
    aggregate_value: float | None = None
    looked_up: list[Resident] | None = None
    checks: list[bool] | None = None  # lookup + filters → does each resident satisfy the filters?


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
            groups = await repo.group_count(f, plan.group_by)
            values = []
            for g in groups:
                gf = f.model_copy(deep=True)
                for field_name, key in zip(plan.group_by, g.key, strict=True):
                    setattr(gf, field_name, SetFilter[str](include=[key]))
                values.append(
                    (
                        g,
                        await repo.aggregate(
                            Filters.model_validate(gf.model_dump(exclude_none=True)), plan.aggregate.op, "age"
                        ),
                    )
                )
            return ExecutionData(total=total, matched=matched, group_values=values)
        value = await repo.aggregate(f, plan.aggregate.op, plan.aggregate.field)
        return ExecutionData(total=total, matched=matched, aggregate_value=value)

    if plan.action in ("count", "list"):
        matched = await repo.count(f)
        rows = [list_view(r) for r in await repo.find(f, sort=plan.sort, limit=plan.limit or SAMPLE_SIZE)]
        return ExecutionData(total=total, matched=matched, rows=rows)

    return ExecutionData(total=total)
