"""C — the plan policy, and the request's principal.

`check_plan` runs right before EVERY execution (app/engine/executor.py): fast path, template, follow-up and model plans
all pass through it, so no path can skip it. It derives the capabilities a plan needs from its entity, filters, shown
fields, grouping, sorting and aggregate (access_policy.yaml) and raises AccessDenied when the caller lacks one.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from app.auth.policy import AccessPolicy, access_policy
from app.auth.principal import SYSTEM, Principal
from app.models.plan import STAFF_ATTRIBUTES, STAFF_ENTITY_SET_FIELDS, STAFF_SORT_FIELDS, Filters, QueryPlan

_PRINCIPAL: ContextVar[Principal | None] = ContextVar("principal", default=None)
# a staff plan's own fields (the rest of a staff plan are conditions on the staff member's assigned residents)
_STAFF_OWN = frozenset({*STAFF_ENTITY_SET_FIELDS, *STAFF_ATTRIBUTES, *STAFF_SORT_FIELDS, "staff_resident_count"})
# staff facts that read residents (counted / named within the caller's own resident scope)
_STAFF_READS_RESIDENTS = frozenset({"staff_resident_count", "staff_residents"})


class AccessDenied(Exception):
    """The caller lacks a capability. `layer`: gate (before any model call) or plan (right before execution)."""

    def __init__(self, capability: str, layer: str, policy: AccessPolicy | None = None):
        p = policy or access_policy()
        self.capability = capability
        self.label = p.capabilities[capability].label
        self.layer = layer
        self.message = p.message("module_denied", label=self.label)
        super().__init__(self.message)


@contextmanager
def use_principal(principal: Principal) -> Iterator[None]:
    token = _PRINCIPAL.set(principal)
    try:
        yield
    finally:
        _PRINCIPAL.reset(token)


def current_principal() -> Principal:
    return _PRINCIPAL.get() or SYSTEM


def plan_fields(plan: QueryPlan) -> set[str]:
    f = plan.filters
    used = {n for n in Filters.model_fields if getattr(f, n) is not None}
    used |= set(plan.output_fields) | set(plan.group_by or [])
    if plan.sort is not None and getattr(plan.sort, "field", None):
        used.add(plan.sort.field)
    agg = getattr(plan, "aggregate", None)
    if agg is not None and getattr(agg, "field", None):
        used.add(agg.field)
    return used


def required_capabilities(plan: QueryPlan, policy: AccessPolicy | None = None) -> list[str]:
    """Every capability the plan needs, in a stable order (the first missing one is named to the user)."""
    p = policy or access_policy()
    if not plan.is_data_query:
        return []
    fields = plan_fields(plan)
    needed: list[str] = []
    if plan.entity == "staff":
        needed.append(p.base_capability("staff"))
        resident_fields = {n for n in fields if n not in _STAFF_OWN}
        if resident_fields or fields & _STAFF_READS_RESIDENTS:
            needed.append(p.base_capability("resident"))
        needed += [p.capability_for_field(n, "resident") for n in sorted(resident_fields)]
    else:
        needed.append(p.base_capability("resident"))
        needed += p.capabilities_for_action(plan.action)  # e.g. a lookup of one resident = resident details
        needed += [p.capability_for_field(n, "resident") for n in sorted(fields)]
    return list(dict.fromkeys(needed))


def require(capabilities, layer: str, principal: Principal | None = None, policy: AccessPolicy | None = None) -> None:
    p = policy or access_policy()
    who = principal or current_principal()
    for cap in capabilities:
        if not who.can(cap, p):
            raise AccessDenied(cap, layer, p)


def check_plan(plan: QueryPlan, principal: Principal | None = None) -> None:
    """C: raises AccessDenied when the plan needs a capability the caller does not have."""
    require(required_capabilities(plan), "plan", principal)
