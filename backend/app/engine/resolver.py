"""Turns Claude's raw plan into a validated QueryPlan with exact database values.

Anything Claude wrote that does not exist in the data (a wing, caregiver, room or resident) becomes a
clarifying question — never a guess and never a silently dropped condition.
"""

import copy
import re
from dataclasses import dataclass

from pydantic import ValidationError

from app.models.plan import QueryPlan
from app.nlu.prompt import conditions_to_filters
from app.nlu.vocabulary import EntityIndex, MaskResult, Resolution


@dataclass
class ResolveOutcome:
    plan: QueryPlan
    problem: str | None = None  # set when the plan was replaced by a clarification


def _clean(value):
    """Drop empty values models sometimes emit (empty lists/objects/strings, nulls)."""
    if isinstance(value, dict):
        out = {k: _clean(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}
    if isinstance(value, list):
        return [x for x in (_clean(v) for v in value) if x not in (None, "", [], {})]
    return value


def _clarify(message: str, options: list[dict] | None = None) -> QueryPlan:
    return QueryPlan(action="clarify", message=message, options=options or None)


def resolve_plan(raw: dict, index: EntityIndex, mask: MaskResult, question: str) -> ResolveOutcome:
    plan = _clean(copy.deepcopy(raw)) if isinstance(raw, dict) else {}
    # The model returns an order-free list of conditions; convert it to filters (older cached plans use filters).
    if isinstance(plan.get("conditions"), list):
        converted = conditions_to_filters(plan.pop("conditions"))
        plan["filters"] = {**converted, **plan.get("filters", {})}
    plan.setdefault("filters", {})
    plan["filters"] = {k: v for k, v in plan["filters"].items() if v not in ({}, [], None)}
    unknown: list[str] = []

    resolvers = {
        "wing": index.resolve_wing,
        "primary_caregiver": index.resolve_caregiver,
        "room_number": index.resolve_room,
    }
    for field_name, resolve in resolvers.items():
        f = plan["filters"].get(field_name)
        if not isinstance(f, dict):
            continue
        for bucket in ("include", "exclude"):
            values: list[str] = []
            for text in f.get(bucket, []) or []:
                res: Resolution = resolve(str(text))
                if not res.values:
                    unknown.append(str(text))
                elif res.ambiguous:
                    label = field_name.replace("_", " ")
                    return ResolveOutcome(
                        _clarify(
                            f'"{text}" matches more than one {label}. Which one do you mean?',
                            [{"label": v, "query": _replace(question, str(text), v)} for v in res.values[:6]],
                        ),
                        problem="ambiguous",
                    )
                else:
                    values.extend(res.values)
            if values:
                f[bucket] = list(dict.fromkeys(values))
            else:
                f.pop(bucket, None)
        if not f:
            plan["filters"].pop(field_name)

    lookup = plan.get("lookup")
    if isinstance(lookup, dict):
        ids: list[str] = []
        for ref in lookup.get("residents", []) or []:
            res = index.resolve_resident(str(ref), mask.placeholders)
            if not res.values:
                unknown.append(mask.originals.get(str(ref).lower(), str(ref)))
            elif res.ambiguous:
                original = mask.originals.get(str(ref).lower(), str(ref))
                return ResolveOutcome(
                    _clarify(
                        f'"{original}" matches {len(res.values)} residents. Which one do you mean?',
                        [
                            {
                                "label": f"{index.display_by_id.get(i, i)} ({i})",
                                "query": _replace(question, original, index.display_by_id.get(i, i)),
                            }
                            for i in res.values[:6]
                        ],
                    ),
                    problem="ambiguous",
                )
            else:
                ids.extend(res.values)
        lookup["residents"] = list(dict.fromkeys(ids))
        if not lookup["residents"] and not unknown:
            unknown.append("the resident mentioned")

    if unknown:
        names = ", ".join(f'"{u}"' for u in dict.fromkeys(unknown))
        return ResolveOutcome(
            _clarify(f"I couldn't find {names} in the resident records. Please check the spelling."), problem="unknown"
        )

    try:
        return ResolveOutcome(QueryPlan.model_validate(plan))
    except ValidationError as exc:
        errors = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:3])
        return ResolveOutcome(
            _clarify("I couldn't turn that into a safe query. Could you rephrase it?"), problem=f"invalid: {errors}"
        )


def _replace(question: str, original: str, replacement: str) -> str:
    pattern = re.compile(re.escape(original), re.IGNORECASE)
    return pattern.sub(replacement, question, count=1) if pattern.search(question) else f"Tell me about {replacement}"
