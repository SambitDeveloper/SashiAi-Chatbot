"""Turns a planner's raw plan (deterministic or local model) into a validated QueryPlan with exact database values.

Anything the planner wrote that does not exist in the data (a wing, caregiver, room or resident) becomes a
clarifying question — never a guess and never a silently dropped condition.
"""

import copy
import re
from dataclasses import dataclass

from pydantic import ValidationError

from app.models.plan import QueryPlan, canonical_field
from app.nlu.prompt import conditions_to_filters
from app.nlu.vocabulary import EntityIndex, MaskResult, Resolution


@dataclass
class ResolveOutcome:
    plan: QueryPlan
    problem: str | None = None  # set when the plan was replaced by a clarification
    hint: str | None = None  # feedback for a model retry, when the problem is a planning error a model can fix


def _resident_reference(text: str, mask: MaskResult) -> str | None:
    """The resident `text` refers to (as the user wrote it), when a planner put a RESIDENT where a caregiver, facility
    or room belongs: a resident_N placeholder, or the name of a resident the question mentions. None otherwise."""
    t = str(text).strip().rstrip(".,;:!?").lower()
    if re.fullmatch(r"resident_\d+", t):
        return mask.originals.get(t, t)
    for original in mask.originals.values():
        if t == original.strip().rstrip(".,;:!?").lower():
            return original
    return None


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


def _reference_was_mentioned(ref: str, question: str, mask: MaskResult) -> bool:
    value = ref.strip()
    lowered = value.lower()
    if re.fullmatch(r"resident_\d+", lowered):
        return lowered in mask.placeholders
    mentioned_ids = {rid.upper() for ids in mask.placeholders.values() for rid in ids}
    if value.upper() in mentioned_ids:
        return True
    return bool(re.search(r"(?<![\w])" + re.escape(value) + r"(?![\w])", question, re.IGNORECASE))


def resolve_plan(raw: dict, index: EntityIndex, mask: MaskResult, question: str) -> ResolveOutcome:
    plan = _clean(copy.deepcopy(raw)) if isinstance(raw, dict) else {}
    # The model returns an order-free list of conditions; convert it to filters (older cached plans use filters).
    if isinstance(plan.get("conditions"), list):
        converted = conditions_to_filters(plan.pop("conditions"))
        plan["filters"] = {**converted, **plan.get("filters", {})}
    plan.setdefault("filters", {})
    if not isinstance(plan["filters"], dict):
        plan["filters"] = {"_invalid": plan["filters"]}
    # legacy field names ("wing") → the canonical name BEFORE values are resolved, so an old-style plan is checked
    # against the database exactly like a new one
    if any(canonical_field(k) != k and canonical_field(k) in plan["filters"] for k in plan["filters"]):  # both names
        return ResolveOutcome(
            _clarify("I couldn't turn that into a safe query. Could you rephrase it?"),
            problem="invalid: duplicate field names",
        )
    plan["filters"] = {canonical_field(k): v for k, v in plan["filters"].items() if v not in ({}, [], None)}
    unknown: list[str] = []
    misused: list[tuple[str, str, str]] = []  # (field, resident as written, value the planner wrote)

    resolvers = {
        "facility": index.resolve_wing,
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
                if not res.values and (who := _resident_reference(str(text), mask)) is not None:
                    misused.append((field_name, who, str(text)))
                elif not res.values:
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
    if isinstance(lookup, dict) and isinstance(lookup.get("residents"), list):
        invented = [
            str(ref)
            for ref in lookup["residents"]
            if not _reference_was_mentioned(str(ref), question, mask)
        ]
        if invented:
            return ResolveOutcome(
                _clarify("I couldn't safely identify the resident reference in that plan. Please rephrase the question."),
                problem="invented:resident_reference",
                hint="Only look up resident placeholders or identifiers explicitly present in the question. Do not invent resident references.",
            )
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

    if misused:
        # A planning error, not a missing record: the resident exists, but was used as a caregiver/facility/room.
        field_name, who, written = misused[0]
        label = field_name.replace("_", " ").replace("primary ", "")
        return ResolveOutcome(
            _clarify(
                f"I couldn't work out how {who} relates to the {label} in your question. Please rephrase it, for "
                f"example: 'residents whose caregiver is <caregiver name>' or 'what is {who}'s {label}?'."
            ),
            problem=f"role:{field_name}",
            hint=" ".join(
                f"'{w}' is a resident, not a {f.replace('_', ' ')}: never use a resident as a {f.replace('_', ' ')} value."
                for f, _, w in misused
            ),
        )

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
