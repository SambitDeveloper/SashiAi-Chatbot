"""Turns a planner's raw plan (deterministic or local model) into a validated QueryPlan with exact database values.

Anything the planner wrote that does not exist in the data (a wing, caregiver, room or resident) becomes a
clarifying question — never a guess and never a silently dropped condition.
"""

import copy
import re
from dataclasses import dataclass

from pydantic import ValidationError

from app.data.relations import FAMILY_ATTRIBUTES, FAMILY_PLAN_FIELDS, family_relation
from app.data.staff import staff_config
from app.engine.caller import current_caller
from app.models.plan import QueryPlan, canonical_field
from app.nlu import placeholders as ph
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

    # ---------------------------------------------------------------- family members as people (family_N / person_N)
    caller = current_caller()
    attrs = [*(plan.get("fields") or []), *((plan.get("lookup") or {}).get("attributes") or [])]
    uses_family = any(k in FAMILY_PLAN_FIELDS for k in plan["filters"]) or any(a in FAMILY_ATTRIBUTES for a in attrs)
    if uses_family and not caller.can_read_family:
        return ResolveOutcome(_clarify("You don't have access to family information."), problem="forbidden:family")
    people = {**mask.family, **mask.persons} if hasattr(mask, "family") else {}
    if people:
        written = _placeholders_written(plan)
        unused = [p for p in people if p not in written]
        if unused:  # a person the question names is never silently dropped
            who = mask.entity_originals.get(unused[0], unused[0])
            return ResolveOutcome(
                _clarify(f'I couldn\'t tell how "{who}" relates to your question. Could you rephrase it?'),
                problem="unused:person",
                hint=" ".join(f"The question names {p}: use it (family_member eq {p}, or a person_N as a primary_caregiver"
                              " value when the question is about a caregiver); never drop it." for p in unused),
            )  # fmt: skip
        caregiver = plan["filters"].get("primary_caregiver")
        if isinstance(caregiver, dict):
            for bucket in ("include", "exclude"):
                values: list[str] = []
                for text in caregiver.get(bucket, []) or []:
                    kind = ph.kind_of(str(text))
                    if kind == ph.FAMILY:  # a family member is never turned into a caregiver condition
                        who = mask.entity_originals.get(str(text).lower(), str(text))
                        return ResolveOutcome(
                            _clarify(f'"{who}" is recorded as a family member, not as a caregiver. Could you rephrase it?'),
                            problem="role:family_as_caregiver",
                            hint=f"{text} is a family member, not a caregiver: use family_member eq {text}.",
                        )
                    if kind == ph.PERSON:  # the staff reading of a name that is also a family member's
                        staff = mask.persons.get(str(text).lower(), {}).get("staff") or []
                        if not staff:
                            who = mask.entity_originals.get(str(text).lower(), str(text))
                            return ResolveOutcome(
                                _clarify(f'"{who}" is not a staff member in the records. Could you rephrase it?'),
                                problem="role:person_not_staff",
                                hint=f"{text} is not a staff member: use family_member eq {text}.",
                            )
                        values.extend(staff)
                    else:
                        values.append(text)
                if values:
                    caregiver[bucket] = list(dict.fromkeys(values))
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

    # family relation words → the normalised relation values they mean ("spouse" → spouse, wife, husband;
    # "daughters" → daughter). Not looked up as names: a relation nobody has recorded simply matches no one.
    fr = plan["filters"].get("family_relation")
    rel = family_relation()
    if isinstance(fr, dict) and rel is not None:
        for bucket in ("include", "exclude"):
            values = [v for text in fr.get(bucket, []) or [] for v in rel.expand(str(text))]
            if values:
                fr[bucket] = list(dict.fromkeys(values))
            else:
                fr.pop(bucket, None)
        if not fr:
            plan["filters"].pop("family_relation")

    # ---------------------------------------------------------------- staff (app/data/staff.py)
    problem = _resolve_staff(plan, index, mask, question)
    if problem is not None:
        return problem

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
        if not lookup["residents"] and not unknown and not lookup.get("staff"):
            unknown.append("the resident mentioned")

    fm = plan["filters"].get("family_member")
    if isinstance(fm, dict):
        scope_ids = {str(r).upper() for r in ((lookup or {}).get("residents") or [])} if isinstance(lookup, dict) else set()
        ids: list[str] = []
        for text in fm.get("include", []) or []:
            key = str(text).lower()
            who = mask.entity_originals.get(key, str(text))
            cands = (mask.family.get(key) or (mask.persons.get(key) or {}).get("family") or []) if people else []
            cands = [c for c in cands if caller.allows_facility(c.get("facilityId"))]  # authorised facilities only
            if scope_ids:
                cands = [c for c in cands if str(c.get("residentId", "")).upper() in scope_ids]
            if not cands:  # never says whether the name exists for someone the caller cannot see
                whose = " and ".join(index.display_by_id.get(i, i) for i in sorted(scope_ids)) if scope_ids else ""
                msg = (f'No family member named "{who}" is recorded for {whose}.' if whose
                       else f'No family member named "{who}" is recorded.')  # fmt: skip
                return ResolveOutcome(_clarify(msg), problem="not_found:family_member")
            residents = list(dict.fromkeys(str(c["residentId"]).upper() for c in cands))
            if len(cands) > 1:
                if len(residents) > 1:  # e.g. the same name in two residents' families: ask, authorised residents only
                    return ResolveOutcome(
                        _clarify(f'"{who}" is a family member of more than one resident. Which resident do you mean?',
                                 [{"label": index.display_by_id.get(r, r),
                                   "query": f"What is the relationship between {who} and {index.display_by_id.get(r, r)}?"}
                                  for r in residents[:6]]),
                        problem="ambiguous",
                    )  # fmt: skip
                return ResolveOutcome(
                    _clarify(f'{index.display_by_id.get(residents[0], residents[0])} has more than one family contact named '
                             f'"{who}". Please check the family records.'),
                    problem="ambiguous",
                )  # fmt: skip
            ids.append(str(cands[0]["id"]))
        fm["include"] = list(dict.fromkeys(ids))

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


def _placeholders_written(plan: dict) -> set[str]:
    """family_N / person_N placeholders the plan uses (family_member or caregiver values), lower case."""
    out: set[str] = set()
    for name in ("family_member", "primary_caregiver", "staff_name"):
        f = plan.get("filters", {}).get(name)
        if isinstance(f, dict):
            out |= {str(v).lower() for b in ("include", "exclude") for v in f.get(b, []) or [] if ph.is_placeholder(str(v))}
    lookup = plan.get("lookup")
    if isinstance(lookup, dict):
        out |= {str(v).lower() for v in lookup.get("staff") or [] if ph.is_placeholder(str(v))}
    return out


def _staff_people(text: str, index: EntityIndex, mask: MaskResult) -> tuple[Resolution, str]:
    """A staff member as the planner wrote them: a name, or person_N (a name that is also a family member's)."""
    key = str(text).lower()
    if ph.kind_of(key) == ph.PERSON:
        staff = (getattr(mask, "persons", {}).get(key) or {}).get("staff") or []
        return Resolution(list(staff)), mask.entity_originals.get(key, str(text))
    if ph.kind_of(key) is not None:  # resident_N / family_N are never staff
        return Resolution([]), mask.all_originals.get(key, str(text))
    return index.resolve_staff(str(text)), str(text)


def _resolve_staff(plan: dict, index: EntityIndex, mask: MaskResult, question: str) -> "ResolveOutcome | None":
    """Role words → canonical roles, designations → as recorded, staff names → as recorded (any status). A role word
    the configuration does not know but that is a recorded designation ("Manager") becomes a designation condition, and
    the reverse; anything else asks the user — never a guess."""
    cfg = staff_config()
    filters = plan["filters"]
    if cfg is None:
        return None
    unknown: list[str] = []
    moved: dict[str, dict[str, list[str]]] = {"staff_role": {}, "staff_designation": {}}
    for field_name in ("staff_role", "staff_designation"):
        f = filters.get(field_name)
        if not isinstance(f, dict):
            continue
        for bucket in ("include", "exclude"):
            values: list[str] = []
            for text in f.get(bucket, []) or []:
                roles = cfg.expand(str(text))
                designations = index.resolve_designation(str(text)).values
                if field_name == "staff_role" and roles:
                    values.extend(roles)
                elif field_name == "staff_designation" and designations:
                    values.extend(designations)
                elif roles:  # a role word written as a designation ("doctors")
                    moved["staff_role"].setdefault(bucket, []).extend(roles)
                elif designations:  # a designation without a role ("Manager")
                    moved["staff_designation"].setdefault(bucket, []).extend(designations)
                else:
                    unknown.append(str(text))
            if values:
                f[bucket] = list(dict.fromkeys(values))
            else:
                f.pop(bucket, None)
        if not f:
            filters.pop(field_name)
    for field_name, buckets in moved.items():
        for bucket, values in buckets.items():
            target = filters.setdefault(field_name, {})
            target[bucket] = list(dict.fromkeys([*(target.get(bucket) or []), *values]))
    if unknown:
        words = ", ".join(f'"{u}"' for u in dict.fromkeys(unknown))
        return ResolveOutcome(
            _clarify(f"I don't know the staff role or designation {words}. Please use a designation from the staff records."),
            problem="unknown:staff_role",
        )

    def names(texts: list, label: str) -> tuple[list[str], "ResolveOutcome | None"]:
        out: list[str] = []
        for text in texts:
            res, who = _staff_people(str(text), index, mask)
            if not res.values:
                return [], ResolveOutcome(_clarify(f'I couldn\'t find a staff member named "{who}". Please check the spelling.'),
                                          problem="unknown:staff")  # fmt: skip
            if res.ambiguous:
                return [], ResolveOutcome(
                    _clarify(f'"{who}" matches more than one {label}. Which one do you mean?',
                             [{"label": v, "query": _replace(question, who, v)} for v in res.values[:6]]),
                    problem="ambiguous",
                )  # fmt: skip
            out.extend(res.values)
        return list(dict.fromkeys(out)), None

    sn = filters.get("staff_name")
    if isinstance(sn, dict):
        for bucket in ("include", "exclude"):
            if sn.get(bucket):
                sn[bucket], problem = names(sn[bucket], "staff member")
                if problem is not None:
                    return problem
    sf = filters.get("staff_facility")
    if isinstance(sf, dict):
        for bucket in ("include", "exclude"):
            values = []
            for text in sf.get(bucket, []) or []:
                res = index.resolve_wing(str(text))
                if not res.values:
                    return ResolveOutcome(_clarify(f'I couldn\'t find the facility "{text}".'), problem="unknown")
                values.extend(res.values)
            if values:
                sf[bucket] = list(dict.fromkeys(values))
    lookup = plan.get("lookup")
    if isinstance(lookup, dict) and lookup.get("staff"):
        lookup["staff"], problem = names(lookup["staff"], "staff member")
        if problem is not None:
            return problem
    return None


def _replace(question: str, original: str, replacement: str) -> str:
    pattern = re.compile(re.escape(original), re.IGNORECASE)
    return pattern.sub(replacement, question, count=1) if pattern.search(question) else f"Tell me about {replacement}"
