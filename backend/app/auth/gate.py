"""E — the pre-model gate: decide from the question's own words, before any model call, whether it needs something the
caller cannot read. A denial here costs 0 tokens. It only denies on positive evidence (a word rule of
access_policy.yaml); a question it cannot place goes on, and the plan check (C) still applies to whatever plan results.
"""

import re
from dataclasses import dataclass
from functools import lru_cache

from app.auth.guard import AccessDenied
from app.auth.policy import AccessPolicy, access_policy
from app.auth.principal import Principal
from app.models.plan import CARE_TYPES, STATUSES


@lru_cache
def _values() -> re.Pattern:
    """Care types and statuses as written ("skilled nursing", "on leave"): values, never staff or family words."""
    phrases = sorted({v.replace("_", " ") for v in (*CARE_TYPES, *STATUSES)}, key=len, reverse=True)
    return re.compile(r"(?<![a-z])(?:" + "|".join(re.escape(p) for p in phrases) + r")(?![a-z])")


@dataclass
class GateResult:
    needed: list[tuple[str, ...]]  # the rules that matched (each: any one of these capabilities)
    denied: AccessDenied | None = None


def check_question(text: str, principal: Principal, names_resident: bool = False, app_sections: bool = False,
                   policy: AccessPolicy | None = None) -> GateResult:  # fmt: skip
    """`text`: the question (any case). `names_resident`: the question names a resident (masked as resident_N).
    `app_sections`: the person-question grammar read it as "can <staff> access <app section>?"."""
    p = policy or access_policy()
    low = _values().sub(" ", text.lower())  # "skilled nursing" is a care type, not the staff role "nurse"
    result = GateResult(needed=[])
    for rule in p.gate:
        hit = (rule.pattern is not None and rule.pattern.search(low) is not None) or (
            rule.names_residents and names_resident) or (rule.app_sections and app_sections)  # fmt: skip
        if hit and ((rule.unless is not None and rule.unless.search(low)) or (rule.unless_names_resident and names_resident)):
            hit = False
        if not hit:
            continue
        result.needed.append(rule.any_of)
        if result.denied is None and not any(principal.can(c, p) for c in rule.any_of):
            result.denied = AccessDenied(rule.any_of[0], "gate", p)
    return result
