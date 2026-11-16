"""Recognise "no value recorded" phrases: "no caregiver assigned", "room not recorded", "without an admission date".

Configuration only (intent_routes.yaml `direct.missing_values`): which words mean absent/present and which words name
which field. Shared by the deterministic planner (to answer without the model) and the grounding check (to verify a
model plan neither drops nor invents a missing-value condition), so both read a question the same way.

Conservative by design:
  • absent: "no / without / missing … X", "X missing / unknown / not recorded / not assigned", "don't have a X",
    "not assigned to a X";
  • present only with an explicit verb: "with/has a X assigned/recorded" (so "the staff assigned to resident_1" in a
    lookup is never read as a condition);
  • words inside a recognised resident's name never count;
  • "or" between two different fields ("no caregiver or no room") is reported: the plan format cannot express OR
    across fields, so the question must not silently become an AND.
"""

import re
from dataclasses import dataclass, field

_ART = r"(?:(?:a|an|any|the|their|his|her)\s+)?"
_BE = r"(?:(?:is|are|was|were|has\s+been|have\s+been)\s+)?"


def _alt(words: list[str]) -> str:
    return "|".join(r"\s+".join(map(re.escape, w.lower().split())) for w in sorted(words, key=len, reverse=True))


@dataclass
class MissingMention:
    field: str  # plan field, e.g. "primary_caregiver"
    missing: bool  # True: no value recorded · False: a value is recorded
    span: tuple[int, int]


@dataclass
class MissingScan:
    mentions: list[MissingMention] = field(default_factory=list)
    or_across_fields: bool = False  # "no caregiver or no room": cannot be expressed, must not become an AND

    @property
    def by_field(self) -> dict[str, bool]:
        return {m.field: m.missing for m in self.mentions}

    @property
    def conflicting(self) -> bool:
        """The same field said to be both missing and recorded."""
        seen: dict[str, bool] = {}
        for m in self.mentions:
            if seen.get(m.field, m.missing) != m.missing:
                return True
            seen[m.field] = m.missing
        return False


class MissingValueDetector:
    def __init__(self, routes_config: dict):
        cfg = (routes_config.get("direct") or {}).get("missing_values") or {}
        self.field_of: dict[str, str] = {
            " ".join(str(w).lower().split()): f for f, words in (cfg.get("fields") or {}).items() for w in words
        }
        if not self.field_of:
            self.patterns: list[tuple[re.Pattern, bool]] = []
            return
        f = _alt(list(self.field_of))
        absent_before = _alt([str(w) for w in cfg.get("absent_before") or []])
        absent_after = _alt([str(w) for w in cfg.get("absent_after") or []])
        absent_have = _alt([str(w) for w in cfg.get("absent_have") or []])
        present_before = _alt([str(w) for w in cfg.get("present_before") or []])
        present_after = _alt([str(w) for w in cfg.get("present_after") or []])
        tail = rf"(?:\s+(?:{present_after}))?" if present_after else ""
        b, e = r"(?<![a-z])", r"(?![a-z])"
        self.patterns = []
        # absent first: in "no caregiver assigned" the "no" decides, "assigned" is just part of the phrase
        if absent_before:
            self.patterns.append((re.compile(rf"{b}(?:{absent_before})\s+{_ART}(?P<f>{f}){tail}{e}"), True))
        if absent_have:
            self.patterns.append((re.compile(rf"(?:{absent_have})\s+{_ART}(?P<f>{f}){tail}{e}"), True))
        if absent_after:
            self.patterns.append((re.compile(rf"{b}(?:{absent_after})\s+to\s+{_ART}(?P<f>{f}){e}"), True))
            self.patterns.append((re.compile(rf"{b}(?P<f>{f})\s+{_BE}(?:{absent_after}){e}"), True))
        if present_before and present_after:
            self.patterns.append(
                (re.compile(rf"{b}(?:{present_before})\s+(?P<f>{f})\s+{_BE}(?:{present_after}){e}"), False)
            )

    def scan(self, text: str, blocked: list[bool] | None = None) -> MissingScan:
        """Find missing/present mentions in `text` (any case). `blocked[i]` marks characters that belong to a
        recognised resident name; a match touching them is ignored."""
        low = text.lower()
        taken = [False] * len(low)
        mentions: list[MissingMention] = []
        for pattern, missing in self.patterns:
            for m in pattern.finditer(low):
                s, e = m.span()
                if any(taken[s:e]) or (blocked and any(blocked[s:e])):
                    continue
                phrase = " ".join(m.group("f").split())
                mentions.append(MissingMention(self.field_of[phrase], missing, (s, e)))
                for i in range(s, e):
                    taken[i] = True
        mentions.sort(key=lambda x: x.span[0])
        or_across = any(
            a.field != b.field and re.search(r"(?<![a-z])or(?![a-z])", low[a.span[1] : b.span[0]])
            for a, b in zip(mentions, mentions[1:], strict=False)
        )
        return MissingScan(mentions, or_across)
