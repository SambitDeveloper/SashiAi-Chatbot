"""Projection detection: which attributes does the question ask to SEE (as opposed to filter by)?

Shared by the deterministic planner (to build `fields`) and the grounding check (to verify a model plan neither drops
nor invents requested output fields). Configuration only (intent_routes.yaml): the attribute words
(`lookup_attributes`) and the role cues (`projection`).

The role comes from the phrase's position in the sentence, independently for every occurrence, so one attribute can
be both a condition and a projection ("residents in room 101 and show their room numbers"):
  • shown — the phrase follows a display verb or a possessive (articles may sit in between), or continues a list of
    shown attributes ("room and caregiver");
  • a condition — the phrase is directly bound to a value of that attribute ("room 101", "caregiver Sarah Jenkins",
    "age over 80", "admission date last month"); such an occurrence is never shown, whatever precedes it.
"""

import re
from dataclasses import dataclass, field

from app.nlu.grammar import parse_dates, parse_number_ranges

_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+|[,&=:]")


@dataclass
class ProjectionMention:
    attribute: str | None  # None: an identity word ("name", "names") — always shown, but it continues a list
    span: tuple[int, int]
    cue_spans: list[tuple[int, int]] = field(default_factory=list)  # display verb / possessive / joiner words


class ProjectionDetector:
    def __init__(self, routes_config: dict):
        direct = routes_config.get("direct") or {}
        cfg = direct.get("projection") or {}
        self.verbs = [" ".join(str(w).lower().split()) for w in cfg.get("display_verbs") or []]
        self.possessives = {str(w).lower() for w in cfg.get("possessives") or []}
        self.determiners = {str(w).lower() for w in cfg.get("determiners") or []}
        self.joiners = {str(w).lower() for w in cfg.get("joiners") or []}
        self.value_links = {str(w).lower() for w in cfg.get("value_links") or []}
        words = [(attr, " ".join(str(w).lower().split())) for attr, ws in (direct.get("lookup_attributes") or {}).items() for w in ws]
        # identity words name what every result row already shows (the resident's name); they produce no field but
        # are part of a list of shown attributes ("show the name, age and room")
        words += [(None, " ".join(str(w).lower().split())) for w in cfg.get("identity_words") or []]
        words.sort(key=lambda aw: len(aw[1]), reverse=True)  # longest phrase first ("room number" before "room")
        self.phrases = [
            (attr, re.compile(r"(?<![a-z])" + r"\s+".join(map(re.escape, w.split())) + r"(?:e?s)?(?![a-z])"))
            for attr, w in words
        ]

    # ------------------------------------------------------------------ public
    def scan(self, text: str, blocked: list[bool] | None = None, values: dict[str, list[int]] | None = None):
        """Projection mentions in `text`. `blocked` marks characters that are not attribute words (resident names,
        "no caregiver" missing-value phrases); `values[attribute]` lists start offsets of recognised values of that
        attribute (a phrase directly followed by one is a condition, not a projection)."""
        low = text.lower()
        tokens = [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(low)]
        taken = [False] * len(low)
        found: list[tuple[str, int, int]] = []
        for attr, pattern in self.phrases:
            for m in pattern.finditer(low):
                s, e = m.span()
                if any(taken[s:e]) or (blocked and any(blocked[s:e])):
                    continue
                found.append((attr, s, e))
                for i in range(s, e):
                    taken[i] = True
        found.sort(key=lambda x: x[1])

        mentions: list[ProjectionMention] = []
        last_end = -1
        for attr, s, e in found:
            if attr is not None and self._bound_to_value(low, tokens, e, (values or {}).get(attr, [])):
                last_end = -1  # a condition interrupts a list of shown attributes
                continue
            cues = self._cue_before(low, tokens, s, last_end)
            if cues is not None:
                mentions.append(ProjectionMention(attr, (s, e), cues))
                last_end = e
            else:
                last_end = -1
        return mentions

    # ------------------------------------------------------------------ helpers
    def _bound_to_value(self, low: str, tokens, end: int, value_starts: list[int]) -> bool:
        if not value_starts:
            return False
        pos = end
        for word, s, e in tokens:
            if s < end:
                continue
            if s in value_starts:
                return True
            if word in self.value_links:
                pos = e
                continue
            break
        return pos in value_starts or any(abs(v - pos) <= 1 for v in value_starts)

    def _cue_before(self, low: str, tokens, start: int, last_end: int) -> list[tuple[int, int]] | None:
        """The cue words that make the phrase starting at `start` a projection, or None."""
        before = [t for t in tokens if t[2] <= start]
        cues: list[tuple[int, int]] = []
        i = len(before) - 1
        while i >= 0 and before[i][0] in self.determiners | self.possessives:
            cues.append((before[i][1], before[i][2]))
            if before[i][0] in self.possessives:  # "their room numbers", "along with their room numbers"
                return [*cues, *self._verb_ending_at(before, i - 1)]
            i -= 1
        if i < 0:
            return None
        word, s, e = before[i]
        verb = self._verb_ending_at(before, i)  # "show the room", "along with caregivers"
        if verb:
            return [*cues, *verb]
        if word in self.joiners and last_end >= 0 and last_end <= s:  # "room and caregiver"
            return [*cues, (s, e)]
        return None


    def _verb_ending_at(self, tokens, i: int) -> list[tuple[int, int]]:
        """Spans of the display verb whose last word is tokens[i] (longest verb first), or []."""
        for verb in sorted(self.verbs, key=lambda v: len(v.split()), reverse=True):
            vw = verb.split()
            if i - len(vw) + 1 >= 0 and [t[0] for t in tokens[i - len(vw) + 1 : i + 1]] == vw:
                return [(t[1], t[2]) for t in tokens[i - len(vw) + 1 : i + 1]]
        return []


def value_starts(question: str, index, today=None, dates=None, numbers=None) -> dict[str, list[int]]:
    """Start offsets of recognised attribute values in the question, per attribute (for the condition/projection
    decision). Values come from the live vocabulary and the date/number grammar, never from fixed lists."""
    low = question.lower()
    out: dict[str, list[int]] = {}

    def add(attr: str, text: str) -> None:
        for m in re.finditer(r"(?<![a-z0-9])" + re.escape(text.lower()) + r"(?![a-z0-9])", low):
            out.setdefault(attr, []).append(m.start())

    for text in index.mentioned_caregivers(question):
        add("primary_caregiver", text)
    for text in index.mentioned_wings(question):  # the text as written ("FACILITY_12")
        add("facility", text)
    for status in index.statuses:
        add("status", status)
    for care in index.care_types:
        add("care_type", care.replace("_", " "))
    for g in ("female", "male", "other"):
        add("gender", g)
    for m in re.finditer(r"(?<![a-z0-9])[a-z]{0,3}-?\d[\w-]*", low):  # room-like tokens: "101", "mc-101"
        out.setdefault("room_number", []).append(m.start())
    for n in numbers if numbers is not None else parse_number_ranges(low):
        out.setdefault("age", []).append(n.start)
    for d in dates if dates is not None else (parse_dates(low, today) if today else []):
        for attr in ("admission_date", "discharge_date", "created_date"):
            out.setdefault(attr, []).append(d.start)
    return out
