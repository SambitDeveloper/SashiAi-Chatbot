"""Data-driven entity handling (no question-parsing rules).

Everything here is driven by values loaded from the database:
  • mask():    replaces resident names in the question with placeholders before it is sent to Claude,
               so resident names never leave the server and similar questions share one cache entry.
  • resolve_*: maps what Claude wrote ("Jenkins", "garden wing", "resident_1") to exact database values,
               reporting unknown or ambiguous references instead of guessing.
"""

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from app.data.repository import Vocabulary

# Staff titles/credentials are stripped when comparing caregiver names (data normalisation).
_TITLE_WORDS = {
    "nurse",
    "head",
    "care",
    "aide",
    "coordinator",
    "wellness",
    "rn",
    "lpn",
    "cna",
    "dr",
    "senior",
    "charge",
}
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z.'’-]*")
_RESIDENT_ID_RE = re.compile(r"\bRES[-_ ]?(\d{3,6})\b", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"^resident_\d+$")

FULL_NAME_THRESHOLD = 85
SINGLE_NAME_THRESHOLD = 86


def _norm(token: str) -> str:
    t = token.lower().replace("’", "'")
    t = re.sub(r"'s$", "", t)
    return re.sub(r"[^a-z]", "", t)


def _name_parts(name: str) -> list[str]:
    return [p for p in (_norm(x) for x in re.split(r"[\s-]+", name)) if p]


@dataclass
class _IndexedResident:
    resident_id: str
    display: str
    first: list[str]
    last: list[str]
    keys: list[str]  # concatenated forms: first+last, and first+each last-name part


@dataclass
class _IndexedCaregiver:
    value: str
    person: str  # normalised person name without titles, e.g. "sarahjenkins"
    tokens: list[str]


@dataclass
class MaskResult:
    text: str
    placeholders: dict[str, list[str]] = field(default_factory=dict)  # resident_1 → [resident ids]
    originals: dict[str, str] = field(default_factory=dict)  # resident_1 → text the user typed


@dataclass
class Resolution:
    values: list[str]
    ambiguous: bool = False


class EntityIndex:
    def __init__(self, vocab: Vocabulary):
        self.display_by_id: dict[str, str] = {}
        self._residents: list[_IndexedResident] = []
        for r in vocab.residents:
            first, last = _name_parts(r["firstName"]), _name_parts(r["lastName"])
            keys = ["".join(first + last)] + (["".join(first + [p]) for p in last] if len(last) > 1 else [])
            display = f"{r['firstName']} {r['lastName']}"
            self.display_by_id[r["residentId"].upper()] = display
            self._residents.append(_IndexedResident(r["residentId"].upper(), display, first, last, keys))

        self._caregivers: list[_IndexedCaregiver] = []
        for value in vocab.caregivers:
            tokens = [t for t in _name_parts(value.split(",")[0]) if t not in _TITLE_WORDS]
            self._caregivers.append(_IndexedCaregiver(value, "".join(tokens), tokens))
        self._caregiver_first_names = {c.tokens[0] for c in self._caregivers if c.tokens}

        self.wings = list(vocab.wings)
        self._rooms = {re.sub(r"[^a-z0-9]", "", r.lower()): r for r in vocab.rooms}

    # ------------------------------------------------------------------ masking
    def mask(self, question: str) -> MaskResult:
        tokens = [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(question)]
        spans: list[tuple[int, int, list[str]]] = []
        used: set[int] = set()

        # Multi-token windows: full resident names (typo tolerant), longest first. Every word in the window
        # must belong to the name, so filler such as "of" or "about" is never swallowed. Caregivers stay in place.
        for size in (4, 3, 2):
            for i in range(len(tokens) - size + 1):
                idx = range(i, i + size)
                if any(j in used for j in idx):
                    continue
                words = [_norm(tokens[j][0]) for j in idx]
                concat = "".join(words)
                if len(concat) < 6 or any(len(w) < 2 for w in words):
                    continue
                if any(
                    self._window_is_name(words, c.tokens) and fuzz.ratio(concat, c.person) >= FULL_NAME_THRESHOLD
                    for c in self._caregivers
                    if len(c.tokens) > 1
                ):
                    used.update(idx)
                    continue
                ids = self._best_full_match(concat, words)
                if ids:
                    spans.append((tokens[i][1], tokens[i + size - 1][2], ids))
                    used.update(idx)

        # Single capitalised tokens that exactly equal a resident first/last name ("Tell me about Roosevelt").
        for j, (tok, start, end) in enumerate(tokens):
            if j in used or not tok[0].isupper():
                continue
            n = _norm(tok)
            if len(n) < 3 or n in self._caregiver_first_names:
                continue
            # Whole surname or first name only: "May" must not match "May-Alcott" ("May I see ...").
            ids = [r.resident_id for r in self._residents if n == "".join(r.last) or n == "".join(r.first)]
            if ids:
                spans.append((start, end, ids))

        # Keep a trailing possessive outside the placeholder: "Vance's room" → "resident_1's room".
        spans = [
            (st, en - 2 if question[en - 2 : en].lower() in ("'s", "\u2019s") else en, ids) for st, en, ids in spans
        ]

        result = MaskResult(text=question)
        text = question
        for k, (start, end, ids) in enumerate(sorted(spans, key=lambda s: s[0])):
            result.placeholders[f"resident_{k + 1}"] = ids
            result.originals[f"resident_{k + 1}"] = question[start:end]
        for k, (start, end, _ids) in reversed(list(enumerate(sorted(spans, key=lambda s: s[0])))):
            text = text[:start] + f"resident_{k + 1}" + text[end:]
        result.text = text
        return result

    @staticmethod
    def _window_is_name(words: list[str], name_parts: list[str]) -> bool:
        joined = "".join(name_parts)
        return all(fuzz.partial_ratio(w, joined) >= 80 for w in words)

    def _best_full_match(self, concat: str, words: list[str] | None = None) -> list[str]:
        best, ids = 0.0, []
        for r in self._residents:
            if words is not None and not self._window_is_name(words, r.first + r.last):
                continue
            score = max(fuzz.ratio(concat, k) for k in r.keys)
            if score > best + 0.01:
                best, ids = score, [r.resident_id]
            elif abs(score - best) <= 0.01 and score > 0:
                ids.append(r.resident_id)
        return ids if best >= FULL_NAME_THRESHOLD else []

    # ------------------------------------------------------------------ resolution
    def resolve_resident(self, ref: str, placeholders: dict[str, list[str]]) -> Resolution:
        ref = ref.strip()
        if _PLACEHOLDER_RE.match(ref.lower()):
            ids = placeholders.get(ref.lower(), [])
            return Resolution(ids, ambiguous=len(ids) > 1)
        m = _RESIDENT_ID_RE.search(ref)
        if m:
            rid = f"RES-{m.group(1)}"
            return Resolution([rid] if rid in self.display_by_id else [])
        parts = _name_parts(ref)
        if not parts:
            return Resolution([])
        if len(parts) > 1:
            ids = self._best_full_match("".join(parts))
            return Resolution(ids, ambiguous=len(ids) > 1)
        n = parts[0]
        exact = [r.resident_id for r in self._residents if n in r.last or n in r.first]
        if exact:
            return Resolution(exact, ambiguous=len(exact) > 1)
        fuzzy = [
            r.resident_id
            for r in self._residents
            if len(n) >= 5 and any(fuzz.ratio(n, p) >= SINGLE_NAME_THRESHOLD for p in r.first + r.last)
        ]
        return Resolution(fuzzy, ambiguous=len(fuzzy) > 1)

    def resolve_caregiver(self, text: str) -> Resolution:
        tokens = [t for t in _name_parts(text.split(",")[0]) if t not in _TITLE_WORDS]
        if not tokens:
            return Resolution([])
        joined = "".join(tokens)
        hits = [c for c in self._caregivers if all(t in c.tokens for t in tokens)]
        if not hits:
            hits = [c for c in self._caregivers if fuzz.ratio(joined, c.person) >= FULL_NAME_THRESHOLD]
        if not hits and len(tokens) == 1:
            hits = [
                c for c in self._caregivers if any(fuzz.ratio(tokens[0], t) >= SINGLE_NAME_THRESHOLD for t in c.tokens)
            ]
        people = {c.person for c in hits}
        # The same person can appear under two job titles; that is one person, not an ambiguity.
        return Resolution([c.value for c in hits], ambiguous=len(people) > 1)

    def resolve_wing(self, text: str) -> Resolution:
        # Compare only the distinctive part: "West Wing" must not match "East Wing" via the shared word "wing".
        q = " ".join(p for p in _name_parts(text) if p != "wing")
        if not q:
            return Resolution([])
        scored = []
        for w in self.wings:
            wn = " ".join(p for p in _name_parts(w) if p != "wing")
            score = max(fuzz.ratio(q, wn), fuzz.token_set_ratio(q, wn) if len(q) >= 4 else 0)
            scored.append((score, w))
        scored.sort(reverse=True)
        if not scored or scored[0][0] < 85:
            return Resolution([])
        close = [w for sc, w in scored if sc >= 85 and sc >= scored[0][0] - 2]
        return Resolution(close, ambiguous=len(close) > 1)

    def resolve_room(self, text: str) -> Resolution:
        key = re.sub(r"[^a-z0-9]", "", text.lower())
        if key in self._rooms:
            return Resolution([self._rooms[key]])
        digits = re.sub(r"\D", "", key)
        if digits:
            hits = [v for k, v in self._rooms.items() if re.sub(r"^[a-z]+", "", k) == digits]
            return Resolution(hits, ambiguous=len(hits) > 1)
        return Resolution([])
