"""Data-driven entity handling (no question-parsing rules).

Everything here is driven by values loaded from the database:
  • mask():    replaces resident names in the question with placeholders before it reaches the model,
               so resident names never leave the server and similar questions share one cache entry.
  • resolve_*: maps what the model wrote ("Jenkins", "garden wing", "resident_1") to exact database values,
               reporting unknown or ambiguous references instead of guessing.
"""

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from app.data.repository import Vocabulary
from app.data.schema_config import is_missing
from app.models.plan import CARE_TYPES, STATUSES

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


def _wing_tokens(text: str) -> list[str]:
    """Wing/facility name tokens with numbers kept: "(train) FACILITY_12" → ["train", "facility", "12"].

    `_name_parts` drops digits (right for people's names), which made every "FACILITY_n" look identical.
    """
    return re.findall(r"[a-z]+|\d+", text.lower().replace("’", "'"))


def _same_numbers(a: list[str], b: list[str]) -> bool:
    """Position by position, any token that is a number must be identical."""
    return all(x == y for x, y in zip(a, b, strict=True) if x.isdigit() or y.isdigit())


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
            # surname-first is common in staff questions ("Jain Aarya", "Vance, Eleanor"): the same person
            keys.append("".join(last + first))
            display = f"{r['firstName']} {r['lastName']}"
            self.display_by_id[r["residentId"].upper()] = display
            self._residents.append(_IndexedResident(r["residentId"].upper(), display, first, last, keys))
        self._by_id = {r.resident_id: r for r in self._residents}

        self._caregivers: list[_IndexedCaregiver] = []
        for value in (c for c in vocab.caregivers if not is_missing(c, "primaryCaregiver")):
            tokens = [t for t in _name_parts(value.split(",")[0]) if t not in _TITLE_WORDS]
            self._caregivers.append(_IndexedCaregiver(value, "".join(tokens), tokens))
        self._caregiver_first_names = {c.tokens[0] for c in self._caregivers if c.tokens}
        # caregivers recorded under one word only ("Vik"): that word is their whole recorded name
        self._single_word_caregivers: dict[str, list[str]] = {}
        for c in self._caregivers:
            if len(c.tokens) == 1:
                self._single_word_caregivers.setdefault(c.tokens[0], []).append(c.value)

        # placeholders such as wing "Unknown" (resident_schema.yaml missing_values) are not names: "unknown status"
        # must never be read as the facility "Unknown"
        self.wings = [w for w in vocab.wings if not is_missing(w, "wing")]
        # statuses / care types that exist in the database; all known ones when the database did not say
        self.statuses = [s for s in STATUSES if not vocab.statuses or s in vocab.statuses]
        self.care_types = [c for c in CARE_TYPES if not vocab.care_types or c in vocab.care_types]
        self._rooms = {re.sub(r"[^a-z0-9]", "", r.lower()): r for r in vocab.rooms if not is_missing(r, "roomNumber")}

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

        # A single-letter initial right after a recognised name ("Jone Aarya J", "Aarya J.") is part of that name
        # when it is the first letter of one of that resident's names; otherwise it is left alone.
        with_initials = []
        for st, en, ids in spans:
            m = re.match(r"\s+([A-Za-z])\.?(?![A-Za-z'’])", question[en:])
            if m and len(ids) == 1 and m.group(1).lower() in self._initials(ids[0]):
                en += m.end()
            with_initials.append((st, en, ids))
        spans = with_initials

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

    def _initials(self, resident_id: str) -> set[str]:
        r = self._by_id.get(resident_id.upper())
        return {p[0] for p in (r.first + r.last) if p} if r else set()

    @staticmethod
    def _window_is_name(words: list[str], name_parts: list[str]) -> bool:
        """Every word of the window belongs to the name. Short words (≤ 3 letters) must BE a name part: a substring
        match would let function words join the name ("as" is inside "aakash", "on" inside "jones")."""
        joined = "".join(name_parts)

        def short_ok(i: int, w: str) -> bool:  # a name part, or written apart from its neighbour ("van gogh" = VanGogh)
            nxt = words[i + 1] if i + 1 < len(words) else ""
            prev = words[i - 1] if i > 0 else ""
            return w in name_parts or any((nxt and (w + nxt) in p) or (prev and (prev + w) in p) for p in name_parts)

        return all(short_ok(i, w) if len(w) <= 3 else fuzz.partial_ratio(w, joined) >= 80 for i, w in enumerate(words))

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

    def caregiver_values(self) -> list[str]:
        return [c.value for c in self._caregivers]

    # ------------------------------------------------------------------ mentions (completeness check)
    def _outside_fixed_values(self, text: str, found: dict[str, list[str]]) -> dict[str, list[str]]:
        """Drop name matches that are only part of a status, care type or gender in the question.

        Staff and facility names are free text ("Director Nursing"); "skilled nursing" in a question is the care type,
        never that staff member.
        """
        low = text.lower()
        spans = []
        for phrase in [*self.statuses, *(c.replace("_", " ") for c in self.care_types), "female", "male"]:
            spans += [m.span() for m in re.finditer(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", low)]
        if not spans:
            return found

        def inside(raw: str) -> bool:
            hits = [m.span() for m in re.finditer(r"(?<![a-z])" + re.escape(raw.lower()) + r"(?![a-z])", low)]
            return bool(hits) and all(any(a <= s and e <= b for a, b in spans) for s, e in hits)

        return {raw: values for raw, values in found.items() if not inside(raw)}

    def mentioned_caregivers(self, text: str) -> dict[str, list[str]]:
        """Caregivers named in the question, as {text written: [exact database values]}.

        Full names (typo tolerant, titles ignored) or an exact surname. A first name alone is too ambiguous.
        Used to check that the plan did not drop a caregiver condition.
        """
        tokens = [(m.group(0), _norm(m.group(0))) for m in _TOKEN_RE.finditer(text)]
        found: dict[str, list[str]] = {}
        used: set[int] = set()
        for size in (4, 3, 2):
            for i in range(len(tokens) - size + 1):
                idx = range(i, i + size)
                if any(j in used for j in idx):
                    continue
                words = [tokens[j][1] for j in idx if tokens[j][1] not in _TITLE_WORDS]
                if len(words) < 2:
                    continue
                concat = "".join(words)
                hits = [
                    c.value
                    for c in self._caregivers
                    if len(c.tokens) > 1
                    and self._window_is_name(words, c.tokens)
                    and fuzz.ratio(concat, c.person) >= FULL_NAME_THRESHOLD
                ]
                if hits:
                    found[" ".join(tokens[j][0] for j in idx)] = hits
                    used.update(idx)
        for j, (raw, n) in enumerate(tokens):
            if j in used:
                continue
            hits = [c.value for c in self._caregivers if len(c.tokens) > 1 and n == c.tokens[-1]] if len(n) >= 4 else []
            # A caregiver recorded under a single word is named in full by that word. It must be written as a name
            # (capitalised, not the sentence's first word) so ordinary words never match a one-word staff record.
            if not hits and j > 0 and raw[:1].isupper() and len(n) >= 3:
                hits = list(self._single_word_caregivers.get(n, []))
            if hits:
                found[raw] = hits
        # the text written, without trailing punctuation ("Nursing." at the end of the sentence is "Nursing")
        found = {raw.rstrip(".,;:!?'’-"): values for raw, values in found.items()}
        return self._outside_fixed_values(text, found)

    def mentioned_wings(self, text: str) -> dict[str, list[str]]:
        """Wings named in the question (typo tolerant), as {text written: [wing]}.

        Numbers in a name must match exactly ("(train) FACILITY_42" is not FACILITY_4 or FACILITY_420); only the
        letters are compared with typo tolerance. A bracketed qualifier in a recorded name ("(train) FACILITY_12") is
        optional: "FACILITY_12" names that wing too.
        """
        low = text.lower().replace("’", "'")
        tokens = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[a-z]+|\d+", low)]
        words = [t[0] for t in tokens]
        found: dict[str, list[str]] = {}
        for wing in self.wings:
            forms = [_wing_tokens(wing)]
            core = _wing_tokens(re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", wing))
            if core and core != forms[0]:
                forms.append(core)
            for parts in forms:
                if not parts:
                    continue
                target = " ".join(parts)
                hit = next(
                    (
                        i
                        for i in range(len(words) - len(parts) + 1)
                        if _same_numbers(words[i : i + len(parts)], parts)
                        and fuzz.ratio(" ".join(words[i : i + len(parts)]), target) >= 90
                    ),
                    None,
                )
                if hit is not None:
                    start, end = tokens[hit][1], tokens[hit + len(parts) - 1][2]
                    start -= 1 if start > 0 and text[start - 1] in "([" else 0  # "(train) …" as written
                    written = text[start:end]
                    found.setdefault(written, []).append(wing)
                    break
        return self._outside_fixed_values(text, found)

    # ------------------------------------------------------------------ resolution
    def resolve_resident(self, ref: str, placeholders: dict[str, list[str]]) -> Resolution:
        ref = ref.strip()
        if _PLACEHOLDER_RE.match(ref.lower()):
            ids = placeholders.get(ref.lower(), [])
            return Resolution(ids, ambiguous=len(ids) > 1)
        if ref.upper() in self.display_by_id:  # a database id (e.g. a MongoDB ObjectId), as direct answers pass them
            return Resolution([ref.upper()])
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
        # Numbers must match exactly: "FACILITY_12" is never FACILITY_1 or FACILITY_120 (letters stay typo tolerant).
        q_tokens = [p for p in _wing_tokens(text) if p != "wing"]
        q = " ".join(q_tokens)
        if not q:
            return Resolution([])
        q_numbers = [t for t in q_tokens if t.isdigit()]
        scored = []
        for w in self.wings:
            w_tokens = [p for p in _wing_tokens(w) if p != "wing"]
            if [t for t in w_tokens if t.isdigit()] != q_numbers and q_numbers:
                continue
            wn = " ".join(w_tokens)
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
