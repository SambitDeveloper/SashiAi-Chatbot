"""Data-driven entity handling (no question-parsing rules).

Everything here is driven by values loaded from the database:
  • mask():    replaces resident names in the question with placeholders before it reaches the model,
               so resident names never leave the server and similar questions share one cache entry.
  • resolve_*: maps what the model wrote ("Jenkins", "garden wing", "resident_1") to exact database values,
               reporting unknown or ambiguous references instead of guessing.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
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
class _IndexedFamily:
    """One family member of a live resident (familymembers): names and relation only, never contact details."""

    member_id: str
    resident_id: str  # upper case, like every resident id in this index
    facility_id: str | None  # the RESIDENT's facility (the member's own facilityId is often missing)
    name: str
    tokens: list[str]
    person: str  # normalised name, e.g. "jiyapatel"
    rel: str  # normalised relation ("daughter")
    relation: str  # as recorded ("Daughter")


# Words that make the following name a staff member's ("Dr. Jiya Patel", "Nurse Kim"): never a family reading.
_STAFF_TITLES = {"dr", "doctor", "nurse", "rn", "lpn", "cna", "aide", "coordinator"}


@lru_cache
def _app_words() -> frozenset[str]:
    """Every word of the assistant's own wording (app/data/intent_routes.yaml: route utterances, filler, field and
    attribute words — generic language only, never a person's name) plus statuses and care types. A lowercase word
    from this set is the question's wording, never a staff name ("will", "active", "status")."""
    path = Path(__file__).resolve().parents[1] / "data" / "intent_routes.yaml"
    words: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(k)
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str):
            words.update(re.findall(r"[a-z]+", node.lower()))

    walk(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    words |= {w for v in (*STATUSES, *CARE_TYPES) for w in v.split("_")}
    return frozenset(words)


# Words right before a lowercase single name that make it a NAME position ("who is jone", "does jone look after",
# "named jone", "nurse jone"): with no such word (or a possessive "jone's"), a lowercase word is never masked.
_NAME_SLOT_BEFORE = {"is", "was", "about", "does", "did", "named", "called", "by", "and"}


def _has_family_word(words: list[str]) -> bool:
    """Family vocabulary (resident_schema.yaml cue words: family, daughter, kin, ...) is the question's wording: a window
    containing one is never a family member's name, whatever a record happens to be called ("Family Member")."""
    from app.data.relations import family_relation  # local: vocabulary is imported by the data layer's users

    rel = family_relation()
    return rel is not None and any(rel.is_family_word(w) for w in words)


@dataclass
class MaskResult:
    text: str
    placeholders: dict[str, list[str]] = field(default_factory=dict)  # resident_1 → [resident ids]
    originals: dict[str, str] = field(default_factory=dict)  # resident_1 → text the user typed
    # family-member names (feature flag): family_1 → candidate family members; person_1 → every candidate of each kind.
    # Candidates stay on the server; the model only ever sees the placeholder.
    family: dict[str, list[dict]] = field(default_factory=dict)
    persons: dict[str, dict[str, list]] = field(default_factory=dict)  # {"staff": [...], "family": [...], "residents": [...]}
    entity_originals: dict[str, str] = field(default_factory=dict)  # family_1 / person_1 → text the user typed
    entity_spans: list[tuple[int, int]] = field(default_factory=list)  # their positions in the ORIGINAL question
    # What the MODEL reads: `text` with the remaining recognised people (staff names, a name several kinds of person
    # share) also replaced by person_N. `text` itself is unchanged, so the deterministic stages read staff names exactly
    # as before. None = same as `text`. Candidates of these person_N: `model_persons` (server side only).
    model_text: str | None = None
    model_persons: dict[str, dict[str, list]] = field(default_factory=dict)
    # person_N of the PREVIOUS turn (conversation memory), for a follow-up whose model plan reuses them: used to resolve
    # values only, never required to appear in this question's plan
    inherited_persons: dict[str, dict[str, list]] = field(default_factory=dict)

    def renumber_model_persons(self, after: int) -> None:
        """Number this question's model-only person_N after `after` (the previous turn's), so a follow-up never has
        two meanings for one placeholder."""
        if not self.model_persons:
            return
        order = sorted(self.model_persons, key=lambda k: int(k.split("_")[1]))
        mapping = {old: f"person_{after + i + 1}" for i, old in enumerate(order)}
        text = self.for_model
        text = re.sub(r"\bperson_\d+\b", lambda m: mapping.get(m.group(0), m.group(0)), text)
        self.model_text = text
        self.model_persons = {mapping[k]: v for k, v in self.model_persons.items()}
        self.entity_originals = {mapping.get(k, k): v for k, v in self.entity_originals.items()}

    def hide_names_in(self, value):
        """A plan / RequestIR for conversation memory: any text value containing a recognised staff name becomes that
        person_N (the next model call reads the previous plan; it must not carry the name)."""
        if isinstance(value, dict):
            return {k: self.hide_names_in(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.hide_names_in(v) for v in value]
        if isinstance(value, str) and self.model_persons:
            for key in self.model_persons:
                original = self.entity_originals.get(key)
                if original and re.search(rf"(?<![A-Za-z]){re.escape(original)}(?![A-Za-z])", value, re.I):
                    return key
        return value

    @property
    def for_model(self) -> str:
        return self.model_text if self.model_text is not None else self.text

    def hide_names(self, text: str | None) -> str | None:
        """Any text bound for the model (brief, retry hints) with the names behind model_persons replaced too."""
        if not text or not self.model_persons:
            return text
        for key in sorted(self.model_persons, key=lambda k: -len(self.entity_originals.get(k, ""))):
            original = self.entity_originals.get(key)
            if original:
                text = re.sub(rf"(?<![A-Za-z]){re.escape(original)}(?![A-Za-z])", key, text, flags=re.I)
        return text

    @property
    def all_originals(self) -> dict[str, str]:
        return {**self.originals, **self.entity_originals}


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
        # family members (familymembers.name): a third kind of person, matched only when family names are enabled
        self._family: list[_IndexedFamily] = []
        for m in getattr(vocab, "family_members", []) or []:
            tokens = [t for t in _name_parts(str(m.get("name") or "")) if t]
            if not tokens:
                continue
            self._family.append(_IndexedFamily(
                str(m.get("id")), str(m.get("residentId", "")).upper(), m.get("facilityId"), str(m.get("name") or "").strip(),
                tokens, "".join(tokens), str(m.get("rel") or ""), str(m.get("relation") or ""),
            ))  # fmt: skip
        self.family_relations = list(getattr(vocab, "family_relations", []) or [])
        # every staff member who is a person (staffs; app/data/staff.py), assigned or not, any status: staff questions
        # ("Priya Raman's residents", "Dr. Hart's phone") name staff the caregiver list may not have
        self._staff: list[_IndexedCaregiver] = []
        self._staff_rows: list[dict] = []  # parallel to _staff: designation and status, for "who is <name>" answers
        for row in getattr(vocab, "staff", []) or []:
            value = str(row.get("name") or "").strip()
            tokens = [t for t in _name_parts(value.split(",")[0]) if t not in _TITLE_WORDS]
            if value and tokens:
                self._staff.append(_IndexedCaregiver(value, "".join(tokens), tokens))
                self._staff_rows.append({"name": value, "designation": row.get("designation") or "", "status": row.get("status") or ""})
        self.staff_designations = sorted({str(r.get("designation")).strip() for r in getattr(vocab, "staff", []) or []
                                          if isinstance(r.get("designation"), str) and r.get("designation").strip()})  # fmt: skip
        self._rooms = {re.sub(r"[^a-z0-9]", "", r.lower()): r for r in vocab.rooms if not is_missing(r, "roomNumber")}

    # ------------------------------------------------------------------ masking
    def mask(self, question: str, family: bool = False) -> MaskResult:
        """Resident names → resident_N (always). With `family` (Settings.family_name_resolution_enabled), family-member
        names → family_N, or person_N when the name also belongs to staff or to another resident named in the question.
        Without family names in the question the result is exactly what it was before family names existed."""
        if not (family and self._family):
            result = self._mask_residents(question)
            self._mask_people_for_model(result)
            return result
        # A family member's FULL name ("Margaret Vance") is not a resident's surname ("Vance" → Eleanor Vance): it is
        # kept away from the resident matching, unless the same words are a resident's full name (then the resident
        # wins and the resident/family collision rule below applies).
        result = self._mask_residents(question, protected=self._family_full_spans(question))
        self._mask_family(question, result)
        self._mask_people_for_model(result)
        return result

    def _mask_people_for_model(self, result: MaskResult) -> None:
        """Names still in the masked text that belong to known people → person_N, in `model_text` only.

        Full names (2-4 words, typo tolerant like resident names) of staff members, and single capitalised words that
        are a staff member's first or last name ("What is Jone's designation?"). A single word is never the first word
        of the question unless it is possessive ("Jone's …"), and never a staff word ("Nurse", "Physician"), a family
        word or a title. Candidates of every kind are kept (staff, residents, family members), so the backend — not
        the model — decides who is meant."""
        if not self._staff:
            return
        from app.data.staff import staff_config  # local: the data layer imports this module

        cfg = staff_config()
        text = result.text
        tokens = [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]

        def placeholder(j: int) -> bool:
            return text[tokens[j][2] : tokens[j][2] + 1] == "_"  # resident_1 / family_2 / person_3

        def vocabulary_word(tok: str) -> bool:
            n = _norm(tok)
            return (n in _TITLE_WORDS or n in _STAFF_TITLES or _has_family_word([n])
                    or (cfg is not None and cfg.cue_in(tok.rstrip(".")) is not None and cfg.cue_in(tok.rstrip(".")).lower() == tok.lower().rstrip(".")))  # fmt: skip

        used: set[int] = set()
        found: list[tuple[int, int, dict]] = []
        for size in (4, 3, 2):
            for i in range(len(tokens) - size + 1):
                idx = range(i, i + size)
                if any(j in used or placeholder(j) or vocabulary_word(tokens[j][0]) for j in idx):
                    continue
                words = [_norm(tokens[j][0]) for j in idx]
                concat = "".join(words)
                if len(concat) < 5 or any(len(w) < 2 for w in words):
                    continue
                staff = [c.value for c in self._staff if len(c.tokens) > 1 and self._window_is_name(words, c.tokens)
                         and fuzz.ratio(concat, c.person) >= FULL_NAME_THRESHOLD]  # fmt: skip
                if staff:
                    found.append((tokens[i][1], tokens[i + size - 1][2], self._candidates(words, staff)))
                    used.update(idx)
        for j, (tok, start, end) in enumerate(tokens):
            if j in used or placeholder(j) or vocabulary_word(tok):
                continue
            possessive = tok.lower().replace("’", "'").endswith("'s")
            if j == 0 and not possessive:
                continue  # "Will you …", "May I …": a first word is not evidence of a name
            n = _norm(tok)
            if not tok[0].isupper():
                # lowercase ("what is jone's designation"): only a word the assistant's own wording never uses, in a
                # name position (possessive, after "is / does / named …", or after a title / staff word)
                prev = _norm(tokens[j - 1][0]) if j > 0 else ""
                titled = prev in _TITLE_WORDS or prev in _STAFF_TITLES or (cfg is not None and cfg.cue_in(prev) is not None)
                if n in _app_words() or not (possessive or prev in _NAME_SLOT_BEFORE or titled):
                    continue
            staff = [c.value for c in self._staff if n in c.tokens] if len(n) >= 3 else []
            if staff:
                found.append((start, end, self._candidates([n], staff)))
        # a resident placeholder from ONE word that is also a staff member's name ("Who is Jane?": resident Jane Austen
        # or staff member Jane Okafor) is a person_N for the model too; the deterministic stages keep resident_N
        for key, original in result.originals.items():
            words = _name_parts(original)
            staff = [c.value for c in self._staff if len(words) == 1 and words[0] in c.tokens]
            for m in re.finditer(rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![0-9])", text) if staff else []:
                cand = self._candidates(words, staff)
                cand["residents"] = list(dict.fromkeys([*result.placeholders.get(key, []), *cand["residents"]]))
                found.append((m.start(), m.end(), {**cand, "_original": original}))
        if not found:
            return
        taken = [int(k.split("_")[1]) for k in result.persons]
        number = max(taken, default=0)
        pieces, last = [], 0
        for start, end, cand in sorted(found, key=lambda x: x[0]):
            while end > start and text[end - 1] in ".-":  # "Sarah Jenkins." — the full stop is not part of the name
                end -= 1
            if text[end - 2 : end].lower() in ("'s", "\u2019s"):
                end -= 2  # the possessive stays outside the placeholder
            number += 1
            key = f"person_{number}"
            original = cand.pop("_original", None)
            result.model_persons[key] = cand
            result.entity_originals[key] = original or text[start:end]
            pieces.append(text[last:start] + key)
            last = end
        result.model_text = "".join(pieces) + text[last:]

    def _candidates(self, words: list[str], staff: list[str]) -> dict[str, list]:
        """Everyone a recognised name may be: the matched staff, plus residents and family members with that name."""
        residents = [r.resident_id for r in self._residents if all(w in [*r.first, *r.last] for w in words)]
        family = [self._family_view(m) for m in self._family if all(w in m.tokens for w in words)]
        return {"staff": list(dict.fromkeys(staff)), "residents": residents, "family": family}

    def people_hint(self, mask: "MaskResult") -> str | None:
        """For the model: what kind of person each person_N is — kinds and counts only, never a name."""
        persons = {**getattr(mask, "persons", {}), **getattr(mask, "model_persons", {})}
        lines = []
        for key in sorted(persons, key=lambda k: int(k.split("_")[1])):
            c = persons[key]
            kinds = [f"{label} ({len(c.get(k) or [])} record{'s' if len(c.get(k) or []) != 1 else ''})"
                     for k, label in (("staff", "staff member"), ("residents", "resident"), ("family", "family member"))
                     if c.get(k)]  # fmt: skip
            if kinds:
                lines.append(f"{key} = {' or '.join(kinds)}")
        return "- people named (the backend resolves them): " + "; ".join(lines) if lines else None

    def _family_full_spans(self, question: str) -> list[tuple[int, int]]:
        tokens = [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(question)]
        spans: list[tuple[int, int]] = []
        used: set[int] = set()
        for size in (4, 3, 2):
            for i in range(len(tokens) - size + 1):
                idx = range(i, i + size)
                if any(j in used for j in idx) or (i > 0 and _norm(tokens[i - 1][0]) in _STAFF_TITLES):
                    continue
                words = [_norm(tokens[j][0]) for j in idx]
                concat = "".join(words)
                if len(concat) < 6 or any(len(w) < 2 for w in words) or _has_family_word(words):
                    continue
                if not any(len(m.tokens) > 1 and self._window_is_name(words, m.tokens)
                           and fuzz.ratio(concat, m.person) >= FULL_NAME_THRESHOLD for m in self._family):  # fmt: skip
                    continue
                if self._best_full_match(concat, words):
                    continue  # a resident's full name: resident masking decides
                spans.append((tokens[i][1], tokens[i + size - 1][2]))
                used.update(idx)
        return spans

    def _mask_residents(self, question: str, protected: list[tuple[int, int]] | None = None) -> MaskResult:
        tokens = [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(question)]
        spans: list[tuple[int, int, list[str]]] = []
        # tokens inside a family member's full name are not resident words (empty without family names: unchanged)
        used: set[int] = {j for j, (_, s, _) in enumerate(tokens) if any(a <= s < b for a, b in protected or [])}

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

    # ------------------------------------------------------------------ family members
    def _family_view(self, m: _IndexedFamily) -> dict:
        return {"id": m.member_id, "residentId": m.resident_id, "facilityId": m.facility_id, "name": m.name,
                "rel": m.rel, "relation": m.relation}  # fmt: skip

    def _staff_for(self, words: list[str]) -> list[str]:
        concat = "".join(words)
        return [c.value for c in self._caregivers if len(c.tokens) > 1 and self._window_is_name(words, c.tokens)
                and fuzz.ratio(concat, c.person) >= FULL_NAME_THRESHOLD]  # fmt: skip

    def _mask_family(self, question: str, result: MaskResult) -> None:
        """Family-member spans OUTSIDE the resident spans, then a combined, typed rewrite of the masked text."""
        # resident spans in the original question, recovered from the resident masking (same order as resident_N)
        res_spans: list[tuple[int, int, str]] = []
        pos = 0
        for key in sorted(result.originals, key=lambda k: int(k.split("_")[1])):
            start = question.find(result.originals[key], pos)
            if start < 0:
                return  # cannot place the resident spans reliably: leave the resident-only result untouched
            res_spans.append((start, start + len(result.originals[key]), key))
            pos = start + len(result.originals[key])
        named_residents = {i.upper() for ids in result.placeholders.values() for i in ids}
        tokens = [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(question)]
        free = [not any(a <= s < b for a, b, _ in res_spans) for _, s, _ in tokens]
        used: set[int] = set()
        found: list[tuple[int, int, str, dict]] = []  # start, end, kind, candidates

        def titled(i: int) -> bool:
            return i > 0 and _norm(tokens[i - 1][0]) in _STAFF_TITLES

        # 1. full names (two or more words), typo tolerant like resident names; a titled name is staff only
        for size in (4, 3, 2):
            for i in range(len(tokens) - size + 1):
                idx = range(i, i + size)
                if any(j in used or not free[j] for j in idx) or titled(i):
                    continue
                words = [_norm(tokens[j][0]) for j in idx]
                concat = "".join(words)
                if len(concat) < 6 or any(len(w) < 2 for w in words) or _has_family_word(words):
                    continue
                fam = [m for m in self._family if len(m.tokens) > 1 and self._window_is_name(words, m.tokens)
                       and fuzz.ratio(concat, m.person) >= FULL_NAME_THRESHOLD]  # fmt: skip
                if not fam:
                    continue
                staff = self._staff_for(words)
                kind = "person" if staff else "family"
                found.append((tokens[i][1], tokens[i + size - 1][2], kind,
                              {"family": [self._family_view(m) for m in fam], "staff": staff, "residents": []}))  # fmt: skip
                used.update(idx)
        # 2. one word (67 family names are one word): only a member of a resident named in this question
        for j, (tok, start, end) in enumerate(tokens):
            if j in used or not free[j] or not tok[0].isupper() or titled(j):
                continue
            n = _norm(tok)
            if len(n) < 3 or _has_family_word([n]):
                continue
            fam = [m for m in self._family if m.resident_id in named_residents and (n == m.tokens[0] or [n] == m.tokens)]
            if not fam:
                continue
            staff = [c.value for c in self._caregivers if n in c.tokens]
            found.append((start, end, "person" if staff else "family",
                          {"family": [self._family_view(m) for m in fam], "staff": staff, "residents": []}))  # fmt: skip
            used.add(j)
        # 3. a resident's name that is also a family member of ANOTHER resident named here (spouses, siblings)
        collisions: dict[str, dict] = {}
        for start, end, key in res_spans:
            words = [_norm(t) for t in _TOKEN_RE.findall(question[start:end])]
            concat = "".join(words)
            mine = {i.upper() for i in result.placeholders[key]}
            fam = [m for m in self._family if m.resident_id in named_residents - mine and len(m.tokens) > 1
                   and self._window_is_name(words, m.tokens) and fuzz.ratio(concat, m.person) >= FULL_NAME_THRESHOLD]  # fmt: skip
            if fam:
                collisions[key] = {"family": [self._family_view(m) for m in fam], "staff": [],
                                   "residents": list(result.placeholders[key])}  # fmt: skip
        if not found and not collisions:
            return

        # rewrite: residents keep their numbers; family_N / person_N numbered by position
        spans: list[tuple[int, int, str, dict | None]] = []
        for start, end, key in res_spans:
            spans.append((start, end, "person" if key in collisions else key, collisions.get(key)))
        spans += [(s, e, k, c) for s, e, k, c in found]
        spans.sort(key=lambda x: x[0])
        counters = {"family": 0, "person": 0}
        last, pieces = 0, []
        for start, end, kind, cand in spans:
            if end >= 2 and question[end - 2 : end].lower() in ("'s", "\u2019s") and kind in ("family", "person"):
                end -= 2  # keep a possessive outside the placeholder, as resident masking does
            if kind in ("family", "person"):
                counters[kind] += 1
                name = f"{kind}_{counters[kind]}"
                if kind == "family":
                    result.family[name] = cand["family"]
                else:
                    result.persons[name] = cand
                result.entity_originals[name] = question[start:end]
                result.entity_spans.append((start, end))
            else:
                name = kind  # resident_N, unchanged
            pieces.append(question[last:start] + name)
            last = end
        for key in collisions:  # a resident span that became person_N is no longer a resident placeholder
            result.placeholders.pop(key, None)
            result.originals.pop(key, None)
        result.text = "".join(pieces) + question[last:]

    def family_member(self, member_id: str) -> dict | None:
        m = next((f for f in self._family if f.member_id == member_id), None)
        return self._family_view(m) if m else None

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
        if len(ids) > 1:
            in_order = [r.resident_id for r in self._residents if r.resident_id in ids and "".join(r.first + r.last) == concat]
            if len(in_order) == 1:
                ids = in_order
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
            # A caregiver recorded under a single word is named in full by that word. It must be written as a name
            # (capitalised, not the sentence's first word) so ordinary words never match a one-word staff record.
            # Such an exact name wins over a surname: "Jone" is the staff member recorded as "Jone", not "Hogan Jone".
            hits = list(self._single_word_caregivers.get(n, [])) if j > 0 and raw[:1].isupper() and len(n) >= 3 else []
            # a surname written in lower case must not be one of the assistant's own words ("staff members" is not
            # the surname of a caregiver recorded as "RM staff")
            if not hits and len(n) >= 4 and (raw[:1].isupper() or n not in _app_words()):
                hits = [c.value for c in self._caregivers if len(c.tokens) > 1 and n == c.tokens[-1]]
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
        return self._resolve_person(text, self._caregivers)

    def people_named(self, name: str, kind: str | None = None) -> dict[str, list[dict]]:
        """Every known person whose name contains ALL words of `name` (exact words, any order; staff titles ignored):
        {"residents": [{id, name}], "staff": [{name, designation, status}], "family": [{id, name, relation, residentId}]}.
        `kind` (resident / staff / family) limits the search. Exact words only: a typo is not a person here."""
        words = [t for t in _name_parts(name) if t not in _TITLE_WORDS]
        out: dict[str, list[dict]] = {"residents": [], "staff": [], "family": []}
        if not words:
            return out
        if kind in (None, "resident"):
            out["residents"] = [{"id": r.resident_id, "name": r.display} for r in self._residents
                                if all(w in [*r.first, *r.last] for w in words)]  # fmt: skip
            if len(out["residents"]) > 1 and len(words) > 1:  # "Sagar Dodia" written in order is not "Dodia Sagar"
                in_order = [p for p, r in ((p, next(r for r in self._residents if r.resident_id == p["id"])) for p in out["residents"])
                            if "".join(r.first + r.last) == "".join(words)]  # fmt: skip
                if len(in_order) == 1:
                    out["residents"] = in_order
        if kind in (None, "staff"):
            seen: set[tuple[str, str]] = set()
            # a name written exactly as recorded ("Care Coordinator Maya Patel") is that record, not every Maya Patel
            exact = " ".join(name.split()).lower()
            whole = any(c.value.lower() == exact for c in self._staff)
            for c, row in zip(self._staff, self._staff_rows, strict=True):
                key = (c.value, row["designation"])
                hit = c.value.lower() == exact if whole else all(w in c.tokens for w in words)
                if hit and key not in seen:
                    seen.add(key)
                    out["staff"].append(dict(row))
        if kind in (None, "family"):
            out["family"] = [{"id": m.member_id, "name": m.name, "relation": m.relation, "residentId": m.resident_id}
                             for m in self._family if all(w in m.tokens for w in words)]  # fmt: skip
        return out

    def name_is_unknown(self, name: str) -> bool:
        """True when no word of `name` is, or is close to, a word of any person's name this index holds (residents,
        staff, caregivers, family members): a question about that person has no answer anywhere, the model included
        (it only ever sees known names, as placeholders). A typo of a known name is NOT unknown (the usual matching
        and the model still get their chance)."""
        words = [t for t in _name_parts(name) if t not in _TITLE_WORDS]
        known = {*(t for r in self._residents for t in (*r.first, *r.last)), *self.staff_name_tokens(),
                 *(t for m in self._family for t in m.tokens)}  # fmt: skip
        return bool(words) and not any(
            w in known or any(fuzz.ratio(w, k) >= SINGLE_NAME_THRESHOLD for k in known) for w in words
        )

    def staff_name_tokens(self) -> set[str]:
        """Every word of a staff or caregiver name (titles excluded): "Dr. Raman" is a title + a name, not a role."""
        return {t for c in [*self._staff, *self._caregivers] for t in c.tokens}

    def resolve_staff(self, text: str) -> Resolution:
        """A staff member named in a staff question: every staff person (any status), not only assigned caregivers."""
        return self._resolve_person(text, self._staff or self._caregivers)

    def resolve_designation(self, text: str) -> Resolution:
        """A designation as recorded, compared without case and surrounding spaces ("case manager" → Case Manager)."""
        want = " ".join(str(text).split()).lower()
        hits = [d for d in self.staff_designations if d.lower() == want]
        if not hits and want.endswith("s"):  # "case managers"
            hits = [d for d in self.staff_designations if d.lower() == want[:-1]]
        return Resolution(hits)

    @staticmethod
    def _resolve_person(text: str, pool: list[_IndexedCaregiver]) -> Resolution:
        tokens = [t for t in _name_parts(text.split(",")[0]) if t not in _TITLE_WORDS]
        if not tokens:
            return Resolution([])
        joined = "".join(tokens)
        hits = [c for c in pool if all(t in c.tokens for t in tokens)]
        exact = [c for c in hits if c.person == joined]  # written exactly as recorded: that person, not every namesake
        if exact:
            hits = exact
        if not hits:
            hits = [c for c in pool if fuzz.ratio(joined, c.person) >= FULL_NAME_THRESHOLD]
        if not hits and len(tokens) == 1:
            hits = [c for c in pool if any(fuzz.ratio(tokens[0], t) >= SINGLE_NAME_THRESHOLD for t in c.tokens)]
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
