"""Question structure: what the question asks, ABOUT whom, and COMPARED WITH whom.

Built from the masked question (resident names → resident_N) with closed grammatical classes (intent_routes.yaml
`direct.structure`), the field words already used everywhere (`lookup_attributes`, `date_fields`) and the field kinds in
resident_schema.yaml `comparable`. No question-specific rules, names or values.

  • operation  — count / list / both / breakdown / aggregate, from the existing `route_cues`.
  • subjects   — residents the answer is about ("what is resident_1's room"): only these may be looked up.
  • references — residents other residents are compared with. A resident is a reference only as the OPERAND of a
                 comparison construction: "same … AS resident_1", "admitted BEFORE resident_1", "in common WITH
                 resident_1", "overlaps with the list OF resident_1", "different FROM resident_1"; a pronoun in that
                 position ("admitted after HER") stands for the nearest resident named before it.
  • comparisons — (field, comparison, reference) for each construction; the field from the field words in the same
                 clause, the comparison from the construction and the field's kind (date granularity, set overlap vs
                 set equality, ordering direction).
  • inclusion  — whether the reference itself belongs in the answer. It is excluded only when the ANSWER SET is
                 explicitly the others: "other" modifying the entity noun ("all other residents"), "else" after
                 who/anyone, an exclusion preposition governing the reference ("besides resident_1"), "the rest of".
  • cross_record — a comparison of residents among themselves with nobody named ("residents who share the same room").

Every part records whether it was determined, so a validator can refuse to accept what it could not establish.
"""

import re
from dataclasses import asdict, dataclass, field

from app.data.schema_config import comparable_fields
from app.nlu import placeholders as ph

_TOKEN = re.compile(ph.TOKEN + r"|[a-z]+(?:'[a-z]+)?|\d+|[,.?!;]")
_PLACEHOLDER = re.compile(r"^resident_\d+$")
_DETERMINERS = {"the", "a", "an", "their", "his", "her", "its", "this", "that"}


@dataclass
class Comparison:
    field: str | None  # None: the clause names no comparable field (undetermined)
    op: str | None  # None: the construction does not fix a comparison for that field (undetermined)
    reference: str  # resident_N
    cue: str  # the construction as written, for traces ("same day as", "before")


@dataclass
class Reference:
    key: str  # resident_N
    inclusion: str  # "include" | "exclude"
    inclusion_cue: str = ""  # what made it "exclude" ("other residents"), empty for include


@dataclass
class QuestionStructure:
    operations: list[str] = field(default_factory=list)  # count, list, breakdown, aggregate (several possible)
    subjects: list[str] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    comparisons: list[Comparison] = field(default_factory=list)
    cross_record: str = ""  # the cue when residents are compared among themselves (nobody named)
    undetermined: list[str] = field(default_factory=list)

    @property
    def is_comparison(self) -> bool:
        return bool(self.references or self.cross_record)

    @property
    def reference_keys(self) -> set[str]:
        return {r.key for r in self.references}

    def to_dict(self) -> dict:
        return asdict(self)


def _phrases(words) -> list[list[str]]:
    return sorted(([*str(w).lower().split()] for w in words or []), key=len, reverse=True)


class StructureAnalyzer:
    def __init__(self, routes_config: dict):
        direct = routes_config.get("direct") or {}
        cfg = direct.get("structure") or {}
        self.messages = {k: " ".join(str(v).split()) for k, v in (cfg.get("messages") or {}).items()}
        self.boundaries = {str(w).lower() for w in cfg.get("clause_boundaries") or []}
        self.frames = [
            {"words": _phrases(f.get("words")), "family": f["family"], "needs": f.get("needs"), "direction": f.get("direction")}
            for f in cfg.get("frames") or []
        ]
        self.classes = {
            name: _phrases(cfg.get(name))
            for name in ("equality_words", "exact_words", "overlap_words", "no_overlap_words", "difference_words")
        }
        ordering = cfg.get("ordering_words") or {}
        self.ordering = {d: _phrases(ws) for d, ws in ordering.items()}
        self.classes["ordering_words"] = [p for ws in self.ordering.values() for p in ws]
        self.implied = {str(k).lower(): v for k, v in (cfg.get("ordering_implies_field") or {}).items()}
        self.granularity = [(op, _phrases(ws)) for op, ws in (cfg.get("granularity") or {}).items()]
        self.cross_words = _phrases(cfg.get("cross_record_words"))
        ex = cfg.get("exclusion") or {}
        self.ex_modifiers = {str(w).lower() for w in ex.get("modifiers") or []}
        self.ex_else = {str(w).lower() for w in ex.get("else_after") or []}
        self.ex_preps = _phrases(ex.get("prepositions"))
        self.ex_rest = _phrases(ex.get("rest"))
        refs = direct.get("references") or {}
        self.pronouns = {str(w).lower() for w in refs.get("pronouns") or []}
        self.entity_nouns = {str(w).lower() for w in refs.get("entity_nouns") or []}
        # field words: attribute phrases + date anchors (longest first); kinds from the schema
        fw: list[tuple[list[str], str]] = []
        for attr, ws in (direct.get("lookup_attributes") or {}).items():
            fw += [(p, attr) for p in _phrases(ws)]
        for attr, ws in (direct.get("date_fields") or {}).items():
            fw += [(p, attr) for p in _phrases(ws)]
        self.field_words = sorted(fw, key=lambda x: len(x[0]), reverse=True)
        self.kinds = {f: c.get("kind") for f, c in comparable_fields().items()}
        cues = direct.get("route_cues") or []
        self.operation_cues: list[tuple[str, list[list[str]]]] = []
        for group in cues:
            routes = list(group.get("routes") or [])
            op = (
                "aggregate" if any(r in ("average_age",) for r in routes)
                else "breakdown" if any(r.startswith("breakdown") for r in routes)
                else "count" if "count" in routes
                else "list" if "list" in routes
                else "list" if any(r in ("oldest", "youngest") for r in routes)
                else None
            )  # fmt: skip
            if op:
                self.operation_cues.append((op, _phrases(group.get("words"))))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _at(words: list[str], i: int, phrase: list[str]) -> bool:
        return words[i : i + len(phrase)] == phrase

    def _find(self, words: list[str], phrases: list[list[str]], lo: int = 0, hi: int | None = None) -> list[tuple[int, list[str]]]:
        hi = len(words) if hi is None else hi
        return [(i, p) for i in range(lo, hi) for p in phrases if i + len(p) <= hi and self._at(words, i, p)]

    def _has(self, words, cls: str, lo: int, hi: int) -> bool:
        return bool(self._find(words, self.classes.get(cls) or [], lo, hi))

    def _clause_start(self, words: list[str], i: int) -> int:
        j = i - 1
        while j >= 0 and words[j] not in self.boundaries:
            j -= 1
        return j + 1

    def _fields_in(self, words: list[str], lo: int, hi: int) -> list[str]:
        found: list[str] = []
        taken: set[int] = set()
        for phrase, attr in self.field_words:
            for i, _ in self._find(words, [phrase], lo, hi):
                span = set(range(i, i + len(phrase)))
                if span & taken:
                    continue
                taken |= span
                if attr not in found:
                    found.append(attr)
        return found

    # ------------------------------------------------------------------ analysis
    def analyse(self, masked_question: str) -> QuestionStructure:
        words = [m.group(0) for m in _TOKEN.finditer(masked_question.lower())]
        out = QuestionStructure()
        for op, phrases in self.operation_cues:
            matches = self._find(words, phrases)
            if op == "count":
                matches = [
                    (i, phrase)
                    for i, phrase in matches
                    if not self._count_cue_is_part_of_field(words, i, phrase)
                ]
            if matches and op not in out.operations:
                out.operations.append(op)

        placeholders = [i for i, w in enumerate(words) if _PLACEHOLDER.match(w)]
        reference_keys: list[str] = []
        for i, w in enumerate(words):
            target = w if _PLACEHOLDER.match(w) else None
            if target is None and w in self.pronouns:  # "admitted after her" → the nearest resident named before it
                before = [p for p in placeholders if p < i]
                target = words[before[-1]] if before else None
            if target is None:
                continue
            frame, frame_at = self._frame_before(words, i)
            if frame is None:
                continue
            comps = self._comparisons(words, frame, frame_at, i, target)
            if not comps:
                continue
            out.comparisons.extend(comps)
            if target not in reference_keys:
                reference_keys.append(target)

        for key in reference_keys:
            cue = self._exclusion(words, key)
            out.references.append(Reference(key, "exclude" if cue else "include", cue))
        out.subjects = [w for w in dict.fromkeys(words[i] for i in placeholders) if w not in reference_keys]

        if not reference_keys:
            out.cross_record = self._cross_record(words)
        for c in out.comparisons:
            if c.field is None:
                out.undetermined.append(f"comparison field ({c.cue} {c.reference})")
            elif c.op is None:
                out.undetermined.append(f"comparison for {c.field} ({c.cue} {c.reference})")
        return out

    def _count_cue_is_part_of_field(self, words: list[str], start: int, phrase: list[str]) -> bool:
        """A field name such as "room number" must not turn "number of resident" into a count operation."""
        end = start + len(phrase)
        for field_phrase, _ in self.field_words:
            for field_start in range(max(0, start - len(field_phrase) + 1), end):
                field_end = field_start + len(field_phrase)
                if field_start < end and start < field_end and self._at(words, field_start, field_phrase):
                    return True
        return False

    def _frame_before(self, words: list[str], i: int):
        """The comparison construction whose last word sits directly before position i (determiners may intervene:
        "the caregiver list of resident_1" → "of"), or (None, -1)."""
        j = i - 1
        while j >= 0 and words[j] in _DETERMINERS:
            j -= 1
        in_field_phrase = self._inside_field_phrase(words, j)
        for frame in self.frames:
            for phrase in frame["words"]:
                start = j - len(phrase) + 1
                if start >= 0 and words[start : j + 1] == phrase:
                    if in_field_phrase:  # "looks after resident_1", "cares for": a field phrase, not a comparison
                        continue
                    lo = self._clause_start(words, start)
                    if frame["needs"] and not self._has(words, frame["needs"], lo, start):
                        continue
                    return frame, start
        return None, -1

    def _comparisons(self, words, frame, frame_at: int, operand_at: int, ref: str) -> list[Comparison]:
        lo = self._clause_start(words, frame_at)
        cue = " ".join(words[lo:operand_at]).strip()
        fields = self._fields_in(words, lo, frame_at)
        family = frame["family"]
        if family == "ordering":
            direction = frame.get("direction") or next(
                (d for d, ps in self.ordering.items() if self._find(words, ps, lo, frame_at)), None
            )
            implied = [f for w, f in self.implied.items() if w in words[lo:frame_at]]
            fields = fields or implied
            if not fields:  # "before"/"after" without a compared field is not a comparison ("ask after her")
                return [Comparison(None, None, ref, cue)] if not frame.get("direction") else []
            out = []
            for f in fields:
                kind = self.kinds.get(f)
                op = direction if kind in ("date", "number") and direction else None
                out.append(Comparison(f, op, ref, cue))
            return out
        if not fields:  # a comparison word was required for this construction: compared, but on what is unknown
            return [Comparison(None, None, ref, cue)] if frame.get("needs") else []
        exact = self._has(words, "exact_words", lo, frame_at)
        no_overlap = self._has(words, "no_overlap_words", lo, frame_at)
        out = []
        for f in fields:
            kind = self.kinds.get(f)
            op: str | None = None
            if kind is None:
                op = None  # not a comparable field
            elif family == "difference":
                op = "not_equal"  # for sets: not exactly the same set (NOT "no overlap")
            elif kind == "relation_set":
                op = "disjoint" if no_overlap else "same_set" if exact else "overlaps"
            elif kind == "date":
                op = next((g for g, ps in self.granularity if self._find(words, ps, lo, frame_at) and g == "same_day"), None)
                op = op or next((g for g, ps in self.granularity if self._find(words, ps, lo, frame_at) and g == "same_month"), None)
                op = op or next((g for g, ps in self.granularity if self._find(words, ps, lo, frame_at)), None) or "same_day"
            else:  # number, value, relation
                op = "not_equal" if no_overlap else "equal"
            out.append(Comparison(f, op, ref, cue))
        return out

    def _inside_field_phrase(self, words: list[str], j: int) -> bool:
        """Is words[j] part of a multi-word field phrase ("looks after", "cares for")?"""
        for phrase, _ in self.field_words:
            if len(phrase) < 2:
                continue
            for k in range(len(phrase)):
                if words[j - k : j - k + len(phrase)] == phrase and j - k >= 0:
                    return True
        return False

    def _exclusion(self, words: list[str], key: str) -> str:
        for i, w in enumerate(words):
            nxt = words[i + 1] if i + 1 < len(words) else ""
            if w in self.ex_modifiers and nxt in self.entity_nouns:
                return f"{w} {nxt}"
            if w == "else" and i > 0 and words[i - 1] in self.ex_else:
                return f"{words[i - 1]} else"
        for i, p in self._find(words, self.ex_preps):
            j = i + len(p)
            while j < len(words) and words[j] in _DETERMINERS:
                j += 1
            if j < len(words) and words[j] == key:
                return " ".join(p) + f" {key}"
        for i, p in self._find(words, self.ex_rest):
            rest = [w for w in words[i + len(p) : i + len(p) + 3] if w not in _DETERMINERS]
            if rest and rest[0] in self.entity_nouns:
                return " ".join(p) + f" {rest[0]}"
        return ""

    def _cross_record(self, words: list[str]) -> str:
        """Residents compared among themselves with nobody named: an equality/overlap word followed closely by a
        comparable field word ("share the same room number"), or an explicit duplicate word."""
        if (hit := self._find(words, self.cross_words)):
            return " ".join(hit[0][1])
        for i, p in self._find(words, (self.classes.get("equality_words") or []) + (self.classes.get("overlap_words") or [])):
            after = i + len(p)
            fields = self._fields_in(words, after, min(len(words), after + 4))
            if any(f in self.kinds for f in fields) and "as" not in words[after : after + 6]:
                return " ".join(words[i : after + 3])
        return ""
