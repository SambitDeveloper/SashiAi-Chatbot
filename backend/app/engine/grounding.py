"""Grounding check: the shared semantic validator for every plan (deterministic planner and local model alike).

Valid JSON and a schema-valid QueryPlan are not proof that the plan means what the user asked. After the resolver
(exact database values) and Pydantic (types, enums, limits), this module checks the plan against the question:

  • preserved — every value the question names (status, care type, gender, wing, caregiver, date phrase) is used;
  • not invented — every filter value is justified by the question's words (or by the previous plan for a follow-up);
  • dates — date bounds equal the deterministic resolution of the question's date phrase (dates.py), and the date
    field matches the question's verb (admitted → admission_date, discharged → discharge_date, ...).

Date bounds and the date field are canonicalised deterministically when unambiguous; every other problem is
reported, and the service retries the model once with the problem as a hint, then asks the user to clarify.
The synonym lists are application vocabulary (how staff talk), not database content.
"""

import re
from dataclasses import dataclass, field
from datetime import date as date_type

from rapidfuzz import fuzz

from app.data import text_match
from app.data.relations import (
    FAMILY_ATTRIBUTES,
    FAMILY_PLAN_FIELDS,
    FAMILY_SELECTORS,
    family_relation,
    plan_uses_family,
)
from app.data.repository import FIELD_MAP
from app.data.schema_config import output_fields, single_record_only
from app.data.staff import staff_config
from app.models.plan import (
    ALL_TEXT_MATCH_FIELDS,
    STAFF_RELATION_ATTRIBUTES,
    STAFF_RELATION_FIELDS,
    AgeRange,
    DateRange,
    QueryPlan,
)
from app.nlu.grammar import DateMatch, NumberMatch, parse_dates, parse_number_ranges
from app.nlu.missing_values import MissingValueDetector
from app.nlu.projection import ProjectionDetector, value_starts
from app.nlu.references import ReferenceResolver
from app.nlu.structure import QuestionStructure
from app.nlu.vocabulary import _TITLE_WORDS, EntityIndex, MaskResult, _name_parts

# Words that signal date language. If one is present but grammar.parse_dates resolved nothing around it, the
# question uses a date form we do not resolve deterministically ("3 weeks ago", "between March and May").
# "recent"/"recently" are deliberately absent: they ask for an ordering, never justify a guessed date range.
_DATE_WORDS = {
    "today", "yesterday", "tomorrow", "week", "weeks", "month", "months", "year", "years", "quarter", "ago",
    "since", "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december", "sept",
}  # fmt: skip


@dataclass
class DateScan:
    """Date and number expressions found in the question, resolved deterministically by grammar.py."""

    dates: list[DateMatch]
    numbers: list[NumberMatch]
    texts: dict[int, str]  # DateMatch.start → the words of the phrase
    unresolved: bool  # date language present that the grammar did not resolve

    def hints(self) -> list[str]:
        out = []
        for d in self.dates:
            r = d.range
            if "start" in r and "end" in r:
                out.append(f"'{self.texts[d.start]}' = {r['start']} to {r['end']}")
            elif "start" in r:
                out.append(f"'{self.texts[d.start]}' = from {r['start']} (no end)")
            else:
                out.append(f"'{self.texts[d.start]}' = until {r['end']} (no start)")
        return out


def scan_question(text: str, today: date_type) -> DateScan:
    low = text.lower()
    dates = parse_dates(low, today)
    numbers = [n for n in parse_number_ranges(low) if not any(d.start < n.end and n.start < d.end for d in dates)]
    taken = [(d.start, d.end) for d in dates]
    unresolved = any(
        w.group(0) in _DATE_WORDS and not any(s <= w.start() < e for s, e in taken) for w in re.finditer(r"[a-z]+", low)
    )
    return DateScan(dates, numbers, {d.start: text[d.start : d.end] for d in dates}, unresolved)


def date_fields_mentioned(question: str, anchors: dict[str, list[str]]) -> list[str]:
    low = question.lower()
    return [f for f, words in anchors.items() if any(_has(w, low) for w in words)]

# value → words that justify it. A value is "named" by its first (canonical) forms; every form justifies it.
SYNONYMS: dict[str, dict[str, tuple[str, ...]]] = {
    "status": {
        "active": ("active", "current", "currently living", "living here", "in residence"),
        "away": ("away", "hospital", "hospitalized", "hospitalised", "on leave", "leave", "absent", "out of facility"),
        "inactive": ("inactive",),
        "pending": ("pending", "waitlist", "wait list", "waiting list", "waiting", "awaiting", "prospective"),
        "admitted": ("admitted", "newly admitted", "new admission", "new admissions"),
        "discharged": ("discharged", "discharge", "discharges", "left", "moved out", "former", "departed"),
    },
    "care_type": {
        "memory_care": ("memory care", "memory", "dementia", "alzheimer", "alzheimers", "alzheimer's"),
        "assisted_living": ("assisted living", "assisted"),
        "independent_living": ("independent living", "independent"),
        "skilled_nursing": ("skilled nursing", "skilled", "nursing care", "snf", "rehab", "rehabilitation"),
        "respite_care": ("respite care", "respite", "short stay", "short-stay", "short term", "short-term"),
    },
    "gender": {
        "Female": ("female", "females", "women", "woman", "ladies", "lady"),
        "Male": ("male", "males", "men", "man", "gentlemen", "gentleman"),
        "Other": ("other gender", "non-binary", "nonbinary", "unspecified gender"),
    },
}
# Forms that, when written, mean the user NAMED that value (must appear in the plan). Looser synonyms only justify.
NAMED: dict[str, dict[str, tuple[str, ...]]] = {
    "status": {v: (v,) for v in SYNONYMS["status"]},
    "care_type": {
        "memory_care": ("memory care", "dementia"),
        "assisted_living": ("assisted living",),
        "independent_living": ("independent living",),
        "skilled_nursing": ("skilled nursing",),
        "respite_care": ("respite",),
    },
    "gender": {"Female": ("female", "females", "women"), "Male": ("male", "males", "men")},
}
# Negation in the question. An exclusion must be justified by one; a negated question must produce one.
_NEGATION = re.compile(
    r"\b(?:not|non|never|without|except|excluding|exclude|excludes|neither|nor|outside|besides)\b|n't\b|"
    r"\bother than\b|\bapart from\b"
)
_AGE_WORDS = re.compile(r"\b(?:octogenarian|nonagenarian|centenarian|sixties|seventies|eighties|nineties)\b")
_NUMBER_WORDS = re.compile(
    r"\b(?:\d+|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|octogenarian|nonagenarian|"
    r"centenarian|seventies|eighties|nineties|sixties)\b"
)


def _has(phrase: str, low: str) -> bool:
    return re.search(r"(?<![a-z])" + r"\s+".join(map(re.escape, phrase.split())) + r"(?![a-z])", low) is not None


# ---------------------------------------------------------------------------------------------- structure vocabulary
# Topics the resident records do not hold. A question about them is refused before any model call (the model would
# otherwise "answer" with an unrelated lookup). Application policy, not database content.
OUT_OF_SCOPE = (
    "medication", "medications", "medicine", "medicines", "meds", "prescription", "prescriptions", "drug", "drugs",
    "dose", "dosage", "diagnosis", "diagnoses", "diagnosed", "diabetes", "diabetic", "dementia diagnosis", "allergy",
    "allergies", "allergic", "diet", "dietary", "meal", "meals", "billing", "bill", "bills", "invoice", "invoices",
    "balance", "owe", "owes", "payment", "payments", "insurance", "visitor",
    "visitors", "vitals", "blood pressure", "weight",
)  # fmt: skip

GROUP_WORDS: dict[str, tuple[str, ...]] = {
    "status": ("status", "statuses"),
    "care_type": ("care type", "care types", "type of care", "care level", "care levels", "level of care", "category",
                  "categories", "care category"),
    "facility": ("wing", "wings", "facility", "facilities", "building", "buildings", "unit", "units", "location"),
    "gender": ("gender", "genders", "sex"),
    "primary_caregiver": ("caregiver", "caregivers", "caretaker", "caretakers", "nurse", "nurses", "staff", "aide",
                          "aides", "carer", "carers", "staff member"),
}  # fmt: skip
GROUP_CUES = ("per", "each", "every", "by", "breakdown", "split", "distribution", "wise", "grouped", "across")
# question words → (sort field, direction); checked in order
SORT_CUES: list[tuple[tuple[str, ...], tuple[str, str]]] = [
    (("oldest", "eldest"), ("age", "desc")),
    (("youngest",), ("age", "asc")),
    (("most recently", "most recent", "latest", "newest", "last admitted", "recently admitted"), ("admission_date", "desc")),
    (("earliest", "first admitted", "longest staying", "longest-staying"), ("admission_date", "asc")),
    (("alphabetical", "alphabetically", "by name", "by last name"), ("last_name", "asc")),
    # appended (positions 2 and 3 above are read by index as the recency cues)
    (("most family members", "most relatives", "most family contacts", "largest family", "biggest family"),
     ("family_count", "desc")),
    (("fewest family members", "fewest relatives", "least family members", "fewest family contacts", "smallest family"),
     ("family_count", "asc")),
]
AGG_CUES = {"avg": ("average", "mean", "avg"), "min": ("minimum", "min", "lowest"), "max": ("maximum", "max", "highest")}
COUNT_CUES = ("how many", "count", "number of", "total", "census", "headcount", "tally")
_SMALL_NUMBERS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen twenty".split())}  # fmt: skip


def out_of_scope_terms(question: str) -> list[str]:
    low = question.lower()
    return [t for t in OUT_OF_SCOPE if _has(t, low)]


def _list_size(low: str, scan: "DateScan") -> int | None:
    """An explicit result size ("top 5", "the three oldest") that is not part of a date or age expression."""
    taken = [(d.start, d.end) for d in scan.dates] + [(n.start, n.end) for n in scan.numbers]
    for m in re.finditer(r"\b(\d{1,3}|" + "|".join(_SMALL_NUMBERS) + r")\b", low):
        if any(a <= m.start() < b for a, b in taken):
            continue
        tail = low[m.end() : m.end() + 40]
        if re.match(r"\s+(?:years?|yrs?|days?|weeks?|months?)\b", tail):
            continue
        word = m.group(1)
        n = int(word) if word.isdigit() else _SMALL_NUMBERS[word]
        return n if 1 <= n <= 200 else None
    return None


def _check_structure(
    plan: QueryPlan, low: str, scan: "DateScan", g: "Grounding", implied_group_fields: set[str] | None = None
) -> None:
    """Action, grouping, ordering, limit and aggregate must follow from the question's words."""
    groups = plan.group_by or []
    named_groups = [f for f, words in GROUP_WORDS.items() if any(_has(w, low) for w in words)]
    named_groups.extend(f for f in implied_group_fields or set() if f not in named_groups)
    grouping_cue = any(_has(c, low) for c in GROUP_CUES)
    for f in groups:
        if f not in named_groups:
            g.invented.append(f"grouping by {f.replace('_', ' ')}")
    if not groups and plan.action in ("count", "aggregate", "breakdown", "list") and grouping_cue:
        # "per wing", "by caregiver", "each status" — only when the cue word is next to a field word
        for f in named_groups:
            for w in GROUP_WORDS[f]:
                for c in GROUP_CUES:
                    if re.search(rf"(?<![a-z]){c}\s+(?:\w+\s+)?{re.escape(w)}(?![a-z])|(?<![a-z]){re.escape(w)}\s+{c}(?![a-z])", low):
                        g.missing.append(f"grouping by {f.replace('_', ' ')}")
                        break
                else:
                    continue
                break

    # aggregate operation
    agg_ops = [op for op, words in AGG_CUES.items() if any(_has(w, low) for w in words)]
    if plan.action == "aggregate":
        if not agg_ops:
            g.invented.append("an average/minimum/maximum")
        elif len(agg_ops) == 1 and plan.aggregate.op != agg_ops[0]:
            g.fixes.append(f"aggregate {plan.aggregate.op} → {agg_ops[0]}")
            plan.aggregate.op = agg_ops[0]
    elif agg_ops and _has("age", low) and plan.action in ("count", "list", "breakdown"):
        g.missing.append({"avg": "average", "min": "minimum", "max": "maximum"}[agg_ops[0]] + " age")

    # counting questions answered with a plain list
    if (
        plan.action == "list"
        and not plan.sort
        and not plan.limit
        and not plan.output_fields
        and any(_has(c, low) for c in COUNT_CUES)
    ):
        plan.action = "count"
        g.fixes.append("list → count (question asks how many)")

    if plan.action != "list":
        if plan.sort or plan.limit:  # ordering has no effect on a count/breakdown/aggregate; drop it
            plan.sort, plan.limit = None, None
        return
    cue = next((target for words, target in SORT_CUES if any(_has(w, low) for w in words)), None)
    size = _list_size(low, scan)
    if cue:
        from app.models.plan import Sort

        if plan.sort is None or (plan.sort.field, plan.sort.direction) != cue:
            g.fixes.append(f"sort set to {cue[0]} {cue[1]}")
            plan.sort = Sort(field=cue[0], direction=cue[1])
        if size:
            want = size
        elif re.search(r"\b(?:residents|ones|people|patients)\b", low):
            want = plan.limit  # "the oldest residents": how many is not stated; keep what the plan has
        else:
            want = 1  # "who is the oldest resident"
        if want != plan.limit:
            g.fixes.append(f"limit {plan.limit} → {want}")
            plan.limit = want
    else:
        if plan.sort is not None:
            g.fixes.append("dropped an ordering the question does not ask for")
            plan.sort = None
        if plan.limit is not None and plan.limit != size:
            g.fixes.append("dropped a result limit the question does not state")
            plan.limit = size

@dataclass
class Grounding:
    plan: QueryPlan
    missing: list[str] = field(default_factory=list)
    invented: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)  # deterministic canonicalisations applied (for logs)
    # Problems no model output can fix (unsupported structure, conversation context, schema/policy limits):
    # the service asks the user right away instead of retrying the model.
    blocking: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.invented and not self.blocking

    @property
    def correctable(self) -> bool:
        """Failed only on things a second model attempt could fix (dropped or invented conditions/fields)."""
        return not self.ok and not self.blocking

    def hint(self) -> str:
        parts = []
        if self.missing:
            parts.append(f"The question names {', '.join(self.missing)}: every one must be a condition.")
        if self.invented:
            parts.append(f"Remove conditions the question does not state: {', '.join(self.invented)}.")
        return " ".join(parts)

    def problems(self) -> list[str]:
        return [*self.blocking, *self.missing, *(f"not asked: {x}" for x in self.invented)]


# Text-match phrases ("room starts with A", "facility containing 'Garden'"). The needle is the quoted text, or the one
# word after the cue (optionally after "the letter/prefix/text/word"). Used both ways: a phrase the plan lacks is a
# dropped condition, and a plan needle that no such phrase supports is invented.
_TEXT_CUES = {
    "starts_with": r"(?:starts?|starting|begins?|beginning)\s+with",
    "contains": r"(?:contains?|containing)",
}
_TEXT_LEAD = r"\s+(?:(?:the|a|an)\s+(?:letters?|prefix|text|word|string|characters?)\s+)?"
_TEXT_NEEDLE = r"(?:\"([^\"]{1,60})\"|'([^']{1,60})'|“([^”]{1,60})”|‘([^’]{1,60})’|([A-Za-z0-9][\w\-/]*))"
_TEXT_STOP = {"a", "an", "the", "any", "all", "more", "most", "fewer", "less", "only", "at", "no", "some", "each",
              "every", "than", "their", "his", "her", "its"}  # fmt: skip


def text_match_phrases(question: str) -> list[tuple[str, str]]:
    """(operator, needle) for every text-match phrase in the question, in order."""
    out: list[tuple[str, str]] = []
    for op, cue in _TEXT_CUES.items():
        for m in re.finditer(r"\b" + cue + _TEXT_LEAD + _TEXT_NEEDLE, question, re.IGNORECASE):
            quoted = next((grp for grp in m.groups()[:4] if grp), None)
            needle = quoted if quoted is not None else m.group(5)
            if quoted is None and needle.lower() in _TEXT_STOP:
                continue  # "contains the most", "start with a B": not a needle (the invented check still finds "B")
            out.append((op, needle.strip()))
    return out


def _needle_after_cue(op: str, needle: str, question: str) -> bool:
    """The plan's needle follows a cue of the same operator within a few words (quotes allowed)."""
    pattern = r"\b" + _TEXT_CUES[op] + r"\W+(?:[\w'’-]+\W+){0,3}?[\"'“‘]?" + re.escape(needle) + r"(?![\w])"
    return re.search(pattern, question, re.IGNORECASE) is not None


def _check_text_matches(f, question: str, prev: dict, g: "Grounding") -> None:
    used: list[tuple[str, str, str]] = []  # (field, op, needle)
    for name in ALL_TEXT_MATCH_FIELDS:
        sf = getattr(f, name)
        for op in ("starts_with", "contains"):
            for needle in (getattr(sf, op) or []) if sf is not None else []:
                used.append((name, op, needle))
    for name, op, needle in used:
        kept = needle in ((prev.get(name) or {}).get(op) or [])  # carried over from the previous question
        if not kept and not _needle_after_cue(op, needle, question):
            g.invented.append(f'{name.replace("_", " ")} {op.replace("_", " ")} "{needle}"')
    for op, needle in text_match_phrases(question):
        if not any(o == op and n.casefold() == needle.casefold() for _, o, n in used):
            g.missing.append(f'the text match {op.replace("_", " ")} "{needle}"')


def _text_covers(sf, view_field: str, values: list[str]) -> bool:
    """A facility/caregiver the question mentions is covered by a text match on that field (same semantics as execution)."""
    return sf is not None and sf.has_text_match and any(
        text_match.value_matches(v, view_field, sf.starts_with, sf.contains) for v in values
    )


_FAMILY_COUNT_AFTER = re.compile(r"\s*(?:family members?|relatives?|family contacts?|family)\b")


def _check_family(plan: QueryPlan, question: str, prev: dict, g: "Grounding", masked: str | None = None) -> str | None:
    """A question about family must use a family field; a family condition needs a family word in the question (or
    the previous question, for a follow-up). Returns the family word found (or None)."""
    rel = family_relation()
    if rel is None:
        return None
    cue = rel.cue_in(question, masked)
    uses = plan_uses_family(plan.filters, sort_field=plan.sort.field if plan.sort else None) or any(
        a in FAMILY_ATTRIBUTES for a in plan.output_fields
    )
    if cue and not uses:
        g.missing.append(f'the family members the question asks about ("{cue}")')
    if not cue and plan_uses_family(plan.filters) and not any(k in FAMILY_PLAN_FIELDS for k in prev):
        g.invented.append("a family condition")
    return cue


_INACTIVE_WORDS = ("inactive", "former", "deactivated", "disabled", "no longer", "not active", "including", "all",
                   "every", "both", "past", "previous", "left")  # fmt: skip
_RESIDENT_WORDS = re.compile(r"\b(?:residents?|patients?|clients?|people|persons?)\b")


def _without_care_types(question: str) -> str:
    """The question with care-type phrases blanked out: "skilled nursing" is a care type, not the role "nursing"."""
    out = question
    for forms in SYNONYMS["care_type"].values():
        for form in sorted(forms, key=len, reverse=True):
            out = re.sub(rf"(?<![a-z]){re.escape(form)}(?![a-z])", lambda m: " " * len(m.group(0)), out, flags=re.I)
    return out


def _is_staff_word(text: str, index: EntityIndex) -> bool:
    """A mention that is wholly staff vocabulary (a role word, "staff", a recorded designation), not a name — some staff
    records are named after their role ("Dining Staff", "Physician Test")."""
    cfg = staff_config()
    t = re.sub(r"(?:'s|’s)$", "", text.strip(), flags=re.I)
    if cfg is None or not t:
        return False
    cue = cfg._cue_pattern.fullmatch(t) if cfg._cue_pattern else None
    return bool(cue) or any(_designation_supported(t, d) and len(d) >= len(t) - 1 for d in index.staff_designations)


def _role_values_supported(cfg, question: str, values: list[str]) -> list[str]:
    """Role values no word of the question names ("physician" needs physician/doctor/dr/...)."""
    named = {r for _, roles in cfg.roles_in(_without_care_types(question)) for r in roles}
    return [v for v in values if v not in named]


def _designation_supported(question: str, value: str) -> bool:
    low, v = question.lower(), value.lower()
    return v in low or (v.endswith("s") and v[:-1] in low) or f"{v}s" in low


def _check_staff(plan: QueryPlan, question: str, index: EntityIndex, prev: dict, g: "Grounding", mask=None) -> None:
    """Resident plans: a question naming a staff ROLE ("residents without a case manager", "Merry's physician") must use
    a staff field or show the care team / primary physician; a role or designation condition needs words that name it.
    Generic staff words ("nurse", "staff") and titles before a name ("Dr. Raman") demand nothing (primary caregiver)."""
    cfg = staff_config()
    if cfg is None:
        return
    f = plan.filters
    specific = cfg.specific_roles_in(_without_care_types(question), index.staff_name_tokens())
    uses = any(getattr(f, n) is not None for n in STAFF_RELATION_FIELDS) or bool(
        set(plan.output_fields) & set(STAFF_RELATION_ATTRIBUTES)
    )
    if specific and not uses and not any(k in STAFF_RELATION_FIELDS for k in prev):
        g.missing.append(f'the staff the question asks about ("{specific[0][0]}")')
    # "how many nurses are there?", "list the staff": the rows are staff members, never residents. A question that
    # speaks of residents, names one, or names a staff member ("residents of Nurse Sarah Jenkins") stays a resident one.
    cue = cfg.cue_in(_without_care_types(question))
    names_someone = bool(mask is not None and mask.placeholders) or bool(index.mentioned_caregivers(question))
    if cue and not _RESIDENT_WORDS.search(question.lower()) and not names_someone and not prev:
        g.missing.append(f'staff members as the rows ("{cue}"): this is a question about staff, not residents')
    if f.staff_role is not None and "staff_role" not in prev:
        for v in _role_values_supported(cfg, question, [*(f.staff_role.include or []), *(f.staff_role.exclude or [])]):
            g.invented.append(f"staff role {v.replace('_', ' ')}")
    if f.staff_designation is not None and "staff_designation" not in prev:
        for v in [*(f.staff_designation.include or []), *(f.staff_designation.exclude or [])]:
            role_named = _role_values_supported(cfg, question, [cfg.role_for(v) or ""]) == []
            if not _designation_supported(question, v) and not role_named:
                g.invented.append(f'designation "{v}"')
    if f.staff_count is not None and not re.search(r"\d|\b(?:no|any|none|zero|one|two|three|several|multiple|more|fewer|less|most)\b", question.lower()):
        g.invented.append("a number of assigned staff")


def check_staff_plan(
    plan: QueryPlan,
    question: str,
    mask: MaskResult,
    index: EntityIndex,
    previous_filters: dict | None = None,
    uses_previous: bool = False,
) -> "Grounding":
    """A staff plan (the rows are staff members) answers THIS question: the question is about staff, every role word is
    a condition (or the grouping), conditions and names come from the question's words, inactive staff only when asked
    for, and a resident status condition only when the question speaks of residents ("active physicians" is the staff
    member's status)."""
    g = Grounding(plan)
    cfg = staff_config()
    if cfg is None:
        g.blocking.append("staff information is not configured")
        return g
    low = question.lower()
    prev = (previous_filters or {}) if uses_previous else {}
    f = plan.filters
    designation_named = any(_designation_supported(question, d) for d in index.staff_designations)
    if not cfg.cue_in(question) and not designation_named and not plan.lookup and not prev:
        g.invented.append("a question about staff members (the question names no staff)")
    groups = set(plan.group_by or [])
    roles = cfg.specific_roles_in(_without_care_types(question), index.staff_name_tokens(), include_generic=True)
    role_cond = f.staff_role is not None or f.staff_designation is not None
    if roles and not role_cond and not groups & {"staff_role", "staff_designation"} and not plan.lookup and "staff_role" not in prev:
        g.missing.append(f'the staff role "{roles[0][0]}"')
    if f.staff_role is not None and "staff_role" not in prev:
        for v in _role_values_supported(cfg, question, [*(f.staff_role.include or []), *(f.staff_role.exclude or [])]):
            g.invented.append(f"staff role {v.replace('_', ' ')}")
    if f.staff_designation is not None and "staff_designation" not in prev:
        for v in [*(f.staff_designation.include or []), *(f.staff_designation.exclude or [])]:
            if not _designation_supported(question, v):
                g.invented.append(f'designation "{v}"')
    if f.staff_status is not None and "inactive" in (f.staff_status.include or []) and not any(_has(w, low) for w in _INACTIVE_WORDS):
        g.invented.append("inactive staff (the question does not ask for them)")
    if f.status is not None and not _RESIDENT_WORDS.search(low) and "status" not in prev:
        g.invented.append("a resident status (the question's status words describe the staff member: use staff_status)")
    if groups and not any(_has(c, low) for c in (*GROUP_CUES, "breakdown")):
        g.invented.append("grouping (the question asks for no breakdown)")

    words = re.findall(r"[a-z][a-z'’-]*", low)

    def named(name: str) -> bool:
        parts = [p for p in _name_parts(name.split(",")[0]) if p not in _TITLE_WORDS]
        return any(len(p) >= 3 and any(fuzz.ratio(p, w.replace("'", "")) >= 85 for w in words) for p in parts)

    people = [*(plan.lookup.staff or [] if plan.lookup else []), *((f.staff_name.include or []) + (f.staff_name.exclude or []) if f.staff_name else [])]
    for name in people:
        if not named(name) and name not in (prev.get("staff_name") or {}).get("include", []):
            g.invented.append(f'staff member "{name}"')
    for name, values in index.mentioned_caregivers(question).items():
        if _is_staff_word(name, index):  # "staff", "physicians": the question's wording, not a person's name
            continue
        if not set(values) & set(people) and not plan.lookup and f.staff_name is None and f.primary_caregiver is None:
            g.missing.append(f'staff member "{name}"')
    return g


def check_plan(
    plan: QueryPlan,
    question: str,
    mask: MaskResult,
    index: EntityIndex,
    scan: DateScan,
    date_anchors: dict[str, list[str]],
    previous_filters: dict | None = None,
    uses_previous: bool = False,
    missing_detector: MissingValueDetector | None = None,
    reference_resolver: ReferenceResolver | None = None,
    projection_detector: ProjectionDetector | None = None,
    structure: QuestionStructure | None = None,
) -> Grounding:
    """previous_filters: the conversation's previous plan filters (None when there is none);
    uses_previous: the planner says the question refines that plan."""
    g = Grounding(plan)
    if not plan.is_data_query:
        return g
    if plan.entity == "staff":
        return check_staff_plan(plan, question, mask, index, previous_filters, uses_previous)
    low = question.lower()
    # ---------------------------------------------------------------- structure: does the plan answer THIS question?
    if structure is not None:
        _check_structure_semantics(plan, low, mask, structure, g)
    # A follow-up ("of them", "those", "what about ...") refines the previous plan: its conditions may be reused,
    # and must not be silently dropped (dropping them answers a different, broader question).
    # Whether a pronoun refers within this question or to the previous turn is decided by the shared resolver
    # ("residents … their rooms" is local; "how many of them" points back).
    if reference_resolver is not None:
        resolution = reference_resolver.resolve(mask.text, previous_filters is not None)
        referential, trigger = resolution.refers_to_previous, resolution.trigger
    else:  # callers without a resolver (older tests): the previous, conservative reading
        found = re.search(r"\b(?:them|those|these|they|their)\b|\b(?:what|how) about\b", low)
        referential, trigger = bool(found), found.group(0) if found else ""
    prev = (previous_filters or {}) if (uses_previous or referential) else {}
    if referential and previous_filters is None:  # no model output can say whom the user meant: ask
        g.blocking.append(f"what '{trigger}' refers to (there is no earlier question in this conversation)")
    elif referential and previous_filters:
        changed = {f for f in plan.filters.model_fields if getattr(plan.filters, f) is not None} | set(plan.group_by or [])
        for f in previous_filters:
            if f not in changed:
                g.missing.append(f"the earlier condition on {f.replace('_', ' ')}")
    plan = plan.model_copy(deep=True)
    f = plan.filters
    groups = set(plan.group_by or [])
    attrs = set(plan.lookup.attributes) if plan.lookup else set()
    date_cues = date_fields_mentioned(question, date_anchors)

    # "admitted last month" is an admission date, not the status "admitted"
    status_text = low
    recency = any(_has(w, low) for w in SORT_CUES[2][0] + SORT_CUES[3][0])
    if (scan.dates and "admission_date" in date_cues) or recency:  # "admitted" used as a verb, not the status
        status_text = re.sub(r"\badmitted\b", " ", status_text)

    def prev_values(name: str) -> set[str]:
        p = prev.get(name) or {}
        return set(p.get("include", []) or []) | set(p.get("exclude", []) or [])

    # ---------------------------------------------------------------- fixed-vocabulary fields
    for name in ("status", "care_type", "gender"):
        text = status_text if name == "status" else low
        sf = getattr(f, name)
        used = set((sf.include or []) + (sf.exclude or [])) if sf else set()
        for value in used:
            forms = SYNONYMS[name].get(value, (value.replace("_", " "),))
            if not any(_has(x, text) for x in forms) and value not in prev_values(name):
                g.invented.append(f"{name.replace('_', ' ')} {value}")
        # a grouped field names no single value; an attribute exempts only when no value is named
        # ("what is resident_1's care type"), not "does resident_1 receive memory care" (a yes/no check)
        if name in groups:
            continue
        for value, forms in NAMED[name].items():
            if name == "status" and value not in index.statuses:
                continue
            if name == "care_type" and value not in index.care_types:
                continue
            if any(_has(x, text) for x in forms) and value not in used:
                # a date verb ("admitted last month", "discharged this year") anchors a date filter; the date filter
                # already expresses it, so the status itself is optional
                if name == "status" and scan.dates and (
                    (value == "admitted" and "admission_date" in date_cues)
                    or (value == "discharged" and f.discharge_date is not None)
                ):
                    continue
                g.missing.append(f'{name.replace("_", " ")} "{value.replace("_", " ")}"')

    # ---------------------------------------------------------------- polarity (include vs exclude), every set field
    negated = bool(_NEGATION.search(low))
    excluded = {
        n: set(getattr(f, n).exclude or [])
        for n in ("status", "care_type", "gender", "facility", "primary_caregiver", "room_number", "staff_role",
                  "staff_designation")
        if getattr(f, n) is not None and getattr(f, n).exclude
    }
    if not negated:
        for n, values in excluded.items():
            prev_ex = set((prev.get(n) or {}).get("exclude", []) or [])
            for v in sorted(values - prev_ex):
                g.invented.append(f"exclusion of {n.replace('_', ' ')} {v} (the question has no negation)")
    else:
        # A negation applies to what FOLLOWS it in the same clause: "memory care residents not in Garden Terrace"
        # excludes Garden Terrace, never memory care. An excluded value the question names outside any negation's
        # scope was inverted by the planner.
        for n, values in excluded.items():
            prev_ex = set((prev.get(n) or {}).get("exclude", []) or [])
            for v in sorted(values - prev_ex):
                spans = _value_mentions(n, v, low, question, index)
                if spans and not any(_negation_scopes(low, start) for start, _ in spans):
                    g.invented.append(f"exclusion of {n.replace('_', ' ')} {v} (the negation applies to something else)")
    if (
        negated
        and not excluded
        and not (f.age and f.age.max is not None)
        and not any(getattr(f, n) is not None and getattr(f, n).missing is True for n in _ALL_FIELDS)  # "without a room"
        and plan.action != "lookup"
    ):
        g.missing.append("the negation in the question (\"not\", \"except\", \"neither\" ...)")

    # ---------------------------------------------------------------- database-name fields
    words = re.findall(r"[a-z][a-z'’-]*", low)

    def fuzzy_in_question(token: str) -> bool:
        return len(token) >= 3 and any(fuzz.ratio(token, w.replace("'", "")) >= 85 for w in words)

    if f.facility:
        mentioned = {w for vals in index.mentioned_wings(question).values() for w in vals}
        for wing in (f.facility.include or []) + (f.facility.exclude or []):
            parts = [p for p in _name_parts(wing) if p != "wing"]
            if wing not in mentioned and not any(fuzzy_in_question(p) for p in parts) and wing not in prev_values("facility"):
                g.invented.append(f'facility "{wing}"')
    if f.primary_caregiver:
        for cg in (f.primary_caregiver.include or []) + (f.primary_caregiver.exclude or []):
            parts = [p for p in _name_parts(cg.split(",")[0]) if p not in _TITLE_WORDS]
            if not any(fuzzy_in_question(p) for p in parts) and cg not in prev_values("primary_caregiver"):
                g.invented.append(f'caregiver "{cg}"')
    if f.room_number:
        for room in (f.room_number.include or []) + (f.room_number.exclude or []):
            digits = re.sub(r"\D", "", room)
            if digits and digits not in re.sub(r"\D", " ", question).split() and room not in prev_values("room_number"):
                g.invented.append(f'room "{room}"')

    # Words that are part of a recognised resident's name are that resident, not a caregiver or wing: a resident
    # "Allison Akilah" must not read as staff member "… Allison" (the deterministic planner already skips them).
    resident_names = [o.lower() for o in mask.originals.values()]
    # where the masked resident names sit in the question: a mention inside one of these spans IS the resident, however
    # it is inflected ("Merry Johnson's family" must not read as staff member "… Johnson")
    resident_spans = [
        m.span() for o in mask.originals.values() if o.strip() for m in re.finditer(re.escape(o.strip()), question, re.I)
    ]

    def in_resident_name(text: str) -> bool:
        base = re.sub(r"(?:'s|’s|'|’)$", "", text.strip(), flags=re.I)  # "Johnson's" → "Johnson"
        if not base:
            return False
        hits = [m.span() for m in re.finditer(r"(?<![A-Za-z])" + re.escape(base) + r"(?![A-Za-z])", question, re.I)]
        if hits and all(any(s <= a and b <= e for s, e in resident_spans) for a, b in hits):
            return True
        pattern = r"(?<![a-z])" + re.escape(base.lower()) + r"(?![a-z])"
        return any(re.search(pattern, name) for name in resident_names) or in_family_span(base)

    # names the masking classified as family members (family_N / person_N): never demanded as a caregiver or a
    # facility here; how the plan uses them is checked by the resolver (family_member, or person_N as a caregiver)
    entity_spans = list(getattr(mask, "entity_spans", []) or [])

    def in_family_span(text: str) -> bool:
        if not entity_spans:
            return False
        hits = [m.span() for m in re.finditer(r"(?<![A-Za-z])" + re.escape(text) + r"(?![A-Za-z])", question, re.I)]
        return bool(hits) and all(any(s <= a and b <= e for s, e in entity_spans) for a, b in hits)

    staff_used = any(getattr(f, n) is not None for n in STAFF_RELATION_FIELDS) or bool(
        set(plan.output_fields) & set(STAFF_RELATION_ATTRIBUTES)
    )
    if "primary_caregiver" not in groups and "primary_caregiver" not in attrs:
        used = set((f.primary_caregiver.include or []) + (f.primary_caregiver.exclude or [])) if f.primary_caregiver else set()
        for text, values in index.mentioned_caregivers(question).items():
            covered = _text_covers(f.primary_caregiver, FIELD_MAP["primary_caregiver"], list(values))
            if not used & set(values) and not covered and not in_resident_name(text) and not (
                staff_used and _is_staff_word(text, index)  # "case manager" names the role the plan uses, not a person
            ):
                g.missing.append(f'caregiver "{text}"')
    if "facility" not in groups and not attrs & {"facility", "room_number"}:
        used = set((f.facility.include or []) + (f.facility.exclude or [])) if f.facility else set()
        for text, values in index.mentioned_wings(question).items():
            covered = _text_covers(f.facility, FIELD_MAP["facility"], list(values))
            if not used & set(values) and not covered and not in_resident_name(text):
                g.missing.append(f'facility "{text}"')

    # ---------------------------------------------------------------- text matches ("starts with A")
    _check_text_matches(f, question, prev, g)
    # ---------------------------------------------------------------- family relation (familymembers)
    _check_family(plan, question, prev, g, mask.text)
    # ---------------------------------------------------------------- staff relation (assigned staff)
    _check_staff(plan, question, index, prev, g, mask)

    # ---------------------------------------------------------------- age
    # an age range needs an age expression the grammar parsed ("over 80", "between 70 and 80"); any other number in
    # the question ("the 5 oldest", "room 101") is not evidence for one. A number that counts family members
    # ("at least 3 family members") is not an age either.
    numbers = [n for n in scan.numbers if not _FAMILY_COUNT_AFTER.match(low, n.end)]
    if f.age and f.age.has_range and not numbers and not _AGE_WORDS.search(low) and not prev.get("age"):
        g.invented.append("age range")
    elif f.age and f.age.has_range and len(numbers) == 1:
        b = numbers[0].bounds
        if (f.age.min, f.age.max) != (b.get("min"), b.get("max")):
            f.age = AgeRange(**b, missing=f.age.missing)
            g.fixes.append(f"age bounds set from '{question[numbers[0].start : numbers[0].end]}'")
    elif not (f.age and f.age.has_range) and len(numbers) == 1 and "age" not in groups | attrs and not (plan.sort and plan.sort.field == "age"):
        if not plan.limit:  # "the 3 oldest" is a limit, not an age range
            g.missing.append(f"age \"{question[numbers[0].start : numbers[0].end]}\"")

    # ---------------------------------------------------------------- lookups
    if plan.lookup:
        allowed = {i.upper() for ids in mask.placeholders.values() for i in ids}
        allowed |= {f"RES-{m}" for m in re.findall(r"\bres[-_ ]?(\d{3,6})\b", low)}
        for rid in plan.lookup.residents:
            display = index.display_by_id.get(rid.upper(), "")
            # a reference the question never made: a placeholder that was not masked ("resident_1" with no resident
            # named) or a resident whose name is not in the question
            if rid.upper() not in allowed and not (display and all(fuzzy_in_question(p) for p in _name_parts(display) if p)):
                g.invented.append(f'resident "{display or rid}"')

    # ---------------------------------------------------------------- dates
    # only real ranges: a missing-only condition ("no admission date") has no bounds to check
    date_fields = [
        n for n in ("admission_date", "created_date", "discharge_date") if getattr(f, n) is not None and getattr(f, n).has_range
    ]
    found = scan.dates
    if date_fields:
        # Date words that belong to a comparison with another resident ("the same month as resident_1", "after her")
        # carry no date of their own: they never make a literal date in the plan acceptable.
        compared_dates = structure is not None and any(c.field in _DATE_FIELDS for c in structure.comparisons)
        unresolved = scan.unresolved and not compared_dates
        if not found and not unresolved:  # a bare number ("the 5 most recent") is never a date
            g.invented.extend(n.replace("_", " ") for n in date_fields if not prev.get(n))
        elif len(found) == 1 and len(date_fields) == 1:
            r = found[0].range
            name = date_fields[0]
            want = DateRange(start=r.get("start"), end=r.get("end"), missing=getattr(f, date_fields[0]).missing)
            if (getattr(f, name).start, getattr(f, name).end) != (want.start, want.end):
                setattr(f, name, want)
                g.fixes.append(f"{name} bounds set from '{scan.texts[found[0].start]}'")
            if len(date_cues) == 1 and date_cues[0] != name and not prev.get(name):
                setattr(f, date_cues[0], getattr(f, name))
                setattr(f, name, None)
                g.fixes.append(f"date field {name} → {date_cues[0]}")
        elif found:
            wanted = {(d.range.get("start"), d.range.get("end")) for d in found}
            for name in date_fields:
                r = getattr(f, name)
                got = (r.start.isoformat() if r.start else None, r.end.isoformat() if r.end else None)
                if got not in wanted and not prev.get(name):
                    g.invented.append(f"{name.replace('_', ' ')} range {got[0]}..{got[1]}")
    elif found and not attrs & {"admission_date", "discharge_date", "created_date"}:
        if not (plan.sort and plan.sort.field.endswith("_date")):
            g.missing.append(f'date "{scan.texts[found[0].start]}"')

    # ---------------------------------------------------------------- missing values ("no caregiver assigned")
    if missing_detector is not None:
        _check_missing_values(f, question, low, mask, prev, missing_detector, g)

    # ---------------------------------------------------------------- projection (what to show), independently
    _check_projection(plan, question, low, mask, index, scan, missing_detector, projection_detector, g)

    repeated_groups = _repeated_count_groups(question, mask, index, scan, date_anchors)
    _check_structure(plan, low, scan, g, set(repeated_groups))
    if repeated_groups:
        requested_fields = set(repeated_groups)
        if len(requested_fields) > 2:
            g.blocking.append("separate counts across more than two fields are not supported")
        else:
            if plan.action != "breakdown":
                g.missing.append("a breakdown with a separate count for each requested value")
            actual_fields = set(plan.group_by or [])
            for name in sorted(requested_fields - actual_fields):
                g.missing.append(f"grouping by {name.replace('_', ' ')}")
            for name in sorted(actual_fields - requested_fields):
                g.invented.append(f"grouping by {name.replace('_', ' ')}")
            for name, requested_values in repeated_groups.items():
                condition = getattr(f, name)
                included = set(condition.include or []) if condition else set()
                for value in sorted(set(requested_values) - included):
                    g.missing.append(f"including {name.replace('_', ' ')} {value} in the breakdown")
                for value in sorted(included - set(requested_values)):
                    g.invented.append(f"{name.replace('_', ' ')} {value} outside the requested groups")
    g.plan = plan
    return g


_DATE_FIELDS = ("admission_date", "created_date", "discharge_date")
_CLAUSE_BREAK = re.compile(r"[,.;?!]|\b(?:who|which|that|whose|where|but|while|whereas)\b")


def _value_mentions(field_name: str, value: str, low: str, question: str, index: EntityIndex) -> list[tuple[int, int]]:
    """Where the question names `value` of `field_name` (character spans in `low`)."""
    if field_name in SYNONYMS:
        forms = SYNONYMS[field_name].get(value, (value.replace("_", " "),))
    elif field_name == "facility":
        forms = tuple(t.lower() for t, vals in index.mentioned_wings(question).items() if value in vals)
    elif field_name == "primary_caregiver":
        forms = tuple(t.lower() for t, vals in index.mentioned_caregivers(question).items() if value in vals)
    else:
        forms = (value.lower(),)
    spans = []
    for form in forms:
        spans += [m.span() for m in re.finditer(r"(?<![a-z0-9])" + re.escape(form) + r"(?![a-z0-9])", low)]
    return spans


def _negation_scopes(low: str, start: int) -> bool:
    """A negation word occurs before `start` in the same clause."""
    clause_start = max((m.end() for m in _CLAUSE_BREAK.finditer(low, 0, start)), default=0)
    return bool(_NEGATION.search(low[clause_start:start]))


def _check_structure_semantics(plan: QueryPlan, low: str, mask: MaskResult, structure, g: "Grounding") -> None:
    """The plan must represent the question's structure, not merely contain valid conditions:
      • a comparison with another resident (or among residents) cannot be expressed yet → blocking (Stage 1);
      • only SUBJECT residents may be looked up — a reference is compared with, never the answer;
      • the operation must be the one asked for (how many → count; which/list/find → list).
    """
    if structure.references:
        g.blocking.append("a comparison with another resident (not supported yet)")
    elif structure.cross_record:
        g.blocking.append("a comparison of residents with one another (not supported yet)")
    if plan.lookup:
        id_to_keys: dict[str, set[str]] = {}
        for key, ids in mask.placeholders.items():
            for rid in ids:
                id_to_keys.setdefault(rid.upper(), set()).add(key)
        for rid in plan.lookup.residents:
            keys = id_to_keys.get(rid.upper(), set())
            if keys & structure.reference_keys:
                g.invented.append(f"a lookup of {sorted(keys)[0]}, who is compared with, not asked about")
        # residents identified by an id typed in the question are subjects too ("show RES-1001's details")
        typed = {f"RES-{m}" for m in re.findall(r"\bres[-_ ]?(\d{3,6})\b", low)} | {m.upper() for m in re.findall(r"\b[0-9a-f]{24}\b", low)}
        typed_only = bool(typed) and {r.upper() for r in plan.lookup.residents} <= typed
        if not structure.subjects and not typed_only and (structure.operations or structure.references):
            g.missing.append("the residents the question asks for (it asks about many residents, not one)")
    ops = set(structure.operations)
    if not ops & {"breakdown", "aggregate"}:
        if plan.action == "count" and ops == {"list"}:
            g.missing.append("operation: list the residents (the question asks which ones, not how many)")
        elif plan.action == "list" and ops == {"count"}:
            g.missing.append("operation: count the residents (the question asks how many)")


_ALL_FIELDS = (
    "status", "care_type", "facility", "gender", "primary_caregiver", "room_number",
    "age", "admission_date", "created_date", "discharge_date",
)  # fmt: skip


def _check_missing_values(f, question: str, low: str, mask, prev: dict, detector, g: "Grounding") -> None:
    """Missing-value conditions must match the question: none dropped, none invented, no OR across fields."""
    blocked = [False] * len(low)
    for original in mask.originals.values():  # words inside a resident's name are never a field word
        for m in re.finditer(re.escape(original.lower()), low):
            for i in range(*m.span()):
                blocked[i] = True
    found = detector.scan(question, blocked)
    if found.or_across_fields:  # the plan format cannot express it: no retry can help
        g.blocking.append("an OR between different fields (e.g. 'no caregiver or no room'); ask them separately")
    wanted = found.by_field
    for name, flag in wanted.items():
        cond = getattr(f, name)
        if cond is None or cond.missing is not flag:
            g.missing.append(f"{name.replace('_', ' ')} {'not recorded' if flag else 'recorded'}")
    for name in _ALL_FIELDS:
        cond = getattr(f, name)
        if cond is None or cond.missing is None or name in wanted:
            continue
        if (prev.get(name) or {}).get("missing") is cond.missing:  # kept from the previous question
            continue
        g.invented.append(f"'{name.replace('_', ' ')} {'not recorded' if cond.missing else 'recorded'}' condition")


def _check_projection(plan, question, low, mask, index, scan, missing_detector, detector, g: "Grounding") -> None:
    """Output fields are validated on their own, separately from the conditions:
      • every field must be one the schema can return (blocking: no model output can change the schema);
      • fields that are single-record-only (emergency contacts) are never listed for several residents (blocking);
      • for resident rows: every field the question asks to see is present (omitted → correctable) and no field is
        shown that the question did not ask for (invented → correctable)."""
    supported = output_fields()
    for attr in plan.output_fields:
        if attr not in supported:
            g.blocking.append(f"'{attr.replace('_', ' ')}' is not a field the resident records can show")
    if plan.action in ("list", "breakdown"):
        for attr in plan.output_fields:
            if supported.get(attr) in single_record_only():
                g.blocking.append(f"{attr.replace('_', ' ')} is only shown for one resident at a time")
    if detector is None or plan.action not in ("list", "breakdown", "lookup"):
        return
    not_attribute = [False] * len(low)
    for original in mask.originals.values():
        for m in re.finditer(re.escape(original.lower()), low):
            for i in range(*m.span()):
                not_attribute[i] = True
    if missing_detector is not None:
        for mention in missing_detector.scan(question).mentions:
            for i in range(*mention.span):
                not_attribute[i] = True
    starts = value_starts(question, index, dates=scan.dates, numbers=scan.numbers)
    requested = {p.attribute for p in detector.scan(low, not_attribute, starts) if p.attribute is not None}
    if plan.action == "lookup":
        masked_low = mask.text.lower()
        requested |= {
            attribute
            for attribute, pattern in detector.phrases
            if attribute is not None and pattern.search(masked_low)
        }
    shown = set(plan.output_fields)
    cfg = staff_config()
    if cfg is not None and "care_team" in shown and cfg.specific_roles_in(
        _without_care_types(question), index.staff_name_tokens(), include_generic=True
    ):
        # "resident_1's physician / nurse": the care team, narrowed to that role, answers it
        requested -= {"primary_caregiver"}
        requested |= {"care_team"}
    rel = family_relation()
    if rel is not None and rel.cue_in(question, mask.text):  # "Eleanor's son's phone": family details answer it
        shown -= set(FAMILY_ATTRIBUTES) - requested
    conditions = {
        field_name
        for field_name in type(plan.filters).model_fields
        if getattr(plan.filters, field_name) is not None
    } if plan.action == "lookup" else set()
    if plan.action == "lookup" and any(getattr(plan.filters, n) is not None for n in FAMILY_SELECTORS):
        conditions |= {"family_members"}  # "is X a family member of R?": the words name the condition, not a display
    if plan.action == "lookup" and (plan.filters.staff_role is not None or plan.filters.staff_designation is not None):
        conditions |= {"care_team"}  # "does resident_1 have a case manager?": a condition on the care team
    for attr in sorted(requested - (shown | conditions)):
        g.missing.append(f"show {attr.replace('_', ' ')}")
    for attr in sorted(shown - requested - conditions):
        g.invented.append(f"show {attr.replace('_', ' ')}")


def _repeated_count_groups(
    question: str, mask: MaskResult, index: EntityIndex, scan: DateScan, date_anchors: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Recognize separate count clauses over multiple named values of the same groupable field."""
    low = mask.text.lower()
    count_mentions = sum(
        len(list(re.finditer(r"(?<![a-z])" + re.escape(cue) + r"(?![a-z])", low)))
        for cue in COUNT_CUES
    )
    if count_mentions < 2:
        return {}

    values: dict[str, set[str]] = {}
    status_text = low
    if scan.dates and "admission_date" in date_fields_mentioned(question, date_anchors):
        status_text = re.sub(r"\badmitted\b", " ", status_text)
    for field_name in ("status", "care_type", "gender"):
        available = {
            "status": set(index.statuses),
            "care_type": set(index.care_types),
            "gender": {"Female", "Male", "Other"},
        }[field_name]
        for value, forms in NAMED[field_name].items():
            text = status_text if field_name == "status" else low
            if value in available and any(_has(form, text) for form in forms):
                values.setdefault(field_name, set()).add(value)

    def is_resident_name(text: str) -> bool:
        normal = text.lower().strip()
        return any(normal in original.lower() for original in mask.originals.values())

    for field_name, mentions in (
        ("facility", index.mentioned_wings(question)),
        ("primary_caregiver", index.mentioned_caregivers(question)),
    ):
        for text, found in mentions.items():
            if not is_resident_name(text):
                values.setdefault(field_name, set()).update(found)

    return {field_name: sorted(found) for field_name, found in values.items() if len(found) > 1}