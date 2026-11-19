"""Generic, deterministic parsing of date and numeric-range expressions in a question.

This module knows NOTHING about the resident database: no field names, no values, no collections. It only
understands the English grammar of dates ("last 3 months", "in October", "since 2026-01-01") and numeric
comparisons ("over 85", "between 60 and 70", "aged 70"). The caller (DirectQueryBuilder) decides which schema
field each parsed range belongs to, using the anchor words configured in intent_routes.yaml.

Every match carries the character span it covers, so the caller can mark those words explained and never treat a
consumed number (e.g. the "2026" in "October 2026") as an unexplained token.

All relative expressions are resolved against a single reference date (the facility "today", already in the
facility timezone), so results are reproducible and timezone-consistent.
"""

import calendar
import re
from dataclasses import dataclass
from datetime import date

# Generic language constants (not database values): month names and the units of relative time.
_MONTHS: dict[str, int] = {}
for _i in range(1, 13):
    _MONTHS[calendar.month_name[_i].lower()] = _i
    _MONTHS[calendar.month_abbr[_i].lower()] = _i
_MONTHS["sept"] = 9  # common abbreviation calendar does not list

_UNIT_DAYS = {"day": 1, "week": 7}  # months and years are handled with calendar arithmetic
_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

# A single date POINT, in the forms people write: an ISO date, a day date in either order ("august 1, 2026",
# "aug 1st", "1 august 2026", "1st of august"), a month with an optional year, or a bare year.
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ORD = r"(?:st|nd|rd|th)?"
_POINT = (
    r"(?:(?P<iso>\d{4}-\d{2}-\d{2})"
    rf"|(?P<m1>{_MONTH_ALT})\.?\s+(?P<d1>\d{{1,2}}){_ORD}(?![\d:])(?:,?\s+(?P<y1>\d{{4}}))?"
    rf"|(?P<d2>\d{{1,2}}){_ORD}\s+(?:of\s+)?(?P<m2>{_MONTH_ALT})\b\.?(?:,?\s+(?P<y2>\d{{4}}))?"
    rf"|(?P<m3>{_MONTH_ALT})\b(?:\s+(?:of\s+)?(?P<y3>this year|last year|previous year|\d{{4}}))?"
    r"|(?P<y4>\d{4})(?![\d-]))"
)
_POINT_RE = re.compile(r"\b" + _POINT + r"\b")
# "between <point> and <point>" / "from <point> to <point>": the opening word, then the joiner between the two points
_RANGE_OPEN_RE = re.compile(r"\b(?:between|from)\s+(?:the\s+)?")
_RANGE_JOIN_RE = re.compile(r"\s*,?\s*(?:and|to|through|until|till|-)\s+(?:the\s+)?")
# Comparison words before a point. "before"/"prior to" exclude the point itself and "after" starts once it is over;
# "since", "from" and "until"/"up to"/"by"/"through" include it.
_EXCLUSIVE_END = ("before", "prior to")
_INCLUSIVE_END = ("until", "up to", "by", "through", "till", "no later than")
_EXCLUSIVE_START = ("after", "later than")
_INCLUSIVE_START = ("since", "from", "starting", "on or after")
_COMPARISON_RE = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, _EXCLUSIVE_END + _INCLUSIVE_END + _EXCLUSIVE_START + _INCLUSIVE_START),
                              key=len, reverse=True)) + r")\s+(?:the\s+)?" + _POINT + r"\b"
)  # fmt: skip


def skeleton(text: str, today: date) -> str:
    """`text` with its date and numeric-range expressions blanked out.

    Used only to classify the KIND of question (count vs list vs breakdown ...) with the semantic router: date and
    age phrases ("in the last 3 months", "between 60 and 70") carry no intent signal but pull the embedding away
    from the route examples, which would drop the router's confidence below the direct-answer threshold. The actual
    ranges are still parsed from the original text when the plan is built.
    """
    spans = [(d.start, d.end) for d in parse_dates(text, today)] + [(n.start, n.end) for n in parse_number_ranges(text)]
    if not spans:
        return text
    chars = list(text)
    for start, end in spans:
        for i in range(start, end):
            chars[i] = " "
    return re.sub(r"\s+", " ", "".join(chars)).strip()


@dataclass
class DateMatch:
    start: int
    end: int
    range: dict  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} (either key may be omitted for open ranges)


@dataclass
class NumberMatch:
    start: int
    end: int
    bounds: dict  # {"min": int} and/or {"max": int}


# ----------------------------------------------------------------------------- date helpers
def _last_day(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _shift_months(d: date, delta: int) -> date:
    """Move a date by whole months, clamping the day to the target month's length."""
    index = (d.year * 12 + (d.month - 1)) + delta
    year, month = divmod(index, 12)
    month += 1
    return date(year, month, min(d.day, _last_day(year, month)))


def _whole_month(year: int, month: int) -> dict:
    return {"start": date(year, month, 1).isoformat(), "end": date(year, month, _last_day(year, month)).isoformat()}


def _whole_year(year: int) -> dict:
    return {"start": date(year, 1, 1).isoformat(), "end": date(year, 12, 31).isoformat()}


def _iso(d: date) -> str:
    return d.isoformat()


# ----------------------------------------------------------------------------- date parsing
def parse_dates(text: str, today: date) -> list[DateMatch]:
    """Every date expression in `text`, resolved against `today`. Non-overlapping, left to right."""
    low = text.lower()
    matches: list[DateMatch] = []
    taken: list[tuple[int, int]] = []

    def free(start: int, end: int) -> bool:
        return not any(start < te and ts < end for ts, te in taken)

    def add(start: int, end: int, rng: dict | None) -> None:
        if rng and free(start, end):
            matches.append(DateMatch(start, end, rng))
            taken.append((start, end))

    # between D1 and D2 / from D1 to D2 — ONE inclusive range, for every point form (ISO, "january 1, 2024",
    # "1 march 2025", "march 2024", "2024"): from the first day of D1 to the last day of D2. Must run before the
    # single-point rules: read separately, "between january 1, 2024, and december 31, 2025" became two single days,
    # so a correct 2024-01-01..2025-12-31 plan was rejected as "not asked" (benchmark Q36, 2026-11-17).
    for m in _RANGE_OPEN_RE.finditer(low):
        a = _POINT_RE.match(low, m.end())
        joiner = _RANGE_JOIN_RE.match(low, a.end()) if a else None
        b = _POINT_RE.match(low, joiner.end()) if joiner else None
        pa, pb = (_point(a, today), _point(b, today)) if b else (None, None)
        if pa and pb and pa[0] <= pb[1]:
            add(m.start(), b.end(), {"start": _iso(pa[0]), "end": _iso(pb[1])})

    # before / after / since / until / by  <date point>. The point is a whole period ("2026", "june 2026") or one
    # day; "before" ends the day before the period starts, "after" starts the day after it ends.
    for m in _COMPARISON_RE.finditer(low):
        word = " ".join(m.group(1).split())
        point = _point(m, today)
        if not point:
            continue
        lo, hi = point
        if word in _EXCLUSIVE_END:
            add(m.start(), m.end(), {"end": _iso(date.fromordinal(lo.toordinal() - 1))})
        elif word in _INCLUSIVE_END:
            add(m.start(), m.end(), {"end": _iso(hi)})
        elif word in _EXCLUSIVE_START:
            add(m.start(), m.end(), {"start": _iso(date.fromordinal(hi.toordinal() + 1))})
        else:
            add(m.start(), m.end(), {"start": _iso(lo)})

    # a single day written out ("on august 1, 2026", "1st of june") is that one day
    for m in _POINT_RE.finditer(low):
        if m.group("d1") or m.group("d2"):
            point = _point(m, today)
            if point:
                add(m.start(), m.end(), {"start": _iso(point[0]), "end": _iso(point[1])})

    # last / past / previous N days|weeks|months|years  (trailing window; requires an explicit number)
    for m in re.finditer(r"\b(?:last|past|previous|prior|recent)\s+(\d+)\s+(day|week|month|year)s?\b", low):
        n, unit = int(m.group(1)), m.group(2)
        if unit in _UNIT_DAYS:
            start = date.fromordinal(today.toordinal() - n * _UNIT_DAYS[unit])
        elif unit == "month":
            start = _shift_months(today, -n)
        else:  # year
            start = _shift_months(today, -12 * n)
        add(m.start(), m.end(), {"start": _iso(start), "end": _iso(today)})

    # a named month, optionally with a year or "this/last year" (greedy, so "october this year" is one match)
    month_re = r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\b(?:\s+(?:of\s+)?(this year|last year|previous year|\d{4}))?"
    for m in re.finditer(month_re, low):
        month = _MONTHS[m.group(1)]
        year = _resolve_year(m.group(2), month, today)
        add(m.start(), m.end(), _whole_month(year, month))

    # this month / this year  (current whole period)
    for m in re.finditer(r"\b(?:this|current)\s+(month|year)\b", low):
        add(m.start(), m.end(), _whole_month(today.year, today.month) if m.group(1) == "month" else _whole_year(today.year))

    # last month / previous year  (whole previous calendar period, no explicit number)
    for m in re.finditer(r"\b(?:last|previous|prior)\s+(month|year)\b", low):
        if m.group(1) == "month":
            prev = _shift_months(today.replace(day=1), -1)
            add(m.start(), m.end(), _whole_month(prev.year, prev.month))
        else:
            add(m.start(), m.end(), _whole_year(today.year - 1))

    # trailing single period without a number: "past month", "last week", "past day"
    for m in re.finditer(r"\b(?:last|past|previous|prior|recent)\s+(day|week|month|year)s?\b", low):
        unit = m.group(1)
        if unit in _UNIT_DAYS:
            start = date.fromordinal(today.toordinal() - _UNIT_DAYS[unit])
        elif unit == "month":
            start = _shift_months(today, -1)
        else:
            start = _shift_months(today, -12)
        add(m.start(), m.end(), {"start": _iso(start), "end": _iso(today)})

    # today / yesterday
    for m in re.finditer(r"\b(today|yesterday)\b", low):
        d = today if m.group(1) == "today" else date.fromordinal(today.toordinal() - 1)
        add(m.start(), m.end(), {"start": _iso(d), "end": _iso(d)})

    for m in re.finditer(r"\bthis\s+year\b", low):
        add(m.start(), m.end(), _whole_year(today.year))
    for m in _ISO_RE.finditer(low):
        add(m.start(), m.end(), {"start": m.group(0), "end": m.group(0)})
    for m in re.finditer(r"\b(\d{4})\b", low):
        year = int(m.group(1))
        if 1900 <= year <= today.year + 1:
            add(m.start(), m.end(), _whole_year(year))

    matches.sort(key=lambda d: d.start)
    return matches


def _resolve_year(suffix: str | None, month: int, today: date) -> int:
    if suffix in (None, ""):
        # most recent occurrence not in the future: this year if the month has started, else last year
        return today.year if month <= today.month else today.year - 1
    if suffix == "this year":
        return today.year
    if suffix in ("last year", "previous year"):
        return today.year - 1
    return int(suffix)


def _point(m: re.Match, today: date) -> tuple[date, date] | None:
    """The (first day, last day) of the date point matched by _POINT: one day for an ISO or day date, the whole
    month or year otherwise. None when the numbers are not a real date."""
    g = m.groupdict()
    try:
        if g["iso"]:
            d = date.fromisoformat(g["iso"])
            return d, d
        for mk, dk, yk in (("m1", "d1", "y1"), ("m2", "d2", "y2")):
            if g[mk] and g[dk]:
                month = _MONTHS[g[mk]]
                year = int(g[yk]) if g[yk] else _resolve_year(None, month, today)
                d = date(year, month, int(g[dk]))
                return d, d
        if g["m3"]:
            month = _MONTHS[g["m3"]]
            rng = _whole_month(_resolve_year(g["y3"], month, today), month)
            return date.fromisoformat(rng["start"]), date.fromisoformat(rng["end"])
        if g["y4"] and 1900 <= int(g["y4"]) <= today.year + 1:
            return date(int(g["y4"]), 1, 1), date(int(g["y4"]), 12, 31)
    except ValueError:  # "february 30"
        return None
    return None


# ----------------------------------------------------------------------------- numeric comparisons
# Operator phrases → how they bound a number. "strict" means the boundary value itself is excluded.
_MIN_STRICT = ("over", "above", "more than", "greater than", "older than", "exceeding")
_MIN_INCL = ("at least", "minimum", "min", "no fewer than", "no less than")
_MAX_STRICT = ("under", "below", "less than", "fewer than", "younger than")
_MAX_INCL = ("at most", "maximum", "max", "up to", "no more than")


def parse_number_ranges(text: str) -> list[NumberMatch]:
    """Numeric comparison expressions, as bounds. The caller maps these to a numeric field (e.g. age)."""
    low = text.lower()
    matches: list[NumberMatch] = []
    taken: list[tuple[int, int]] = []

    def add(start: int, end: int, bounds: dict) -> None:
        if any(start < te and ts < end for ts, te in taken):
            return
        matches.append(NumberMatch(start, end, bounds))
        taken.append((start, end))

    # between N and M  (inclusive range)
    for m in re.finditer(r"\bbetween\s+(\d{1,3})\s+and\s+(\d{1,3})\b", low):
        a, b = int(m.group(1)), int(m.group(2))
        add(m.start(), m.end(), {"min": min(a, b), "max": max(a, b)})

    # N to M  (inclusive range, e.g. "aged 60 to 70")
    for m in re.finditer(r"\b(\d{1,3})\s+to\s+(\d{1,3})\b", low):
        a, b = int(m.group(1)), int(m.group(2))
        add(m.start(), m.end(), {"min": min(a, b), "max": max(a, b)})

    def phrase_group(words: tuple[str, ...]) -> str:
        return "|".join(sorted((re.escape(w) for w in words), key=len, reverse=True))

    for words, key, delta in (
        (_MIN_STRICT, "min", 1),
        (_MIN_INCL, "min", 0),
        (_MAX_STRICT, "max", -1),
        (_MAX_INCL, "max", 0),
    ):
        for m in re.finditer(r"\b(?:" + phrase_group(words) + r")\s+(\d{1,3})\b", low):
            add(m.start(), m.end(), {key: int(m.group(1)) + delta})

    # "N or older" / "N and above" / "N+"  → inclusive minimum
    for m in re.finditer(r"\b(\d{1,3})\s*(?:\+|or (?:older|above|more)|and (?:above|older|up))\b", low):
        add(m.start(), m.end(), {"min": int(m.group(1))})
    # "N or younger" / "N and below"  → inclusive maximum
    for m in re.finditer(r"\b(\d{1,3})\s+(?:or (?:younger|below|less)|and below)\b", low):
        add(m.start(), m.end(), {"max": int(m.group(1))})

    # an exact value: "aged 70", "age of 70", "70 years old", "70 yo"  → min == max
    for m in re.finditer(r"\b(?:aged?|age\s+of)\s+(\d{1,3})\b", low):
        add(m.start(), m.end(), {"min": int(m.group(1)), "max": int(m.group(1))})
    for m in re.finditer(r"\b(\d{1,3})\s*(?:years?\s+old|yrs?\s+old|yo)\b", low):
        add(m.start(), m.end(), {"min": int(m.group(1)), "max": int(m.group(1))})

    matches.sort(key=lambda n: n.start)
    return matches
