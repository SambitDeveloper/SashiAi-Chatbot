"""The ONE definition of text matching (`starts_with` / `contains` on TEXT_MATCH_FIELDS).

Both filter engines use these functions: `repository.matches` (in-memory) and `resident_view.to_mongo_filter` (MongoDB;
the MCP repository runs the same pipeline), so the two can never disagree.

Semantics, per field:
  • The needle is LITERAL text: it is regex-escaped here and never interpreted as a pattern (no wildcards, no regex).
  • Case-insensitive (Python re.IGNORECASE / MongoDB $options "i": Unicode simple case folding in both).
  • starts_with: the value begins with the needle · contains: the needle occurs anywhere in the value.
  • Needles in one list are OR-ed; starts_with and contains are AND-ed with each other and with the rest of the filter.
  • Only string values can match. null, absent, "", a declared "not recorded" placeholder (resident_schema.yaml
    `missing_values`, e.g. wing "Unknown") and non-string values (numbers, objects) never match.
  • A list field (several caregivers) matches when ANY real element matches, like MongoDB $regex on an array.
"""

import re
from typing import Any

from app.data.schema_config import list_fields, missing_markers


def regex_source(op: str, needle: str) -> str:
    """Regex source for one needle: the escaped literal, anchored at the start for starts_with."""
    if op not in ("starts_with", "contains"):
        raise ValueError(f"unknown text operator {op!r}")
    escaped = re.escape(needle)
    return f"^{escaped}" if op == "starts_with" else escaped


def _compiled(op: str, needles: list[str]) -> list[re.Pattern]:
    return [re.compile(regex_source(op, n), re.IGNORECASE) for n in needles]


def _real_strings(value: Any, view_field: str) -> list[str]:
    markers = missing_markers(view_field)
    if view_field in list_fields() and not isinstance(value, list):
        return []  # mirrors $elemMatch, which never matches a non-array value
    values = value if isinstance(value, list) else [value]
    return [v for v in values if isinstance(v, str) and not any(v == m for m in markers)]


def value_matches(value: Any, view_field: str, starts_with: list[str] | None, contains: list[str] | None) -> bool:
    """In-memory: does one record's field value satisfy the text conditions? (True when there are none.)"""
    strings = _real_strings(value, view_field)
    for op, needles in (("starts_with", starts_with), ("contains", contains)):
        if needles and not any(p.search(s) for p in _compiled(op, needles) for s in strings):
            return False
    return True


def mongo_clauses(view_field: str, starts_with: list[str] | None, contains: list[str] | None) -> list[dict[str, Any]]:
    """MongoDB: one clause per needle list (AND-ed by the caller); needles within a list are OR-ed."""
    markers = missing_markers(view_field)
    out: list[dict[str, Any]] = []
    for op, needles in (("starts_with", starts_with), ("contains", contains)):
        if not needles:
            continue
        alternatives = []
        for needle in needles:
            cond = {"$regex": regex_source(op, needle), "$options": "i", "$nin": markers}
            # a list field: the SAME element must match and be a real value
            alternatives.append({view_field: {"$elemMatch": cond}} if view_field in list_fields() else {view_field: cond})
        out.append(alternatives[0] if len(alternatives) == 1 else {"$or": alternatives})
    return out
