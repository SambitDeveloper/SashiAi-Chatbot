"""The ONE definition of related-collection semantics (resident_schema.yaml `relations`). Today: `family`.

A relation contributes resident-level fields derived from ALL of a resident's related records, so every query still
returns exactly one row per resident (counts, paging and grouping keep their meaning). This module builds:
  • the MongoDB stages (`lookup_stage`, `view_fields`) used by resident_view.ResidentView, and
  • the identical in-memory derivation (`derive`) used by the test repository,
from the same configuration, so the two execution paths cannot disagree.

Family semantics, per resident:
  • members        = familymembers whose residentId is the resident's _id, excluding relations listed in
                     `exclude_relations` (the resident recorded as their own contact), sorted by _id, at most
                     `max_per_resident`.
  • relation       = lower-cased, trimmed text, then `relation_aliases` ("Granddaugter" → "granddaughter"). Empty or
                     non-text → not recorded (still a member, no relation value).
  • familyCount    = number of members (0 = none recorded).
  • familyRelations / familyTypes = the distinct non-empty relations / types of the members (a resident "has a
                     daughter" when ANY member is a daughter).
  • familyFlags    = flag names whose source field is exactly `true` on ANY member (portal_access, ...).
  • familyMembers  = "Name (Relation as recorded)" per member, for display in lists.
  • familyContacts = name, relation, type, phone, email per member: single-resident answers only.
Limitation: conditions on different fields are tested independently (has a daughter AND some member has portal
access), not on the same family record.
"""

import re
from functools import cached_property, lru_cache
from typing import Any

from app.data.schema_config import schema_config

FAMILY_VIEW_FIELDS = ("familyCount", "familyRelations", "familyTypes", "familyFlags", "familyMembers", "familyContacts",
                      "familyIds")  # fmt: skip
FAMILY_PLAN_FIELDS = ("family_relation", "family_type", "family_flags", "family_count", "family_member")
FAMILY_ATTRIBUTES = ("family_members", "family_count", "family_relationship")
# family filters that, on a ONE-resident lookup, select which family members are shown (not a yes/no check)
FAMILY_SELECTORS = ("family_member", "family_relation", "family_type")
FAMILY_TYPES = ("family", "emergency")


def _text(expr: str) -> dict:
    return {"$convert": {"input": expr, "to": "string", "onError": "", "onNull": ""}}


def _py_text(value: Any) -> str:
    return value if isinstance(value, str) else ""  # mirrors _text for the values that occur (str / missing / null)


class FamilyRelation:
    def __init__(self, config: dict):
        self.collection: str = config["collection"]
        self.local_field: str = config.get("local_field", "_id")
        self.foreign_field: str = config.get("foreign_field", "residentId")
        self.cap: int = int(config.get("max_per_resident", 20))
        self.exclude: list[str] = [str(x).strip().lower() for x in config.get("exclude_relations") or []]
        self.aliases: dict[str, str] = {str(k).strip().lower(): str(v) for k, v in (config.get("relation_aliases") or {}).items()}
        self.terms: dict[str, list[str]] = {str(k).lower(): [str(x) for x in v] for k, v in (config.get("relation_terms") or {}).items()}
        self.flags: dict[str, str] = dict(config.get("flags") or {})
        self._cues = self._pattern(config.get("cue_words"))
        self._weak_cues = self._pattern(config.get("weak_cue_words"))

    @staticmethod
    def _pattern(words) -> re.Pattern | None:
        words = sorted({str(w).lower() for w in words or []}, key=len, reverse=True)
        return re.compile(r"(?<![a-z])(" + "|".join(re.escape(w) for w in words) + r")(?![a-z])", re.I) if words else None

    def is_family_word(self, word: str) -> bool:
        """A family vocabulary word ("family", "daughter", "kin"): part of a question's wording, never of a name."""
        w = word.strip().lower()
        return any(p is not None and p.fullmatch(w) for p in (self._cues, self._weak_cues))

    def cue_in(self, question: str, masked: str | None = None) -> str | None:
        """The first word of the question that is about family ("daughter", "family"), or None. A weak word
        ("relationship", "related") counts only when the masked question names a family member (family_N / person_N)."""
        m = self._cues.search(question) if self._cues else None
        if m:
            return m.group(1)
        if masked and self._weak_cues and re.search(r"\b(?:family|person)_\d+\b", masked, re.I):
            w = self._weak_cues.search(question)
            return w.group(1) if w else None
        return None

    # ------------------------------------------------------------------ normalisation (Python)
    def normalize(self, raw: Any) -> str:
        text = _py_text(raw).strip().lower()
        return self.aliases.get(text, text)

    def expand(self, word: str) -> list[str]:
        """A question's relation word → the relation values it means ("spouse" → spouse, wife, husband)."""
        w = word.strip().lower()
        w = self.aliases.get(w, w)
        if w in self.terms:
            return list(self.terms[w])
        singular = w[:-1] if w.endswith("s") and w[:-1] in self.terms else None
        if singular:
            return list(self.terms[singular])
        return [w[:-1]] if w.endswith("s") and len(w) > 3 and not w.endswith("ss") else [w]

    # ------------------------------------------------------------------ MongoDB
    def _relation_expr(self) -> dict:
        r = {"$toLower": {"$trim": {"input": _text("$relation")}}}
        branches = [{"case": {"$eq": ["$$r", k]}, "then": v} for k, v in self.aliases.items()]
        return {"$let": {"vars": {"r": r}, "in": {"$switch": {"branches": branches, "default": "$$r"}} if branches else "$$r"}}

    @cached_property
    def lookup_stage(self) -> dict:
        project = {"_id": 0, "memberId": 1, "rel": 1, "typ": 1, "name": 1, "relationRaw": 1, "phone": 1, "email": 1,
                   **{f"f_{flag}": 1 for flag in self.flags}}  # fmt: skip
        return {
            "$lookup": {
                "from": self.collection,
                "localField": self.local_field,
                "foreignField": self.foreign_field,
                "pipeline": [
                    {"$set": {
                        "memberId": {"$toString": "$_id"},  # internal key: never shown, never sent to a model
                        "rel": self._relation_expr(),
                        "typ": {"$toLower": {"$trim": {"input": _text("$type")}}},
                        "name": {"$trim": {"input": _text("$name")}},
                        "relationRaw": {"$trim": {"input": _text("$relation")}},
                        "phone": {"$trim": {"input": _text("$phone")}},
                        "email": {"$trim": {"input": _text("$email")}},
                        **{f"f_{flag}": {"$eq": [f"${src}", True]} for flag, src in self.flags.items()},
                    }},
                    *([{"$match": {"rel": {"$nin": self.exclude}}}] if self.exclude else []),
                    {"$sort": {"_id": 1}},
                    {"$limit": self.cap},
                    {"$project": project},
                ],
                "as": "_family",
            }
        }  # fmt: skip

    @cached_property
    def view_fields(self) -> dict[str, Any]:
        """$project expressions over the joined `_family` array (one value per resident), with contact details."""
        return self.fields(contacts=True)

    def fields(self, contacts: bool) -> dict[str, Any]:
        """`contacts`: include phone and email in familyContacts (only for an answer that shows them)."""
        member_label = {
            "$concat": ["$$m.name", {"$cond": [{"$eq": ["$$m.relationRaw", ""]}, "", {"$concat": [" (", "$$m.relationRaw", ")"]}]}]
        }
        return {
            "familyCount": {"$size": "$_family"},
            "familyRelations": {"$setDifference": [{"$setUnion": ["$_family.rel", []]}, [""]]},
            "familyTypes": {"$setDifference": [{"$setUnion": ["$_family.typ", []]}, [""]]},
            "familyFlags": {"$concatArrays": [
                {"$cond": [{"$in": [True, f"$_family.f_{flag}"]}, [flag], []]} for flag in self.flags
            ]},
            "familyMembers": {"$map": {"input": "$_family", "as": "m", "in": member_label}},
            "familyContacts": {"$map": {"input": "$_family", "as": "m", "in": {
                "id": "$$m.memberId", "name": "$$m.name", "relation": "$$m.relationRaw", "rel": "$$m.rel", "type": "$$m.typ",
                **({"phone": "$$m.phone", "email": "$$m.email"} if contacts else {}),
            }}},
            "familyIds": "$_family.memberId",
        }  # fmt: skip

    # ------------------------------------------------------------------ in memory (mirrors the stages above)
    def derive(self, raw_members: list[dict], contacts: bool = True) -> dict[str, Any]:
        members = []
        for m in sorted(raw_members, key=lambda d: str(d.get("_id", ""))):
            rel = self.normalize(m.get("relation"))
            if rel in self.exclude:
                continue
            members.append({
                "memberId": str(m.get("_id", "")), "rel": rel, "typ": _py_text(m.get("type")).strip().lower(), "name": _py_text(m.get("name")).strip(),
                "relationRaw": _py_text(m.get("relation")).strip(), "phone": _py_text(m.get("phone")).strip(),
                "email": _py_text(m.get("email")).strip(),
                **{f"f_{flag}": m.get(src) is True for flag, src in self.flags.items()},
            })  # fmt: skip
        members = members[: self.cap]
        return {
            "familyCount": len(members),
            "familyRelations": sorted({m["rel"] for m in members} - {""}),
            "familyTypes": sorted({m["typ"] for m in members} - {""}),
            "familyFlags": [flag for flag in self.flags if any(m[f"f_{flag}"] for m in members)],
            "familyMembers": [m["name"] + (f" ({m['relationRaw']})" if m["relationRaw"] else "") for m in members],
            "familyContacts": [
                {"id": m["memberId"], "name": m["name"], "relation": m["relationRaw"], "rel": m["rel"], "type": m["typ"],
                 **({"phone": m["phone"], "email": m["email"]} if contacts else {})}
                for m in members
            ],
            "familyIds": [m["memberId"] for m in members],
        }  # fmt: skip

    def vocabulary_rows(self, raw_members: list[dict]) -> list[dict]:
        """Family members of one resident for the entity index: id, name and relation only (no contact details)."""
        d = self.derive(raw_members, contacts=False)
        return [{"id": c["id"], "name": c["name"], "rel": c["rel"], "relation": c["relation"]} for c in d["familyContacts"]]

    @cached_property
    def vocabulary_pipeline_tail(self) -> list[dict]:
        """After the resident base match: one row per resident with the family names the index needs."""
        return [
            self.lookup_stage,
            {"$project": {"_id": 0, "residentId": {"$toString": "$_id"},
                          "facilityId": {"$convert": {"input": "$facilityId", "to": "string", "onError": None, "onNull": None}},
                          "members": {"$map": {"input": "$_family", "as": "m", "in": {
                              "id": "$$m.memberId", "name": "$$m.name", "rel": "$$m.rel", "relation": "$$m.relationRaw"}}}}},
            {"$match": {"members.0": {"$exists": True}}},
        ]  # fmt: skip


@lru_cache
def family_relation() -> FamilyRelation | None:
    cfg = (schema_config().get("relations") or {}).get("family")
    return FamilyRelation(cfg) if cfg else None


def flag_names() -> tuple[str, ...]:
    rel = family_relation()
    return tuple(rel.flags) if rel else ()


def plan_uses_family(filters=None, fields: list[str] | None = None, sort_field: str | None = None) -> bool:
    """Does a query need the family join? Only then is it added (every other query runs exactly as before)."""
    if filters is not None and any(getattr(filters, f, None) is not None for f in FAMILY_PLAN_FIELDS):
        return True
    if fields and any(f in FAMILY_VIEW_FIELDS for f in fields):
        return True
    return sort_field == "family_count"
