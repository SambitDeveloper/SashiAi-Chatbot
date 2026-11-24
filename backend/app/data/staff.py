"""The ONE definition of staff semantics (resident_schema.yaml `relations.staff`), used two ways:

  • a RELATION of residents (residents.assignedStaff[] → staffs.cName; residents.primaryPhysicianCName → staffs.cName),
    joined only when a question uses it, contributing resident-level fields so a resident is still one row:
      staffRoles        canonical roles of the resident's ACTIVE assigned staff ("residents without a case manager")
      staffDesignations designations (as recorded) of the resident's ACTIVE assigned staff
      staffCount        number of ACTIVE assigned staff
      careTeam          every assigned staff member as "Name (Designation)", inactive ones marked "(…, inactive)"
      careTeamMembers   the same as objects {name, designation, role, status}: one resident's answer only
      primaryPhysician  residents.primaryPhysicianCName's staff member, "Name (Designation)" (kept distinct from the
                        assigned staff and from the designation "Physician")
  • an ENTITY of its own (the `staffs` collection): one row per staff member with
      staffId (cName, internal key), name, designation, role, status (active / inactive), facility, type,
      residentCount (assigned live residents, optionally only those matching resident conditions),
      residents (their names: one staff member's answer only), email / phone (one staff member's answer only).

Both the MongoDB stages and the identical in-memory derivation are built here from the same configuration, so the two
execution paths cannot disagree. Designations listed in `hidden_designations` (test values, system accounts) are not
people: they are left out of every staff answer, count and care team. The recorded designation is what is shown; the
canonical role is used only to match question words ("doctors" → physician).
"""

import re
from functools import cached_property, lru_cache
from typing import Any

from app.data import text_match
from app.data.schema_config import schema_config
from app.models.plan import STAFF_ENTITY_SET_FIELDS, STAFF_FIELD_MAP, Filters

# resident view fields the staff relation derives
STAFF_RESIDENT_FIELDS = ("staffRoles", "staffDesignations", "staffCount", "staffNames", "careTeam", "careTeamMembers",
                         "primaryPhysician")
STAFF_STATUSES = ("active", "inactive")


def _text(expr: str) -> dict:
    return {"$convert": {"input": expr, "to": "string", "onError": "", "onNull": ""}}


def _py_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _plural(word: str) -> str:
    if word.endswith("s"):
        return word
    if word.endswith("y") and not word.endswith(("ay", "ey", "oy")):
        return word[:-1] + "ies"
    return word + "s"


MEMBERSHIP_COLLECTION = "stafffacilitymemberships"  # per-facility app permissions of a staff member


def _norm_section(name) -> str:
    return " ".join(str(name or "").replace("-", " ").replace("_", " ").lower().split())


def can_access(permissions: list | None, section: str) -> bool | None:
    """Can this staff member open `section` (an app permission name, top level or child)? Allowed AND readable; a
    child section also needs its parent. None = the section is not in their permissions at all."""
    want = _norm_section(section)

    def ok(p: dict) -> bool:
        return p.get("allowed") is True and p.get("isRead") is True

    for top in permissions or []:
        if not isinstance(top, dict):
            continue
        if _norm_section(top.get("name")) == want:
            return ok(top)
        for child in top.get("children") or []:
            if isinstance(child, dict) and _norm_section(child.get("name")) == want:
                return ok(top) and ok(child)
    return None


def _phone_expr() -> dict:
    """The phone as shown: "+1 5551234567" — the recorded countryCode in front, unless the number already starts with
    "+" or no code is recorded. No number → "" (a code alone is not a phone)."""
    return {"$let": {
        "vars": {"p": {"$trim": {"input": _text("$phone")}}, "c": {"$trim": {"input": _text("$countryCode")}}},
        "in": {"$cond": [
            {"$or": [{"$eq": ["$$p", ""]}, {"$eq": ["$$c", ""]}, {"$eq": [{"$substrCP": ["$$p", 0, 1]}, "+"]}]},
            "$$p", {"$concat": ["$$c", " ", "$$p"]},
        ]},
    }}  # fmt: skip


def _py_phone(raw: dict) -> str:
    """In memory, the same as _phone_expr."""
    phone, code = _py_text(raw.get("phone")).strip(), _py_text(raw.get("countryCode")).strip()
    return phone if not phone or not code or phone.startswith("+") else f"{code} {phone}"


class StaffConfig:
    def __init__(self, config: dict):
        self.collection: str = config.get("collection", "staffs")
        self.key: str = config.get("key", "cName")
        self.resident_link: str = config.get("resident_link", "assignedStaff")
        self.physician_link: str = config.get("physician_link", "primaryPhysicianCName")
        self.cap: int = int(config.get("max_per_resident", 200))
        self.roles: dict[str, list[str]] = {str(r): [str(d) for d in ds] for r, ds in (config.get("roles") or {}).items()}
        self.role_of: dict[str, str] = {d.strip().lower(): r for r, ds in self.roles.items() for d in ds}
        self.groups: dict[str, list[str]] = {str(g): [str(r) for r in rs] for g, rs in (config.get("role_groups") or {}).items()}
        self.hidden: list[str] = sorted({str(d).strip().lower() for d in config.get("hidden_designations") or []})
        self.types: dict[str, str] = {str(k): str(v) for k, v in (config.get("types") or {}).items()}
        # question word → the roles it means. Built from the role names, their designations, the groups and the extra
        # role_words, each with its plural ("case managers", "therapists").
        words: dict[str, list[str]] = {}

        def add(word: str, roles: list[str]) -> None:
            for w in {word.strip().lower(), _plural(word.strip().lower())}:
                if w:
                    words[w] = list(dict.fromkeys([*words.get(w, []), *roles]))

        for role, designations in self.roles.items():
            add(role.replace("_", " "), [role])
            for d in designations:
                add(d, [role])
        for group, members in self.groups.items():
            add(group, members)
        for name, extra in (config.get("role_words") or {}).items():
            roles = self.groups.get(name) or ([name] if name in self.roles else [])
            for w in extra or []:
                add(str(w), roles)
        not_roles = {str(w).strip().lower() for w in config.get("not_role_words") or []}
        self.words = {w: rs for w, rs in words.items() if rs and w not in not_roles}
        self.generic: set[str] = {str(w).strip().lower() for w in config.get("generic_words") or []}
        self._role_pattern = self._pattern(self.words)
        self._cue_pattern = self._pattern([*self.words, *(str(w).lower() for w in config.get("cue_words") or [])])

    @staticmethod
    def _pattern(words) -> re.Pattern | None:
        ordered = sorted({str(w) for w in words}, key=len, reverse=True)
        if not ordered:
            return None
        return re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(w) for w in ordered) + r")(?![a-z0-9])", re.I)

    # ------------------------------------------------------------------ vocabulary
    @property
    def role_names(self) -> list[str]:
        return list(self.roles)

    def role_for(self, designation: Any) -> str | None:
        return self.role_of.get(_py_text(designation).lower())

    def is_hidden(self, designation: Any) -> bool:
        return _py_text(designation).lower() in self.hidden

    def expand(self, word: str) -> list[str]:
        """A question's role word → the canonical roles it means ("doctors" → physician; "therapists" → the rehab
        therapy roles). A canonical role name is itself. Unknown → []."""
        w = re.sub(r"\s+", " ", str(word).strip().lower().replace("_", " "))
        if w.replace(" ", "_") in self.roles:
            return [w.replace(" ", "_")]
        return list(self.words.get(w, []))

    def roles_in(self, text: str) -> list[tuple[str, list[str]]]:
        """Role words of a text, longest first and non-overlapping: [("case managers", ["case_manager"]), ...]."""
        if self._role_pattern is None:
            return []
        return [(m.group(1), self.words[m.group(1).lower()]) for m in self._role_pattern.finditer(text)]

    def specific_roles_in(self, text: str, name_tokens: set[str] | frozenset[str] = frozenset(),
                          include_generic: bool = False) -> list[tuple[str, list[str]]]:
        """Role words that name a ROLE in the question: not a generic staff word ("nurse", "staff") and not a title in
        front of a name ("Dr. Patel", "doctor smith": the next word is capitalised or part of a staff name)."""
        out = []
        for m in self._role_pattern.finditer(text) if self._role_pattern else []:
            word = m.group(1)
            if word.lower() in self.generic and not include_generic:
                continue
            nxt = re.match(r"\.?\s+([A-Za-z][\w'’-]*)", text[m.end():])
            if nxt and (nxt.group(1)[0].isupper() and not text[m.end():].startswith((" I ", " A "))
                        or nxt.group(1).lower().rstrip("'’s") in name_tokens):
                continue
            out.append((word, self.words[word.lower()]))
        return out

    def cue_in(self, text: str) -> str | None:
        """The first word of the question that is about staff ("physician", "care team", "staff"), or None."""
        m = self._cue_pattern.search(text) if self._cue_pattern else None
        return m.group(1) if m else None

    # ------------------------------------------------------------------ MongoDB expressions
    def _designation_expr(self, src: str = "$designation") -> dict:
        return {"$trim": {"input": _text(src)}}

    def _role_expr(self, designation: Any) -> dict:
        branches = [{"case": {"$eq": ["$$d", d]}, "then": r} for d, r in self.role_of.items()]
        return {"$let": {"vars": {"d": {"$toLower": designation}}, "in": {"$switch": {"branches": branches, "default": None}}}}

    def _type_expr(self) -> dict:
        branches = [{"case": {"$eq": ["$type", k]}, "then": v} for k, v in self.types.items()]
        return {"$switch": {"branches": branches, "default": ""}} if branches else {"$literal": ""}

    @property
    def visible_match(self) -> dict:
        """Staff records that are people (not a hidden test value / system account)."""
        return {"$match": {"$expr": {"$not": [{"$in": [{"$toLower": self._designation_expr()}, self.hidden]}]}}}

    def _member_project(self) -> dict:
        """One staff member, as the care team / entity rows need it (never contact details or other fields)."""
        return {
            "_id": 0,
            "staffId": _text(f"${self.key}"),
            "name": {"$trim": {"input": _text("$name")}},
            "designation": self._designation_expr(),
            "role": self._role_expr(self._designation_expr()),
            "status": {"$cond": [{"$eq": ["$active", True]}, "active", "inactive"]},
        }

    @cached_property
    def team_lookup_stages(self) -> list[dict]:
        """On residents: the assigned staff (`_team`) and the primary physician (`_physician`)."""
        team = [self.visible_match, {"$project": self._member_project()}, {"$sort": {"name": 1, "staffId": 1}}, {"$limit": self.cap}]
        return [
            {"$lookup": {"from": self.collection, "localField": self.resident_link, "foreignField": self.key,
                         "pipeline": team, "as": "_team"}},
            {"$lookup": {"from": self.collection, "localField": self.physician_link, "foreignField": self.key,
                         "pipeline": [self.visible_match, {"$project": self._member_project()}, {"$limit": 1}],
                         "as": "_physician"}},
        ]  # fmt: skip

    @staticmethod
    def _label_expr(m: str) -> dict:
        """"Name (Designation)", "Name (Designation, inactive)", "Name (inactive)" or "Name"."""
        inactive = {"$eq": [f"{m}.status", "inactive"]}
        has_d = {"$ne": [f"{m}.designation", ""]}
        inner = {"$cond": [has_d, {"$concat": [f"{m}.designation", {"$cond": [inactive, ", inactive", ""]}]},
                           {"$cond": [inactive, "inactive", ""]}]}  # fmt: skip
        return {"$let": {"vars": {"i": inner}, "in": {"$concat": [f"{m}.name", {"$cond": [{"$eq": ["$$i", ""]}, "", {"$concat": [" (", "$$i", ")"]}]}]}}}

    @cached_property
    def resident_fields(self) -> dict[str, Any]:
        active = {"$filter": {"input": "$_team", "as": "m", "cond": {"$eq": ["$$m.status", "active"]}}}
        physician = {"$first": "$_physician"}
        return {
            "staffRoles": {"$setDifference": [{"$setUnion": [{"$map": {"input": active, "as": "m", "in": "$$m.role"}}, []]}, [None, ""]]},
            "staffDesignations": {"$setDifference": [{"$setUnion": [{"$map": {"input": active, "as": "m", "in": "$$m.designation"}}, []]}, [""]]},
            "staffCount": {"$size": active},
            # every assigned staff member's recorded name (active or not): "is Jone part of X's care team?"
            "staffNames": {"$setDifference": [{"$setUnion": [{"$map": {"input": "$_team", "as": "m", "in": "$$m.name"}}, []]}, [""]]},
            "careTeam": {"$map": {"input": "$_team", "as": "m", "in": self._label_expr("$$m")}},
            "careTeamMembers": {"$map": {"input": "$_team", "as": "m", "in": {
                "name": "$$m.name", "designation": "$$m.designation", "role": "$$m.role", "status": "$$m.status"}}},
            "primaryPhysician": {"$cond": [{"$gt": [{"$size": "$_physician"}, 0]},
                                           {"$let": {"vars": {"p": physician}, "in": self._label_expr("$$p")}}, ""]},
        }  # fmt: skip

    def entity_stages(self, contacts: bool = False) -> list[dict]:
        """The staff view: one row per staff member who is a person, with the facility's name (like residents' facility),
        contact details only when `contacts` (one staff member's answer)."""
        project = {
            **self._member_project(),
            "facility": {"$convert": {"input": {"$ifNull": [{"$first": "$_facility.facilityName"}, "$facilityId"]},
                                      "to": "string", "onError": "", "onNull": ""}},
            "type": self._type_expr(),
            **({"email": {"$trim": {"input": _text("$email")}}, "phone": _phone_expr()} if contacts else {}),
        }  # fmt: skip
        return [
            self.visible_match,
            {"$lookup": {"from": "facilities", "localField": "facilityId", "foreignField": "facilityId",
                         "pipeline": [{"$project": {"_id": 0, "facilityName": 1}}], "as": "_facility"}},
            {"$project": project},
        ]  # fmt: skip

    def access_stages(self, name: str, designation: str) -> list[dict]:
        """ONE staff member's app permissions (accessPermissions): the membership of their own facility
        (stafffacilitymemberships, per facility), else the copy on the staff record. Matched by recorded name and
        designation, exactly as the staff view shows them; at most 2 rows so a duplicate is noticed, never guessed."""
        return [
            self.visible_match,
            {"$match": {"$expr": {"$and": [{"$eq": [{"$trim": {"input": _text("$name")}}, name]},
                                           {"$eq": [self._designation_expr(), designation]}]}}},
            {"$limit": 2},
            {"$lookup": {"from": MEMBERSHIP_COLLECTION, "let": {"sid": "$_id", "f": "$facilityId"},
                         "pipeline": [{"$match": {"$expr": {"$and": [{"$eq": ["$staffId", "$$sid"]}, {"$eq": ["$facilityId", "$$f"]}]}}},
                                      {"$project": {"_id": 0, "accessPermissions": 1}}, {"$limit": 1}],
                         "as": "_m"}},
            {"$project": {"_id": 0, "access": {"$ifNull": [{"$first": "$_m.accessPermissions"}, "$accessPermissions"]}}},
        ]  # fmt: skip

    # ------------------------------------------------------------------ in memory (mirrors the stages above)
    def member(self, raw: dict) -> dict:
        designation = _py_text(raw.get("designation"))
        return {
            "staffId": _py_text(raw.get(self.key)) if isinstance(raw.get(self.key), str) else str(raw.get(self.key) or ""),
            "name": _py_text(raw.get("name")),
            "designation": designation,
            "role": self.role_of.get(designation.lower()),
            "status": "active" if raw.get("active") is True else "inactive",
        }

    def entity_record(self, raw: dict, facility_names: dict[str, str], contacts: bool = False) -> dict:
        fid = raw.get("facilityId")
        return {
            **self.member(raw),
            "facility": facility_names.get(fid) or (fid if isinstance(fid, str) else ""),
            "type": self.types.get(raw.get("type"), "") if isinstance(raw.get("type"), str) else "",
            **({"email": _py_text(raw.get("email")).strip(), "phone": _py_phone(raw)} if contacts else {}),
        }

    @staticmethod
    def _label(m: dict) -> str:
        inner = ", ".join(x for x in (m["designation"], "inactive" if m["status"] == "inactive" else "") if x)
        return m["name"] + (f" ({inner})" if inner else "")

    def derive(self, team_raw: list[dict], physician_raw: dict | None) -> dict[str, Any]:
        team = sorted((self.member(s) for s in team_raw if not self.is_hidden(s.get("designation"))),
                      key=lambda m: (m["name"], m["staffId"]))[: self.cap]  # fmt: skip
        active = [m for m in team if m["status"] == "active"]
        physician = self.member(physician_raw) if physician_raw and not self.is_hidden(physician_raw.get("designation")) else None
        return {
            "staffRoles": sorted({m["role"] for m in active if m["role"]}),
            "staffDesignations": sorted({m["designation"] for m in active if m["designation"]}),
            "staffCount": len(active),
            "staffNames": sorted({m["name"] for m in team if m["name"]}),
            "careTeam": [self._label(m) for m in team],
            "careTeamMembers": [{k: m[k] for k in ("name", "designation", "role", "status")} for m in team],
            "primaryPhysician": self._label(physician) if physician else "",
        }


@lru_cache
def staff_config() -> StaffConfig | None:
    cfg = (schema_config().get("relations") or {}).get("staff")
    return StaffConfig(cfg) if cfg else None


# ---------------------------------------------------------------------- staff-entity filters (both engines)
def staff_part(filters: Filters) -> Filters:
    """Only the staff record's own conditions of a staff-entity plan."""
    return Filters(**{n: getattr(filters, n) for n in (*STAFF_ENTITY_SET_FIELDS, "staff_resident_count") if getattr(filters, n) is not None})


def resident_part(filters: Filters) -> Filters:
    """The conditions on a staff member's ASSIGNED residents (every non-staff field of a staff-entity plan)."""
    return filters.model_copy(update=dict.fromkeys((*STAFF_ENTITY_SET_FIELDS, "staff_resident_count")))


def _missing(value: Any) -> bool:
    return value is None or value == ""


def staff_mongo_filter(filters: Filters) -> dict[str, Any]:
    """The staff record's conditions as a MongoDB query on the staff view. Mirrors `staff_matches` exactly."""
    clauses: list[dict[str, Any]] = []
    for plan_field in STAFF_ENTITY_SET_FIELDS:
        f = getattr(filters, plan_field)
        if f is None:
            continue
        field = STAFF_FIELD_MAP[plan_field]
        if f.include or f.missing is True:
            alternatives = [{field: {"$in": list(f.include)}}] if f.include else []
            if f.missing is True:
                alternatives.append({field: {"$in": [None, ""]}})
            clauses.append(alternatives[0] if len(alternatives) == 1 else {"$or": alternatives})
        if f.exclude:
            clauses.append({field: {"$nin": list(f.exclude)}})
        if f.missing is False:
            clauses.append({field: {"$nin": [None, ""]}})
        if f.has_text_match:
            clauses.extend(text_match.mongo_clauses(field, f.starts_with, f.contains))
    r = filters.staff_resident_count
    if r is not None:
        cond: dict[str, Any] = {}
        if r.min is not None:
            cond["$gte"] = r.min
        if r.max is not None:
            cond["$lte"] = r.max
        clauses.append({"residentCount": cond})
    if not clauses:
        return {}
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def staff_matches(record: dict, filters: Filters) -> bool:
    """In memory: does one staff view row satisfy the staff record's conditions? Mirrors `staff_mongo_filter`."""
    for plan_field in STAFF_ENTITY_SET_FIELDS:
        f = getattr(filters, plan_field)
        if f is None:
            continue
        value = record.get(STAFF_FIELD_MAP[plan_field])
        if f.include or f.missing is True:
            if not ((f.include and value in f.include) or (f.missing is True and _missing(value))):
                return False
        if f.exclude and value in f.exclude:
            return False
        if f.missing is False and _missing(value):
            return False
        if f.has_text_match and not text_match.value_matches(value, STAFF_FIELD_MAP[plan_field], f.starts_with, f.contains):
            return False
    r = filters.staff_resident_count
    if r is not None:
        n = record.get("residentCount", 0)
        if (r.min is not None and n < r.min) or (r.max is not None and n > r.max):
            return False
    return True


def plan_uses_staff(filters: Filters | None = None, fields: list[str] | None = None) -> bool:
    """Does a RESIDENT query need the staff join? Only then is it added (every other query runs exactly as before)."""
    from app.models.plan import STAFF_RELATION_FIELDS

    if filters is not None and any(getattr(filters, f, None) is not None for f in STAFF_RELATION_FIELDS):
        return True
    return bool(fields) and any(f in STAFF_RESIDENT_FIELDS for f in fields)
