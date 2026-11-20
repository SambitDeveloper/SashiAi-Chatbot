"""Semantic query-plan template cache (replaces the exact-fingerprint plan cache, 2026-11-18).

A TEMPLATE is a checked plan with every variable value replaced by a typed slot, stored with the embedding of its
slot-replaced question in a dedicated MongoDB Atlas collection (Vector Search). A later question with different
wording can reuse it — but similarity only NOMINATES candidates; it never decides:

    question → (fast path could not answer) → slot-replaced text → BGE embedding → $vectorSearch (not revoked,
    same compatibility key) → for each candidate above the similarity threshold:
        logic words equal? → fill every slot from THIS question (residents, dates, numbers, statuses, care types,
        genders, facilities, caregivers, limit) → existing resolve_plan + check_plan against THIS question
    → exactly one distinct plan passes → reuse;  anything else → the existing Qwen path.

Measured on BGE-small (2026-11-18): "older than 75" vs "younger than 75" = 0.984 cosine, "Active or Discharged" vs
"Active and Discharged" = 0.988 — higher than true paraphrases (0.915–0.927). The threshold is therefore NOT a safety
boundary; the slot shapes, the logic-word signature and the existing validator are.

A template is stored ready to use once its plan passed validation and executed (no review step, decided 2026-11-18);
`revoked: true` (admin) switches a wrong template off immediately. Every reuse is re-validated against the new question.
Nothing here stores a raw question, a resident/caregiver/facility name, a resident id or a query result.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from app.engine.grounding import NAMED, SYNONYMS, DateScan, date_fields_mentioned
from app.nlu import placeholders as ph

log = logging.getLogger(__name__)

TEMPLATE_DOC_VERSION = 2  # bump when the document/slot format changes: older documents are then incompatible
# v2 (2026-11-20): a synonym that only modifies an attribute ("current status") is no longer a {status} slot
WRITE_TIMEOUT_S = 45.0  # background template writes: after the answer, may include a cold Atlas connect
DATA_ACTIONS = {"count", "list", "breakdown", "aggregate", "lookup"}
VOCAB_FIELDS = ("status", "care_type", "gender")
NAME_FIELDS = ("facility", "primary_caregiver")
DATE_FIELDS = ("admission_date", "created_date", "discharge_date")
_PLACEHOLDER = re.compile(r"^resident_\d+$")
_PLACEHOLDER_IN_TEXT = ph.ANY_IN_TEXT  # any typed placeholder (app/nlu/placeholders.py)
_INT = re.compile(r"(?<![\w.])\d+(?![\w.])")

# Logic words: two questions whose slot-replaced texts differ in these are different requests, however close their
# embeddings are ("Active or Discharged" vs "Active and Discharged"; "each care level" vs "all residents"). Synonyms
# map to one token. Comparison direction (older/younger, before/after) is carried by the slot SHAPES instead.
_LOGIC_TOKENS = {
    "and": "AND", "or": "OR",
    "not": "NOT", "no": "NOT", "never": "NOT", "without": "NOT", "except": "NOT", "excluding": "NOT",
    "nor": "NOT", "neither": "NOT", "none": "NOT", "isn't": "NOT", "aren't": "NOT", "doesn't": "NOT",
    "each": "EACH", "every": "EACH", "per": "EACH",
    "both": "BOTH", "only": "ONLY",
    "oldest": "OLDEST", "youngest": "YOUNGEST", "latest": "NEWEST", "newest": "NEWEST", "earliest": "OLDEST_DATE",
    "average": "AVG", "mean": "AVG", "minimum": "MIN", "maximum": "MAX", "total": "TOTAL",
}  # fmt: skip


# ---------------------------------------------------------------------------------------------------- question view
@dataclass
class QuestionView:
    """Everything slot handling needs about ONE question (computed by the service with its existing tools)."""

    masked_text: str  # the question with resident names replaced by resident_N (vocabulary.mask)
    placeholders: list[str]  # resident_N placeholders found in the question
    scan: DateScan  # dates and age numbers parsed from masked_text (grounding.scan_question)
    index: Any  # EntityIndex: statuses, care types, wings, caregivers of the live database
    date_anchors: dict[str, list[str]]
    placeholder_ids: dict[str, list[str]] = field(default_factory=dict)  # resident_N → resident ids (vocabulary.mask)


@dataclass
class Mention:
    start: int
    end: int
    kind: str  # date | number | status | care_type | gender | facility | primary_caregiver | integer
    value: Any


_ROUTES_FILE = Path(__file__).resolve().parents[1] / "data" / "intent_routes.yaml"  # = intent_router.DEFAULT_ROUTES_FILE


@lru_cache
def _attribute_phrases() -> re.Pattern | None:
    """The configured words that NAME an attribute (intent_routes.yaml `direct.lookup_attributes`: status, care type,
    room, caregiver, discharge date ...), as one word-bounded pattern (plural -s / -es allowed)."""
    cfg = yaml.safe_load(_ROUTES_FILE.read_text(encoding="utf-8")) or {}
    attrs = ((cfg.get("direct") or {}).get("lookup_attributes")) or {}
    words = sorted({" ".join(str(w).lower().split()) for ws in attrs.values() for w in ws or []}, key=len, reverse=True)
    if not words:
        return None
    alts = "|".join(r"\s+".join(map(re.escape, w.split())) for w in words)
    return re.compile(r"(?<![a-z])(?:" + alts + r")(?:e?s)?(?![a-z])")


def _modifies_attribute(text: str, start: int, end: int) -> bool:
    """The words at [start, end) only describe an attribute: they are directly followed by an attribute phrase
    ("current status", "former caregiver", "left wing") or are part of a LONGER attribute phrase ("discharge date")."""
    pattern = _attribute_phrases()
    if pattern is None:
        return False
    for m in pattern.finditer(text):
        s, e = m.span()
        if s <= start and end <= e and (e - s) > (end - start):  # strictly inside a longer attribute phrase
            return True
        if s >= end and not text[end:s].strip():  # the very next word(s) name an attribute
            return True
    return False


def _vocab_mentions(view: QuestionView, name: str) -> list[Mention]:
    """Values of a fixed-vocabulary field the question names, by position (the forms grounding.check_plan uses)."""
    low = view.masked_text.lower()
    text = low
    if name == "status":  # "admitted last month" / "discharged in 2025" are date verbs, not statuses
        cues = date_fields_mentioned(view.masked_text, view.date_anchors) if view.scan.dates else []
        if "admission_date" in cues:
            text = re.sub(r"\badmitted\b", " " * len("admitted"), text)
        if "discharge_date" in cues:
            text = re.sub(r"\bdischarged\b", " " * len("discharged"), text)
    allowed = {"status": set(view.index.statuses or SYNONYMS["status"]),
               "care_type": set(view.index.care_types or SYNONYMS["care_type"]),
               "gender": set(SYNONYMS["gender"])}[name]  # fmt: skip
    forms = sorted(((f, v) for v, fs in SYNONYMS[name].items() if v in allowed for f in fs), key=lambda x: -len(x[0]))
    out: list[Mention] = []
    taken: list[tuple[int, int]] = []
    for form, value in forms:
        canonical = form in NAMED.get(name, {}).get(value, ())
        pattern = r"(?<![a-z])" + r"\s+".join(map(re.escape, form.split())) + r"(?![a-z])"
        for m in re.finditer(pattern, text):
            # a non-canonical synonym that only modifies an attribute is not a value: "current status" asks for the
            # status, it does not name "active". Canonical words ("discharged status") and synonyms used as values
            # ("how many current residents") stay values; grounding's own rules are unchanged.
            if not canonical and _modifies_attribute(text, m.start(), m.end()):
                continue
            if not any(m.start() < e and s < m.end() for s, e in taken):
                taken.append((m.start(), m.end()))
                out.append(Mention(m.start(), m.end(), name, value))
    return sorted(out, key=lambda x: x.start)


def _name_mentions(view: QuestionView, name: str) -> list[Mention] | None:
    """Facilities / caregivers named in the question. None when a mention is ambiguous (several database values)."""
    found = view.index.mentioned_wings(view.masked_text) if name == "facility" else view.index.mentioned_caregivers(view.masked_text)
    low = view.masked_text.lower()
    out: list[Mention] = []
    for text, values in found.items():
        if len(set(values)) != 1:
            return None
        pos = low.find(text.lower())
        if pos < 0:
            return None
        out.append(Mention(pos, pos + len(text), name, values[0]))
    return sorted(out, key=lambda x: x.start)


def mentions(view: QuestionView) -> list[Mention] | None:
    """Every slot-able value in the question, non-overlapping, by position. None = a name mention is ambiguous."""
    found: list[Mention] = []
    found += [Mention(d.start, d.end, "date", d.range) for d in view.scan.dates]
    found += [Mention(n.start, n.end, "number", n.bounds) for n in view.scan.numbers]
    for name in NAME_FIELDS:
        got = _name_mentions(view, name)
        if got is None:
            return None
        found += got
    for name in VOCAB_FIELDS:
        found += _vocab_mentions(view, name)
    out: list[Mention] = []
    for m in sorted(found, key=lambda x: (x.start, -(x.end - x.start))):
        if not any(m.start < o.end and o.start < m.end for o in out):
            out.append(m)
    text = view.masked_text
    for m in _INT.finditer(text):  # bare integers left over ("the 5 oldest"), never inside resident_N
        if not any(m.start() < o.end and o.start < m.end() for o in out) and not _inside_placeholder(text, m.start()):
            out.append(Mention(m.start(), m.end(), "integer", int(m.group(0))))
    return sorted(out, key=lambda x: x.start)


def _inside_placeholder(text: str, pos: int) -> bool:
    return any(p.start() <= pos < p.end() for p in _PLACEHOLDER_IN_TEXT.finditer(text))


def slot_text(view: QuestionView, found: list[Mention]) -> str:
    """The question with every value replaced by its type: what is embedded and stored (never a raw question)."""
    text = view.masked_text
    for m in sorted(found, key=lambda x: -x.start):
        text = text[: m.start] + "{" + m.kind + "}" + text[m.end :]
    return " ".join(text.split())


def logic_signature(text: str) -> list[str]:
    words = re.findall(r"[a-z']+", text.lower())
    return sorted({_LOGIC_TOKENS[w] for w in words if w in _LOGIC_TOKENS})


# ---------------------------------------------------------------------------------------------------- templates
class Ineligible(Exception):
    """This plan cannot become a reusable template (the reason says why); it is simply not recorded."""


def _canon(plan: dict) -> str:
    def norm(v):
        if isinstance(v, dict):
            return {k: norm(x) for k, x in v.items() if x is not None}
        if isinstance(v, list):
            items = [norm(x) for x in v]
            return sorted(items, key=lambda x: json.dumps(x, sort_keys=True)) if all(not isinstance(x, (dict, list)) for x in items) else items
        return v

    return json.dumps(norm(plan), sort_keys=True, default=str)


def templatize(raw: dict, view: QuestionView) -> tuple[dict, list[dict]]:
    """Checked masked plan → (plan template, slot definitions). Raises Ineligible.

    Every variable value must be traceable to exactly one mention in the question (otherwise it cannot be filled
    from a later question), and filling the template from its OWN question must give back the same plan."""
    plan = copy.deepcopy(raw)
    # The fast path writes the resolved resident id ("RES-1001"); a template needs the placeholder of the NAME the
    # question used, so a later question's resident fills it. Only an unambiguous name (one id) is mapped back.
    id_to_placeholder = {ids[0].upper(): ph for ph, ids in view.placeholder_ids.items() if len(set(ids)) == 1}
    if (plan.get("lookup") or {}).get("residents"):
        plan["lookup"]["residents"] = [
            r if _PLACEHOLDER.match(str(r)) else id_to_placeholder.get(str(r).upper(), r) for r in plan["lookup"]["residents"]
        ]
    raw = copy.deepcopy(plan)  # the self-check below compares against the placeholder form
    if plan.get("uses_previous") or plan.get("action") not in DATA_ACTIONS:
        raise Ineligible("not a standalone data question")
    if plan.get("options") or plan.get("message"):
        raise Ineligible("plan carries text")
    found = mentions(view)
    if found is None:
        raise Ineligible("an ambiguous facility/caregiver mention")
    slots: list[dict] = []
    filters = plan.get("filters") or {}
    if (filters.get("room_number") or {}).get("include") or (filters.get("room_number") or {}).get("exclude"):
        raise Ineligible("room-number values are not slot-able")

    # residents: the resident_N placeholders the question masked
    lookup = plan.get("lookup") or {}
    residents = lookup.get("residents") or []
    if any(not _PLACEHOLDER.match(str(r)) for r in residents):
        raise Ineligible("a resident given by id, not by a masked name")
    if set(residents) != set(view.placeholders):
        raise Ineligible("the plan does not use exactly the residents the question names")
    if residents:
        slots.append({"name": "residents", "type": "resident", "placeholders": sorted(residents)})

    def by_kind(kind: str) -> list[Mention]:
        return [m for m in found if m.kind == kind]

    # dates: each date range in the plan is the i-th date phrase of the question
    dates = by_kind("date")
    for name in DATE_FIELDS:
        r = filters.get(name)
        if not r or not (r.get("start") or r.get("end")):
            continue
        rng = {k: r[k] for k in ("start", "end") if r.get(k)}
        idx = [i for i, d in enumerate(dates) if d.value == rng]
        if len(idx) != 1:
            raise Ineligible(f"{name} is not exactly one date phrase of the question")
        slot = f"date_{idx[0] + 1}"
        filters[name] = {"$slot": slot, **({"missing": r["missing"]} if "missing" in r else {})}
        slots.append({"name": slot, "type": "date_range", "field": name, "position": idx[0], "shape": sorted(rng)})

    # age: the i-th number phrase
    numbers = by_kind("number")
    age = filters.get("age")
    if age and (age.get("min") is not None or age.get("max") is not None):
        bounds = {k: age[k] for k in ("min", "max") if age.get(k) is not None}
        idx = [i for i, n in enumerate(numbers) if n.value == bounds]
        if len(idx) != 1:
            raise Ineligible("the age range is not exactly one number phrase of the question")
        filters["age"] = {"$slot": f"number_{idx[0] + 1}", **({"missing": age["missing"]} if "missing" in age else {})}
        slots.append({"name": f"number_{idx[0] + 1}", "type": "number_range", "field": "age", "position": idx[0],
                      "shape": sorted(bounds)})  # fmt: skip

    # fixed-vocabulary values and named facilities/caregivers: the values the question names, by field
    for name in VOCAB_FIELDS + NAME_FIELDS:
        sf = filters.get(name)
        if not sf:
            continue
        inc, exc = sf.get("include") or [], sf.get("exclude") or []
        if inc and exc:
            raise Ineligible(f"{name} both includes and excludes values")
        values = inc or exc
        if not values:
            continue  # a "missing"/"present" condition only: structural, kept literally
        named = [m.value for m in by_kind(name)]
        if sorted(set(named)) != sorted(set(values)):
            raise Ineligible(f"{name} values are not exactly the values the question names")
        key = "include" if inc else "exclude"
        sf[key] = {"$slot": f"{name}:{key}"}
        slots.append({"name": f"{name}:{key}", "type": name, "count": len(set(values))})

    if plan.get("limit") is not None:
        ints = [m.value for m in by_kind("integer")]
        if ints != [plan["limit"]]:
            raise Ineligible("the limit is not the only bare number of the question")
        plan["limit"] = {"$slot": "limit"}
        slots.append({"name": "limit", "type": "integer"})

    plan["filters"] = filters
    template = {"plan": plan, "slots": slots, "mention_counts": _counts(found)}
    filled, problem = fill(template, view)
    if problem or _canon(filled) != _canon(raw):
        raise Ineligible(f"the template does not reproduce its own plan ({problem or 'different plan'})")
    return plan, slots


def _counts(found: list[Mention]) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in found:
        out[m.kind] = out.get(m.kind, 0) + 1
    return out


def fill(template: dict, view: QuestionView) -> tuple[dict | None, str | None]:
    """Fill every slot from THIS question. (plan, None) or (None, reason). Never keeps a stored value."""
    found = mentions(view)
    if found is None:
        return None, "ambiguous facility/caregiver mention"
    # the question must contain the same KINDS and NUMBER of values as the template's question: an extra date, number
    # or named value is a condition the template cannot express
    if _counts(found) != template.get("mention_counts", {}):
        return None, "different set of values in the question"
    plan = copy.deepcopy(template["plan"])
    filters = plan.get("filters") or {}
    by_kind: dict[str, list[Mention]] = {}
    for m in found:
        by_kind.setdefault(m.kind, []).append(m)
    for slot in template["slots"]:
        t = slot["type"]
        if t == "resident":
            if sorted(view.placeholders) != slot["placeholders"]:
                return None, "different residents named"
            continue  # resident_N stays: resolve_plan maps it to THIS question's resident
        if t in ("date_range", "number_range"):
            kind = "date" if t == "date_range" else "number"
            items = by_kind.get(kind, [])
            if slot["position"] >= len(items):
                return None, f"missing {kind} value"
            value = items[slot["position"]].value
            if sorted(value) != slot["shape"]:  # after vs before, older vs younger, open vs closed range
                return None, f"different {kind} comparison"
            current = filters.get(slot["field"]) or {}
            filters[slot["field"]] = {**value, **({"missing": current["missing"]} if "missing" in current else {})}
            continue
        if t == "integer":
            ints = [m.value for m in by_kind.get("integer", [])]
            if len(ints) != 1:
                return None, "limit not found"
            plan["limit"] = ints[0]
            continue
        name, key = slot["name"].split(":")
        values = list(dict.fromkeys(m.value for m in by_kind.get(name, [])))
        if len(values) != slot["count"]:
            return None, f"different number of {name} values"
        filters[name][key] = values
    plan["filters"] = filters
    if "$slot" in json.dumps(plan):
        return None, "unfilled slot"
    return plan, None


# ---------------------------------------------------------------------------------------------------- store
class TemplateStore(Protocol):
    async def search(self, vector: list[float], compat_key: str, limit: int, num_candidates: int) -> list[tuple[dict, float]]: ...
    async def store(self, doc: dict) -> str: ...
    async def mark_used(self, template_id: str) -> None: ...


def _aware(value: datetime) -> datetime:
    """MongoDB dates are UTC; a client without tz_aware returns them naive."""
    return value if value.tzinfo else value.replace(tzinfo=UTC)


class AtlasTemplateStore:
    """MongoDB Atlas collection with a Vector Search index. Uses ONLY the cache's own connection string."""

    # A COLD connection to Atlas took ~10 s from the dev machine (SRV lookup + TLS + auth; warm calls ~260 ms,
    # 2026-11-18). The driver may take that long to connect in the background; a REQUEST never waits for it — every
    # lookup is capped by SemanticPlanCache.timeout_s and simply misses (→ Qwen) until the connection is up.
    CONNECT_TIMEOUT_S = 30.0

    TTL_INDEX_NAME = "expires_at_ttl"

    def __init__(self, uri: str, database: str, collection: str, index_name: str, timeout_s: float,
                 ttl_seconds: float = 12 * 3600):  # fmt: skip
        from pymongo import AsyncMongoClient

        self._client = AsyncMongoClient(
            uri,
            serverSelectionTimeoutMS=int(max(timeout_s, self.CONNECT_TIMEOUT_S) * 1000),
            appname="shashicare-plan-cache",
            tz_aware=True,  # expires_at is compared with an aware UTC "now"
        )
        self._coll = self._client[database][collection]
        self._index = index_name
        self.ttl_seconds = ttl_seconds

    async def connect(self) -> None:
        """Open the connection ahead of the first question (called in the background at startup) and make sure the
        TTL index exists: MongoDB then deletes each template once its `expires_at` has passed (checked about once a
        minute by the server, so lookups also skip expired templates themselves)."""
        await self._client.admin.command("ping")
        await self._coll.create_index("expires_at", expireAfterSeconds=0, name=self.TTL_INDEX_NAME)

    async def close(self) -> None:
        await self._client.close()

    async def search(self, vector, compat_key, limit, num_candidates):
        pipeline = [
            {"$vectorSearch": {"index": self._index, "path": "embedding", "queryVector": vector,
                               "numCandidates": num_candidates, "limit": limit,
                               "filter": {"revoked": {"$eq": False}, "compat_key": {"$eq": compat_key}}}},
            {"$project": {"embedding": 0}},
            {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
        ]  # fmt: skip
        out = []
        async for doc in await self._coll.aggregate(pipeline):
            # Atlas reports cosine similarity as (1 + cos) / 2; convert back so the threshold is a plain cosine
            out.append((doc, 2.0 * float(doc.pop("_score")) - 1.0))
        return out

    async def store(self, doc):
        now = datetime.now(UTC)
        fixed = {k: v for k, v in doc.items() if k not in ("_id",)}
        await self._coll.update_one(
            {"_id": doc["_id"]},
            # `revoked` and `expires_at` are set only on insert: storing the same template again never re-enables a
            # revoked one and never extends its lifetime (it is deleted PLAN_CACHE_TTL_HOURS after it was first stored)
            {"$setOnInsert": {**fixed, "revoked": False, "created_at": now, "usage_count": 0,
                              "expires_at": now + timedelta(seconds=self.ttl_seconds)},
             "$set": {"last_seen_at": now}, "$inc": {"seen_count": 1}},
            upsert=True,
        )  # fmt: skip
        return doc["_id"]

    async def mark_used(self, template_id):
        await self._coll.update_one(
            {"_id": template_id}, {"$set": {"last_used_at": datetime.now(UTC)}, "$inc": {"usage_count": 1}}
        )


# ---------------------------------------------------------------------------------------------------- cache
@dataclass
class Compatibility:
    request_ir_version: str
    vocabulary_hash: str  # statuses + care types only: adding a resident/caregiver/room must not invalidate templates
    schema_hash: str
    model_version: str
    embedding_model: str

    @property
    def key(self) -> str:
        payload = json.dumps({**self.__dict__, "doc_version": TEMPLATE_DOC_VERSION}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:32]


@dataclass
class LookupResult:
    plan: dict | None = None
    template_id: str | None = None
    similarity: float | None = None
    reason: str = ""  # why there is no hit (observability; never contains question text or values)
    embed_ms: int = 0
    search_ms: int = 0
    validate_ms: int = 0
    candidates: list[dict] = field(default_factory=list)  # [{id, similarity, verdict}]


@dataclass
class CacheStats:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    errors: int = 0
    recorded: int = 0
    ineligible: int = 0
    rejections: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        key = reason.split(":")[0]
        self.rejections[key] = self.rejections.get(key, 0) + 1


Validator = Callable[[dict], Awaitable[str | None]]  # filled masked plan → None (passes) or the problem


class SemanticPlanCache:
    def __init__(self, store: TemplateStore, embedder_provider: Callable[[], Any], threshold: float, top_k: int,
                 max_candidates: int, timeout_s: float, embedding_model: str, store_templates: bool = True):  # fmt: skip
        self.store = store
        self._embedder_provider = embedder_provider
        self.threshold = threshold
        self.top_k = top_k
        self.max_candidates = max_candidates
        self.timeout_s = timeout_s
        self.embedding_model = embedding_model
        self.store_templates = store_templates
        self.stats = CacheStats()
        self._tasks: set[asyncio.Task] = set()

    async def _embed(self, text: str) -> list[float]:
        embedder = self._embedder_provider()
        if embedder is None:
            raise RuntimeError("embedding model not available")
        vec = await asyncio.to_thread(embedder.embed, [text])
        v = np.asarray(vec[0], dtype=np.float32)
        return (v / max(float(np.linalg.norm(v)), 1e-12)).tolist()

    async def lookup(self, view: QuestionView, compat: Compatibility, validate: Validator) -> LookupResult:
        """Find a stored (not revoked) template that, filled from THIS question, passes the existing validator. Never raises."""
        self.stats.lookups += 1
        res = LookupResult()
        try:
            found = mentions(view)
            if found is None:
                return self._miss(res, "ambiguous_mention")
            text = slot_text(view, found)
            signature = logic_signature(text)
            t = time.perf_counter()
            vector = await asyncio.wait_for(self._embed(text), self.timeout_s)
            res.embed_ms = round((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            hits = await asyncio.wait_for(
                self.store.search(vector, compat.key, self.top_k, self.max_candidates), self.timeout_s
            )
            res.search_ms = round((time.perf_counter() - t) * 1000)
        except Exception as exc:  # embedding/Atlas/index unavailable: a miss, never an error for the user
            self.stats.errors += 1
            log.warning("semantic cache unavailable: %s", type(exc).__name__)
            return self._miss(res, f"unavailable:{type(exc).__name__}")

        passing: list[tuple[dict, float, dict]] = []
        t = time.perf_counter()
        for doc, sim in sorted(hits, key=lambda x: -x[1]):
            verdict = await self._check(doc, sim, compat, signature, view, validate)
            res.candidates.append({"id": doc.get("_id"), "similarity": round(sim, 4), "verdict": verdict[0]})
            if verdict[1] is not None:
                passing.append((doc, sim, verdict[1]))
            else:
                self.stats.reject(verdict[0])
        res.validate_ms = round((time.perf_counter() - t) * 1000)
        if not passing:
            return self._miss(res, "no_candidate" if not hits else "all_candidates_rejected")
        if len({_canon(p) for _, _, p in passing}) > 1:  # two different readings both fit: do not guess
            return self._miss(res, "ambiguous_candidates")
        doc, sim, plan = passing[0]
        res.plan, res.template_id, res.similarity = plan, doc["_id"], sim
        self.stats.hits += 1
        self._background(self.store.mark_used(doc["_id"]))
        return res

    async def _check(self, doc, sim, compat, signature, view, validate) -> tuple[str, dict | None]:
        if sim < self.threshold:
            return "below_threshold", None
        if doc.get("revoked") is not False:  # revoked, or a document without the flag: never reused
            return "revoked", None
        expires = doc.get("expires_at")
        if expires is not None and _aware(expires) <= datetime.now(UTC):  # past its TTL, not yet deleted by MongoDB
            return "expired", None
        if doc.get("doc_version") != TEMPLATE_DOC_VERSION or doc.get("compat_key") != compat.key:
            return "incompatible_version", None
        if doc.get("logic_signature") != signature:
            return "different_logic_words", None
        filled, problem = fill({"plan": doc["plan_template"], "slots": doc["slots"],
                                "mention_counts": doc.get("mention_counts", {})}, view)  # fmt: skip
        if problem:
            return f"slot:{problem}", None
        try:
            invalid = await validate(filled)
        except Exception as exc:
            return f"validation_error:{type(exc).__name__}", None
        if invalid:
            return f"validation:{invalid}", None
        return "passed", filled

    def _miss(self, res: LookupResult, reason: str) -> LookupResult:
        self.stats.misses += 1
        res.reason = reason
        return res

    def record(self, raw: dict, view: QuestionView, compat: Compatibility, source: str) -> str | None:
        """Queue a checked, executed plan as a reusable template. Returns the reason when it is not eligible."""
        if not self.store_templates:
            return "storing_disabled"
        try:
            plan, slots = templatize(raw, view)
        except Ineligible as exc:
            self.stats.ineligible += 1
            return f"ineligible:{exc}"
        found = mentions(view) or []
        text = slot_text(view, found)
        if re.search(r"\d", _PLACEHOLDER_IN_TEXT.sub("", text)):  # a number survived slotting: never persist it
            self.stats.ineligible += 1
            return "ineligible:unslotted number in question text"
        doc = {
            "doc_version": TEMPLATE_DOC_VERSION,
            "question_text": text,
            "logic_signature": logic_signature(text),
            "plan_template": plan,
            "slots": slots,
            "mention_counts": _counts(found),
            "action": plan.get("action"),
            "source": source,
            "compat_key": compat.key,
            "request_ir_version": compat.request_ir_version,
            "vocabulary_hash": compat.vocabulary_hash,
            "schema_hash": compat.schema_hash,
            "model_version": compat.model_version,
            "embedding_model": compat.embedding_model,
        }
        doc["_id"] = hashlib.sha256(_canon({k: doc[k] for k in ("question_text", "plan_template", "compat_key")}).encode()).hexdigest()[:24]
        doc["template_id"] = doc["_id"]

        async def write():
            # off the request path (the answer is already sent): allow a slow first connection
            doc["embedding"] = await asyncio.wait_for(self._embed(text), WRITE_TIMEOUT_S)
            await asyncio.wait_for(self.store.store(doc), WRITE_TIMEOUT_S)
            self.stats.recorded += 1
            log.info("plan template stored", extra={"template_id": doc["_id"], "source": source,
                                                                 "action": doc["action"], "slots": len(slots)})  # fmt: skip

        self._background(write())
        return None

    def _background(self, coro) -> None:
        async def guarded():
            try:
                await coro
            except Exception as exc:
                log.warning("semantic cache write failed: %s", type(exc).__name__)

        task = asyncio.create_task(guarded())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        """Wait for queued writes (tests, shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
