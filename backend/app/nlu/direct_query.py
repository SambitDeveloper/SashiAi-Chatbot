"""Direct answers without the language model: semantic-router route + database values → query plan.

Everything here is driven by configuration and data:
  • which routes may answer directly, with which action / grouping / sort / allowed fields → intent_routes.yaml
  • the confidence rule, route cue words, blocker words and filler words → intent_routes.yaml (`direct:` section)
  • the values recognised in the question (statuses, care types, genders, wings, caregivers, residents) → the database

A plan is returned only when every rule passes; otherwise a reason is returned and the question goes to the
language model. The plan then runs through the same validation and the MongoDB MCP tools as any other plan.
"""

import re
from dataclasses import dataclass, field
from datetime import date

from rapidfuzz import fuzz

from app.nlu.grammar import parse_dates, parse_number_ranges
from app.nlu.intent_router import IntentScore
from app.nlu.missing_values import MissingValueDetector
from app.nlu.projection import ProjectionDetector, value_starts
from app.nlu.references import ReferenceResolver
from app.nlu.vocabulary import EntityIndex, MaskResult

_TOKEN_RE = re.compile(r"resident_\d+|[a-z]+|\d+")
_WH_QUESTION = re.compile(r"^\s*(?:when|what|which|who|whom|whose|where|how|tell|show|give|list|find)\b")
_RESIDENT_ID_RE = re.compile(r"\bres[-_ ]?\d{3,6}\b")
_GENDERS = {"female": "Female", "male": "Male"}


def _phrase_re(phrase: str) -> re.Pattern:
    return re.compile(r"(?<![a-z])" + r"\s+".join(map(re.escape, phrase.lower().split())) + r"(?![a-z])")


@dataclass
class DirectResult:
    plan: dict | None
    reason: str = ""  # why the question goes to the language model
    details: dict = field(default_factory=dict)


class DirectQueryBuilder:
    def __init__(self, routes_config: dict):
        direct = routes_config.get("direct", {})
        self.min_confidence = float(direct.get("min_confidence", 0.80))
        # When a cue word already fixes the kind of question and every token is explained, the embedding score is
        # largely redundant, so a lower bar is safe there (nonsense is still caught by the full-coverage check).
        self.min_cue_confidence = float(direct.get("min_cue_confidence", self.min_confidence))
        self.min_margin = float(direct.get("min_margin", 0.03))
        self.blockers = {str(w).lower() for w in direct.get("blocker_words", [])}
        self.filler = {str(w).lower() for w in direct.get("filler_words", [])}
        self.field_words = {str(w).lower() for w in direct.get("field_words", [])}
        # Date / number anchor words → the schema field a parsed range is attached to (data-driven, no hardcoding).
        self.date_fields: dict[str, list[str]] = {
            f: [str(w).lower() for w in words] for f, words in (direct.get("date_fields") or {}).items()
        }
        self.number_fields: dict[str, list[str]] = {
            f: [str(w).lower() for w in words] for f, words in (direct.get("number_fields") or {}).items()
        }
        self.default_number_field: str | None = direct.get("default_number_field")
        self.negation_words = {str(w).lower() for w in direct.get("negation_words", [])}
        # "No value recorded" phrases → `missing` conditions (intent_routes.yaml `missing_values`).
        self.missing_detector = MissingValueDetector(routes_config)
        # Pronouns: within this question, or a reference to the previous turn (intent_routes.yaml `references`).
        self.references = ReferenceResolver(routes_config)
        # What to SHOW (projection) vs what to filter by, from each phrase's role (intent_routes.yaml `projection`).
        self.projection = ProjectionDetector(routes_config)
        # Lookup attribute words → the resident attribute to answer (intent_routes.yaml `lookup_attributes`).
        self.lookup_attributes: list[tuple[str, str, re.Pattern]] = [
            (attr, str(w).lower(), _phrase_re(str(w)))
            for attr, words in (direct.get("lookup_attributes") or {}).items()
            for w in words
        ]
        self.routes = {r["name"]: r for r in routes_config.get("routes", [])}
        self.cues = [
            (list(group["routes"]), [_phrase_re(str(w)) for w in group["words"]])
            for group in direct.get("route_cues", [])
        ]

    def choose_route(self, question: str, mask: MaskResult, intent: IntentScore) -> tuple[str, float, float, str]:
        """(route, confidence, margin, how) — the question's own words decide the kind of question.

        Embeddings score "how many residents are away" and "list residents who are away" almost the same, so the
        router alone often cannot tell count from list (low margin, or the wrong one first). The first `route_cues`
        group whose words appear in the question (a named resident → the routes that accept a resident) limits the
        candidates; the router then picks among them and the margin is measured only between those candidates.
        Without cue words, or when the router has no score for them, the router's own top route is used.
        """
        low = question.lower()
        candidates: list[str] = []
        how = "router"
        if mask.placeholders or _RESIDENT_ID_RE.search(low):
            candidates = [
                n for n, r in self.routes.items() if "resident" in (r.get("metadata") or {}).get("fields", [])
            ]
            how = "cue:resident"
        else:
            for routes, patterns in self.cues:
                if any(p.search(low) for p in patterns):
                    candidates, how = routes, "cue:" + routes[0]
                    break
        scored = sorted(((intent.scores[n], n) for n in candidates if n in intent.scores), reverse=True)
        if not scored:
            # No cue word: the router chooses. Counting and grouping need an explicit cue ("how many", "per"),
            # so when the router's own top route is a plain list, a close count runner-up is not a real ambiguity —
            # a bare "residents <condition>" statement is a list by default.
            ranked = sorted(intent.scores.items(), key=lambda kv: kv[1], reverse=True)
            if ranked and ranked[0][0] == "list":
                return "list", ranked[0][1], 1.0, "router:list-default"
            return intent.route, intent.confidence, intent.margin, "router"
        best_score, best = scored[0]
        margin = round(best_score - scored[1][0], 4) if len(scored) > 1 else 1.0
        return best, best_score, margin, how

    def build(
        self,
        question: str,
        mask: MaskResult,
        intent: IntentScore | None,
        index: EntityIndex,
        today: date,
        has_previous: bool = False,
    ) -> DirectResult:
        if intent is None:
            return DirectResult(None, "no_router_score")
        name, confidence, margin, how = self.choose_route(question, mask, intent)
        route = self.routes.get(name, {})
        meta = route.get("metadata") or {}
        details = {"route": name, "confidence": confidence, "margin": margin, "chosen_by": how}
        if "action" not in meta:
            return DirectResult(None, "route_not_direct", details)
        threshold = self.min_cue_confidence if how.startswith("cue:") else self.min_confidence
        if confidence < threshold:
            return DirectResult(None, "low_confidence", details)
        if margin < self.min_margin:
            return DirectResult(None, "low_margin", details)

        low = question.lower()
        covered = [False] * len(low)

        def cover(start: int, end: int) -> None:
            for i in range(start, end):
                covered[i] = True

        def span_of(text: str) -> tuple[int, int] | None:
            m = _phrase_re(text).search(low)
            return m.span() if m else None

        def locate_uncovered(text: str) -> tuple[int, int] | None:
            """First occurrence of a value whose characters are all still unexplained."""
            for m in _phrase_re(text).finditer(low):
                if not any(covered[m.start() : m.end()]):
                    return m.span()
            return None

        def find_and_cover(text: str) -> bool:
            sp = span_of(text)
            if sp is None:
                return False
            cover(*sp)
            return True

        # Negation turns the next value into an exclusion. A value is negated when the nearest preceding
        # negation word is separated from it only by filler ("not in memory care", "active but not discharged").
        neg_spans: list[tuple[int, int]] = []
        for word in self.negation_words:
            neg_spans += [m.span() for m in _phrase_re(word).finditer(low)]

        def negation_before(value_start: int) -> tuple[int, int] | None:
            best: tuple[int, int] | None = None
            for ns, ne in neg_spans:
                if ne <= value_start and all(t in self.filler for t in _TOKEN_RE.findall(low[ne:value_start])):
                    if best is None or ne > best[1]:
                        best = (ns, ne)
            return best

        include_values: dict[str, list[str]] = {}
        exclude_values: dict[str, list[str]] = {}
        range_filters: dict[str, dict] = {}

        def add_value(field_name: str, values, value_start: int) -> None:
            vals = values if isinstance(values, list) else [values]
            neg = negation_before(value_start)
            (exclude_values if neg else include_values).setdefault(field_name, []).extend(vals)
            if neg:
                cover(*neg)

        # 1. residents (by placeholder or id) ------------------------------------------------------
        residents: list[str] = []
        for key, ids in mask.placeholders.items():
            if len(ids) != 1:
                return DirectResult(None, "ambiguous_resident", details)
            residents.extend(ids)
            if not find_and_cover(mask.originals.get(key, "")):
                return DirectResult(None, "value_span_unclear", details)
        for m in _RESIDENT_ID_RE.finditer(low):
            rid = "RES-" + re.sub(r"\D", "", m.group(0))
            if rid not in index.display_by_id:
                return DirectResult(None, "unknown_resident_id", details)
            residents.append(rid)
            cover(m.start(), m.end())
        in_resident_name = covered.copy()  # only resident names/ids are covered at this point

        # 1a. pronouns: a reference to the previous turn needs the conversation (the model gets the previous plan);
        #     a pronoun whose antecedent is in this question ("residents … their rooms") is understood here.
        references = self.references.resolve(mask.text, has_previous)
        if references.refers_to_previous:
            return DirectResult(None, "refers_to_previous", {**details, "trigger": references.trigger})
        for s, e in references.local_spans:
            pronoun = mask.text.lower()[s:e]
            for m in _phrase_re(pronoun).finditer(low):
                if not any(in_resident_name[m.start() : m.end()]):
                    cover(*m.span())
        attributes: list[str] = []
        cue_words: set[str] = set()
        if meta["action"] == "lookup":
            attributes, cue_words, typo_spans = self._lookup_attributes(low, in_resident_name)
            for span in typo_spans:  # a recognised attribute word with a small typo ("staus") is understood
                cover(*span)

        # 1b. "no value recorded" conditions ("no caregiver assigned", "room not recorded") -------------------
        missing_scan = self.missing_detector.scan(low, in_resident_name)
        if missing_scan.or_across_fields:  # "no caregiver OR no room" cannot be expressed; never turn it into AND
            return DirectResult(None, "missing_or_across_fields", details)
        if missing_scan.conflicting:
            return DirectResult(None, "missing_conflict", details)
        missing_flags: dict[str, bool] = missing_scan.by_field
        for mention in missing_scan.mentions:
            cover(*mention.span)
        # "no"/"without" inside such a phrase belongs to it, not to a later value ("without a room in memory care")
        neg_spans[:] = [n for n in neg_spans if not any(covered[n[0] : n[1]])]

        # 2. date and numeric ranges (generic grammar) ---------------------------------------------
        #    A date expression is attached to the field whose anchor word is present; the anchor word is then
        #    covered so a word that is also a status value (e.g. "admitted", "discharged") is not double-counted.
        date_matches = parse_dates(low, today)
        if date_matches:
            if len(date_matches) > 1:
                return DirectResult(None, "multiple_date_ranges", details)
            # every occurrence of every anchor word of a field ("admission date", "admitted"), longest phrase first
            anchors: dict[str, list[tuple[int, int]]] = {}
            for f, words in self.date_fields.items():
                for w in sorted(words, key=len, reverse=True):
                    for m in _phrase_re(w).finditer(low):
                        if not any(a <= m.start() < b for a, b in anchors.get(f, [])):
                            anchors.setdefault(f, []).append(m.span())
            if not anchors:
                return DirectResult(None, "date_without_anchor", details)
            if len(anchors) > 1:
                return DirectResult(None, "ambiguous_date_field", details)
            field_name, anchor_spans = next(iter(anchors.items()))
            range_filters[field_name] = dict(date_matches[0].range)
            cover(date_matches[0].start, date_matches[0].end)
            for sp in anchor_spans:
                cover(*sp)

        number_matches = [n for n in parse_number_ranges(low) if not any(covered[n.start : n.end])]
        if number_matches:
            field_name = self.default_number_field
            for f, words in self.number_fields.items():
                if any(span_of(w) for w in words):
                    field_name = f
                    break
            if field_name is None:
                return DirectResult(None, "number_without_field", details)
            bounds: dict = {}
            for n in number_matches:
                bounds.update(n.bounds)
                cover(n.start, n.end)
            range_filters[field_name] = bounds
            for w in self.number_fields.get(field_name, []):
                sp = span_of(w)
                if sp:
                    cover(*sp)

        # 3. fixed values (statuses, care types, genders) first, then caregivers and wings ----------
        for status in index.statuses:  # only statuses that exist in the database
            sp = locate_uncovered(status)
            if sp:
                add_value("status", status, sp[0])
                cover(*sp)
        for care in index.care_types:
            sp = locate_uncovered(care.replace("_", " "))
            if sp:
                add_value("care_type", care, sp[0])
                cover(*sp)
        for word, gender in _GENDERS.items():
            sp = locate_uncovered(word)
            if sp:
                add_value("gender", gender, sp[0])
                cover(*sp)

        def inside_known_value(text: str) -> bool:
            m = _phrase_re(text).search(low)
            return bool(m) and all(covered[m.start() : m.end()])

        for text, caregivers in index.mentioned_caregivers(question).items():
            if inside_known_value(text):
                continue
            sp = locate_uncovered(text)
            if sp is None:
                return DirectResult(None, "value_span_unclear", details)
            add_value("primary_caregiver", caregivers, sp[0])
            cover(*sp)
        for text, wings in index.mentioned_wings(question).items():
            if inside_known_value(text):
                continue
            sp = locate_uncovered(text)
            if sp is None:
                return DirectResult(None, "value_span_unclear", details)  # matched only with a typo: let the model decide
            if len(wings) > 1:
                return DirectResult(None, "ambiguous_wing", details)
            add_value("facility", wings, sp[0])
            cover(*sp)
        # rooms: a room value from the live vocabulary, bound to a room word ("room MC-101", "unit 12")
        room_words = {w for attr, w, _ in self.lookup_attributes if attr == "room_number"}
        for m in re.finditer(r"(?<![a-z0-9])[a-z]{0,3}-?\d[\w-]*", low):
            if any(covered[m.start() : m.end()]):
                continue
            head = re.search(r"([a-z]+(?:\s+[a-z]+)?)\s+(?:no\.?\s+|#\s*)?$", low[: m.start()])
            word = next((w for w in sorted(room_words, key=len, reverse=True) if head and head.group(1).endswith(w)), None)
            if word is None:
                continue
            room = index.resolve_room(m.group(0))
            if room.ambiguous:
                return DirectResult(None, "ambiguous_room", details)
            if not room.values:
                return DirectResult(None, "unknown_room", details)
            add_value("room_number", room.values[0], m.start())
            cover(m.start(), m.end())
            cover(head.end(1) - len(word), head.end(1))

        # 3b. projection: attributes the question asks to SEE ("… and show their room numbers") -------------------
        fields: list[str] = []
        if meta["action"] == "list":
            not_attribute = in_resident_name.copy()
            for mention in missing_scan.mentions:  # "no caregiver" is a condition, never a column
                for i in range(*mention.span):
                    not_attribute[i] = True
            for p in self.projection.scan(low, not_attribute, value_starts(question, index, today)):
                if p.attribute is not None:  # identity words ("name") are always shown: no extra field
                    fields.append(p.attribute)
                cover(*p.span)
                for c in p.cue_spans:
                    cover(*c)

        allowed = set(meta.get("fields", []))
        produced = set(include_values) | set(exclude_values) | set(range_filters) | set(missing_flags)
        for field_name in produced:
            if field_name not in allowed:
                return DirectResult(None, f"field_not_allowed:{field_name}", details)
        if residents and "resident" not in allowed:
            return DirectResult(None, "field_not_allowed:resident", details)

        # 4. every remaining word must be explained; a leftover number is never guessed at -----------
        value_words = self._value_words(index) - self.field_words
        route_words = {w for u in route.get("utterances", []) for w in _TOKEN_RE.findall(u.lower())}
        if meta["action"] == "lookup":  # attribute words ("admission", "room number") are understood for lookups
            route_words |= {w for _, phrase, _ in self.lookup_attributes for w in _TOKEN_RE.findall(phrase)}
        route_words -= value_words  # a database value word must still be recognised as that value
        uncovered = []
        for m in _TOKEN_RE.finditer(low):
            word = m.group(0)
            if any(covered[m.start() : m.end()]):
                continue
            if word in self.blockers:
                return DirectResult(None, f"blocked_word:{word}", details)
            if word.isdigit():
                return DirectResult(None, "has_number", details)
            if word not in self.filler and word not in route_words:
                uncovered.append(word)
        if uncovered:
            return DirectResult(None, "not_understood", {**details, "words": uncovered})

        # 5. plan from the route's metadata plus the resolved conditions -----------------------------
        def build_filters() -> dict:
            filters: dict = {k: dict(v) for k, v in (meta.get("filters") or {}).items()}
            for f, vals in include_values.items():
                filters.setdefault(f, {})["include"] = list(dict.fromkeys(vals))
            for f, vals in exclude_values.items():
                filters.setdefault(f, {})["exclude"] = list(dict.fromkeys(vals))
            for f, rng in range_filters.items():
                filters[f] = dict(rng)
            for f, flag in missing_flags.items():  # within a field: include OR missing (see QueryPlan SetFilter)
                filters.setdefault(f, {})["missing"] = flag
            return filters

        action = meta["action"]
        has_conditions = bool(include_values or exclude_values or range_filters or missing_flags)
        if action == "help":
            if has_conditions or residents:
                return DirectResult(None, "help_with_values", details)
            return DirectResult({"action": "help"}, "", details)
        if action == "lookup":
            if not residents:
                return DirectResult(None, "missing_resident", details)
            # Answer only what was asked ("room number of resident_1" → room_number); none named → the full profile.
            if attributes and _WH_QUESTION.match(low):
                # "when was resident_1 admitted" asks for the date, not the yes/no check "is resident_1 admitted?"
                for value in list(include_values.get("status", [])):
                    if value in cue_words:
                        include_values["status"].remove(value)
                if "status" in include_values and not include_values["status"]:
                    del include_values["status"]
            plan: dict = {
                "action": "lookup",
                "lookup": {"residents": list(dict.fromkeys(residents)), "attributes": attributes},
            }
            filters = build_filters()
            if filters:  # a condition on one resident is a yes/no check ("is resident_1 active?")
                plan["filters"] = filters
            return DirectResult(plan, "", details)

        plan = {"action": action, "filters": build_filters()}
        if fields:
            plan["fields"] = list(dict.fromkeys(fields))
        if meta.get("group_by"):
            plan["group_by"] = list(meta["group_by"])
        if action == "aggregate":
            plan["aggregate"] = {"op": meta.get("op", "avg"), "field": "age"}
        if meta.get("sort"):
            plan["sort"] = dict(meta["sort"])
        if meta.get("limit"):
            plan["limit"] = int(meta["limit"])
        return DirectResult(plan, "", details)

    def _lookup_attributes(
        self, low: str, in_resident_name: list[bool]
    ) -> tuple[list[str], set[str], list[tuple[int, int]]]:
        """Attributes named in the question (in order of appearance), the words that named them, typo-matched spans.

        Words inside a resident's name never count ("Carter" is not "carer"). Single words of 5+ letters may carry a
        small typo ("staus" → status).
        """
        found: dict[str, int] = {}
        words: set[str] = set()
        typo_spans: list[tuple[int, int]] = []
        for attr, phrase, pattern in self.lookup_attributes:
            for m in pattern.finditer(low):
                if any(in_resident_name[m.start() : m.end()]):
                    continue
                found[attr] = min(found.get(attr, m.start()), m.start())
                words.add(phrase)
        for m in re.finditer(r"[a-z]{5,}", low):
            if any(in_resident_name[m.start() : m.end()]):
                continue
            for attr, phrase, _ in self.lookup_attributes:
                if " " not in phrase and len(phrase) >= 5 and attr not in found and fuzz.ratio(m.group(0), phrase) >= 85:
                    found[attr] = m.start()
                    words.add(phrase)
                    if m.group(0) != phrase:
                        typo_spans.append(m.span())
        return sorted(found, key=found.get), words, typo_spans

    @staticmethod
    def _value_words(index: EntityIndex) -> set[str]:
        """Words that identify ONE specific database value ("north", "rostova", "memory", "active").

        They never count as explained just because they appear in a route's examples: they must be recognised as
        a value, or the condition would be silently dropped. Words shared by several values of a field ("wing",
        "nurse", "care") identify nothing on their own, so a route's examples may explain them.
        """
        words = set(index.statuses) | set(_GENDERS)
        groups = [[c.replace("_", " ") for c in index.care_types], list(index.wings), index.caregiver_values()]
        for group in groups:
            seen: dict[str, int] = {}
            for value in group:
                for w in set(_TOKEN_RE.findall(value.lower())):
                    seen[w] = seen.get(w, 0) + 1
            words |= {w for w, n in seen.items() if n == 1}
        return words
