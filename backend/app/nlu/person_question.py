"""Simple questions about ONE named person, answered by the fast path (no model): "Who is Jone?", "What is Jone's
designation?", "Is Jone active?", "Which facility does Jone belong to?", "What are Jone's assigned residents?".

Pure grammar, configured in app/data/intent_routes.yaml (`direct.person_questions`) and `direct.lookup_attributes`:
every word of the question must be explained by a filler word, a kind word ("staff member"), an attribute phrase
("designation", "room number") or a check word ("active") — except ONE run of 1-4 words, which is the name. Anything
else (a second name, a number, a condition, "how many …") is not such a question and the normal pipeline answers it.
Whether the name is a real person, and which kind, is decided by the entity index (EntityIndex.people_named).

A trailing "(…)" is a detail that tells same-named people apart ("Jone (Transport Driver)"); clarification options
write it, and the service narrows the candidates with it.
"""

import re
from dataclasses import dataclass, field

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’.\-]*")
_DETAIL = re.compile(r"\s*\(([^()]{1,80})\)")

# which kinds of person an attribute can be about (resident-only attributes come from lookup_attributes)
STAFF_ONLY = "staff"
SHARED = "shared"
FAMILY_ONLY = "family"
RESIDENT_ONLY = "resident"


@dataclass
class PersonQuestion:
    name: str  # as the user wrote it
    kind: str | None = None  # resident / staff / family when the question says so
    detail: str | None = None  # "(Transport Driver)": picks one of several same-named people
    attributes: list[tuple[str, str]] = field(default_factory=list)  # (attribute, scope) in question order
    check: str | None = None  # "active" / "inactive": a yes/no question
    name_span: tuple[int, int] = (0, 0)  # in `text`
    kind_span: tuple[int, int] | None = None  # the kind words just before the name, in `text`
    text: str = ""  # the question without the detail

    @property
    def kinds(self) -> set[str]:
        """The kinds of person the question can be about, from its attributes and check (all when it has none)."""
        allowed = {"resident", "staff", "family"}
        for _, scope in self.attributes:
            allowed &= {RESIDENT_ONLY: {"resident"}, STAFF_ONLY: {"staff"}, FAMILY_ONLY: {"family"},
                        SHARED: {"resident", "staff"}}[scope]  # fmt: skip
        if self.check:
            allowed &= {"resident", "staff"}
        if self.kind:
            allowed &= {self.kind}
        return allowed


def _norm(token: str) -> str:
    t = token.lower().replace("’", "'").rstrip(".")
    return re.sub(r"'s$", "", t).strip("'")


class PersonQuestionParser:
    def __init__(self, config: dict | None, lookup_attributes: dict | None = None):
        cfg = config or {}
        self.max_options = int(cfg.get("max_options", 6))
        self.kind_words: dict[str, str] = {}
        phrases: list[tuple[tuple[str, ...], str, str]] = []  # (words, role, value)

        def add(phrase: str, role: str, value: str) -> None:
            words = tuple(_norm(w) for w in _WORD.findall(str(phrase)))
            if words:
                phrases.append((words, role, value))

        for w in cfg.get("fillers") or []:
            add(w, "filler", "")
        for kind, words in (cfg.get("kinds") or {}).items():
            for w in words or []:
                add(w, "kind", kind)
            self.kind_words.setdefault(kind, str((words or [kind])[0]))
        for w, value in (cfg.get("checks") or {}).items():
            add(w, "check", str(value))
        scoped = {}
        for attr, words in (lookup_attributes or {}).items():  # resident attributes (the resident fast path's words)
            scoped[attr] = (RESIDENT_ONLY, words)
        for scope, key in ((STAFF_ONLY, "staff_attributes"), (SHARED, "shared_attributes"), (FAMILY_ONLY, "family_attributes")):
            for attr, words in (cfg.get(key) or {}).items():
                scoped[attr] = (scope, [*(scoped.get(attr, (None, []))[1] if scope == SHARED else []), *(words or [])])
        for attr, (scope, words) in scoped.items():
            for w in words or []:
                add(w, f"attr:{scope}", attr)
        self.phrases = sorted(phrases, key=lambda p: -len(p[0]))
        self.openers = {tuple(_norm(w) for w in _WORD.findall(str(o))) for o in cfg.get("openers") or []}

    def parse(self, question: str) -> PersonQuestion | None:
        m = _DETAIL.search(question)
        detail = m.group(1).strip() if m else None
        text = (question[: m.start()] + question[m.end() :]) if m else question
        tokens = [(t.group(0), t.start(), t.end()) for t in _WORD.finditer(text)]
        if not tokens:
            return None
        norms = [_norm(t[0]) for t in tokens]
        roles: list[tuple[str, str] | None] = [None] * len(tokens)
        i = 0
        while i < len(tokens):
            for words, role, value in self.phrases:
                if tuple(norms[i : i + len(words)]) == words:
                    for j in range(i, i + len(words)):
                        roles[j] = (role, value)
                    i += len(words)
                    break
            else:
                i += 1
        free = [j for j, r in enumerate(roles) if r is None]
        if not free or len(free) > 4 or free != list(range(free[0], free[-1] + 1)):
            return None  # nothing, or more than one name-like run: not a simple person question
        if not any(r and r[0] in ("filler", "kind") for r in roles[: free[0]]) and free[0] != 0:
            return None
        start, end = tokens[free[0]][1], tokens[free[-1]][2]
        name = re.sub(r"(?:'s|’s|')$", "", text[start:end]).strip().rstrip(".")
        if not name or any(ch.isdigit() for ch in name):
            return None
        kind, kind_span = None, None
        if free[0] > 0 and roles[free[0] - 1] and roles[free[0] - 1][0] == "kind":
            k = free[0] - 1
            value = roles[k][1]
            while k > 0 and roles[k - 1] == ("kind", value):
                k -= 1
            kind, kind_span = value, (tokens[k][1], tokens[free[0] - 1][2])
        # a kind word anywhere else ("Which RESIDENT is Margaret related to?") asks for records of that kind, not about
        # the named person: not a simple person question
        kind_idx = {j for j in range(len(tokens)) if kind_span and kind_span[0] <= tokens[j][1] < kind_span[1]}
        if any(r and r[0] == "kind" and j not in kind_idx for j, r in enumerate(roles)):
            return None
        attributes = [(r[1], r[0].split(":")[1]) for r in roles if r and r[0].startswith("attr:")]
        checks = [r[1] for r in roles if r and r[0] == "check"]
        return PersonQuestion(
            name=name, kind=kind, detail=detail, attributes=list(dict.fromkeys(attributes)),
            check=checks[0] if checks else None, name_span=(start, end), kind_span=kind_span, text=text,
        )  # fmt: skip
