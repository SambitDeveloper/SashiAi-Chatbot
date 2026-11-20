"""Reference resolution: does a pronoun refer within the current question or to the previous turn?

Shared by the deterministic planner (to decide whether it can answer alone) and the grounding check (to decide
whether the previous plan's conditions must be kept). Built from closed grammatical classes in intent_routes.yaml
(`direct.references`): pronouns, demonstratives, turn openers and the domain's entity nouns.

Rules, applied to the masked question (resident names already replaced by resident_N):
  • A pronoun with an entity antecedent earlier in the question refers within it:
    "list active residents and show THEIR rooms" → local.
  • A pronoun without one refers to the previous turn: "how many of THEM are female?".
  • A demonstrative before an entity noun points back: "how many of THESE residents …" (previous turn, when there
    is one; on a fresh conversation it is read as part of the question).
  • A question opening with a continuation word continues the previous turn: "what about their rooms?".
  • A reference to the previous turn when there is none is ambiguous: ask, never guess.
"""

import re
from dataclasses import dataclass, field

from app.nlu import placeholders as ph

_WORD = re.compile(ph.TOKEN + r"|res[-_ ]?\d{3,6}|[a-z]+(?:'[a-z]+)?")
_PLACEHOLDER = re.compile(r"^(?:resident_\d+|res[-_ ]?\d{3,6})$")


@dataclass
class ReferenceResolution:
    refers_to_previous: bool = False  # the question builds on the previous turn
    local_spans: list[tuple[int, int]] = field(default_factory=list)  # pronouns resolved within the question
    trigger: str = ""  # the word that points back ("them", "these residents", "what about")

    def ambiguous(self, has_previous: bool) -> bool:
        """Points back to a previous turn that does not exist: the user must say whom they mean."""
        return self.refers_to_previous and not has_previous


class ReferenceResolver:
    def __init__(self, routes_config: dict):
        cfg = (routes_config.get("direct") or {}).get("references") or {}
        self.pronouns = {str(w).lower() for w in cfg.get("pronouns") or []}
        self.demonstratives = {str(w).lower() for w in cfg.get("demonstratives") or []}
        self.entity_nouns = {str(w).lower() for w in cfg.get("entity_nouns") or []}
        openers = sorted((" ".join(str(w).lower().split()) for w in cfg.get("continuation_openers") or []), key=len)
        self._opener = (
            re.compile(r"^\W*(?:" + "|".join(r"\s+".join(map(re.escape, o.split())) for o in reversed(openers)) + r")(?![a-z])")
            if openers
            else None
        )

    def _is_entity(self, word: str) -> bool:
        return word in self.entity_nouns or bool(_PLACEHOLDER.match(word))

    def resolve(self, masked_question: str, has_previous: bool) -> ReferenceResolution:
        low = masked_question.lower()
        words = [(m.group(0), m.start(), m.end()) for m in _WORD.finditer(low)]
        result = ReferenceResolution()
        antecedent_seen = False
        for i, (w, s, e) in enumerate(words):
            nxt = words[i + 1][0] if i + 1 < len(words) else ""
            if w in self.demonstratives and self._is_entity(nxt):
                if has_previous:  # "these residents" → the previous answer's residents
                    result.refers_to_previous, result.trigger = True, f"{w} {nxt}"
                # on a fresh conversation the noun phrase is just part of this question
                continue
            if w in self.pronouns:
                if antecedent_seen:
                    result.local_spans.append((s, e))
                else:
                    result.refers_to_previous = True
                    result.trigger = result.trigger or w
                continue
            if self._is_entity(w):
                antecedent_seen = True
        if self._opener is not None and has_previous:
            m = self._opener.match(low)
            if m:
                result.refers_to_previous = True
                result.trigger = result.trigger or m.group(0).strip(" \t,.?!")
        return result
