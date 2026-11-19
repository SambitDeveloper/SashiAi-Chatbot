"""Question brief: what the deterministic analysis understood, handed to the local model when the fast path cannot
answer (LLM_QUESTION_BRIEF, off by default: it adds prompt tokens, see config.py).

The fast path needs every word explained before it answers by itself. When it gives up, it usually understood most
of the question: the bge-small router knows the kind of question, the resident names are masked, attribute words
("room", "caregiver") and database values ("discharged", a facility) are recognised. Before this brief, the model got
only the bare question and had to work all of that out again. Now it also gets:

    Analysis (automatic, may be wrong; the question decides):
    - likely action: lookup (0.89)
    - residents named: resident_1
    - asked about: discharge_date, status
    - values mentioned: status discharged
    - yes/no question: lookup with the mentioned values as conditions (the database checks them)
    - words not recognised: already

The brief is built from configuration and the live vocabulary only (no question-specific rules) and never contains
resident names: residents appear as their resident_N placeholders, and unrecognised words are taken from the masked
question. It is not logged.
"""

from dataclasses import dataclass, field

# A second route is shown only when the router could not separate it from the first.
CLOSE_MARGIN = 0.05


@dataclass
class RouteGuess:
    name: str
    score: float
    metadata: dict


@dataclass
class QuestionBrief:
    routes: list[RouteGuess] = field(default_factory=list)  # best first; a second one only when it is close
    residents: list[str] = field(default_factory=list)  # resident_N placeholders
    ambiguous_residents: list[str] = field(default_factory=list)  # placeholders matching several residents
    attributes: list[str] = field(default_factory=list)  # resident attributes the question asks about
    values: dict[str, list[str]] = field(default_factory=dict)  # field → database values written in the question
    unknown_words: list[str] = field(default_factory=list)
    yes_no: bool = False  # "is/are/does/has …" question about named residents that mentions database values

    @property
    def empty(self) -> bool:
        return not (self.routes or self.residents or self.attributes or self.values or self.unknown_words)

    def to_text(self) -> str | None:
        if self.empty:
            return None
        lines = ["Analysis (automatic, may be wrong; the question decides):"]
        if self.routes:
            best = self.routes[0]
            guess = f"- likely action: {describe_route(best.metadata)} ({best.score:.2f})"
            if len(self.routes) > 1:
                second = self.routes[1]
                guess += f"; or {describe_route(second.metadata)} ({second.score:.2f})"
            lines.append(guess)
        if self.residents:
            lines.append(f"- residents named: {', '.join(self.residents)}")
        elif any(r.metadata.get("action") == "lookup" for r in self.routes):
            # without this the model tends to invent resident_1 for a name the records do not contain
            lines.append("- residents named: none found in the records (do not invent resident_N)")
        if self.ambiguous_residents:
            lines.append(f"- matches several residents: {', '.join(self.ambiguous_residents)}")
        if self.attributes:
            lines.append(f"- asked about: {', '.join(self.attributes)}")
        if self.values:
            lines.append(
                "- values mentioned: " + "; ".join(f"{f} {', '.join(vs)}" for f, vs in self.values.items())
            )
        if self.yes_no and self.residents and self.values:
            # the executor answers "Yes — …/No — …" for a lookup with conditions; without this line the model shows
            # the attribute instead and the plan is rejected for dropping the value the question asked about
            lines.append("- yes/no question: lookup with the mentioned values as conditions (the database checks them)")
        if self.unknown_words:
            lines.append(f"- words not recognised: {', '.join(self.unknown_words)}")
        return "\n".join(lines)


def describe_route(meta: dict) -> str:
    """A route's plan shape in the plan's own terms: "breakdown group_by status", "list sort age desc limit 1"."""
    text = str(meta.get("action", "unknown"))
    if meta.get("op"):
        text += f" {meta['op']} age"
    if meta.get("group_by"):
        text += " group_by " + ", ".join(meta["group_by"])
    sort = meta.get("sort")
    if sort:
        text += f" sort {sort.get('field')} {sort.get('direction', '')}".rstrip()
    if meta.get("limit"):
        text += f" limit {meta['limit']}"
    return text
