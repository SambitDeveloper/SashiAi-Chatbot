"""Optional answer rephrasing by the local Qwen3 model, with a fact check (LLM_RESPONSE_PHRASING, off by default).

The template answer (responder.py) stays the source of truth. When phrasing is enabled, the local model rewrites
only `summaryText` so it reads naturally and in the style the question asked for ("briefly", "explain", one value …).
The rewrite is shown only if it passes a fact check against the template answer:

  • every number in it (counts, ages, rooms, dates, percentages) appears in the template facts;
  • every capitalised word (names, care types, facilities, months) appears in the facts or the question;
  • no gendered pronouns (the records' gender field is not a pronoun preference);
  • a yes/no check keeps its polarity, and an empty result still says so;
  • plain prose: no markdown, no placeholders, a sensible length.

Anything else (failure, timeout, busy model, a fact-check miss) keeps the template answer, so enabling this never
makes an answer less correct than today. The model runs on the private llama-server; nothing leaves this server.
The model's text is never logged, only the outcome and a reason code.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from app.engine.describe import describe_filters
from app.models.plan import QueryPlan
from app.nlu.translator import TranslatorUnavailable

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You rewrite answers from a care home's resident database for the staff member who asked. "
    "Rewrite the ANSWER FACTS as a clear, friendly reply to the QUESTION, in the style the question asks for: "
    "short if they want it short, a little more context if they ask to explain.\n"
    "Rules:\n"
    "- Always write at least one complete sentence that says what each number or value refers to. The bare value "
    "is already shown as a headline above your reply.\n"
    "- Use only the facts given. Never add, calculate, round or change a number, name, date, room or label: copy "
    "them exactly as written.\n"
    "- Write numbers as digits.\n"
    "- Refer to residents by name. Never use he, she, him, her, his or hers.\n"
    "- Plain text only: no markdown, no bullet points, no line breaks. At most 3 sentences.\n"
    "- Only if the facts say a table is shown below, you may say the residents are listed below; never repeat it. "
    "Otherwise do not mention a list or table.\n"
    "Reply with the rewritten answer only."
)

_ACTIONS = {"count", "list", "breakdown", "aggregate", "lookup"}
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_CAPITALISED = re.compile(r"\b[A-Z][A-Za-z'’-]*")
_WORD = re.compile(r"[a-z0-9'’-]+")
_GENDERED = re.compile(r"\b(he|she|him|her|his|hers|himself|herself)\b", re.IGNORECASE)
_NUMBER_WORDS = re.compile(
    r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|dozen|half|"
    r"quarter|majority|most|all of)\b",
    re.IGNORECASE,
)
# Ordinary sentence words that may be capitalised (sentence starts) without being a fact.
_COMMON = {
    "a", "an", "the", "there", "this", "these", "that", "those", "here", "yes", "no", "none", "not", "of", "in", "out",
    "among", "across", "all", "only", "each", "both", "some", "and", "but", "so", "also", "for", "with", "at", "on",
    "by", "from", "as", "they", "their", "them", "it", "its", "you", "your", "we", "our", "i", "currently", "overall",
    "based", "according", "total", "in total", "altogether", "is", "are", "was", "were", "has", "have", "today",
    "resident", "residents", "room", "age", "average", "highest", "lowest", "oldest", "youngest", "newest",
    "status", "care", "type", "facility", "caregiver", "primary", "admitted", "admission", "discharged", "discharge",
    "registered", "registration", "date", "contact", "emergency", "phone", "record", "id", "years", "old",
    "breakdown", "showing", "shown", "listed", "below", "above", "of these", "one", "nobody", "unfortunately",
    "sure", "okay", "right", "now", "just", "unknown", "recorded", "matching", "match", "matches",
}


@dataclass
class Phrasing:
    status: str  # "phrased" | "skipped" | "rejected" | "failed"
    reason: str | None = None
    ms: int | None = None


def eligible(plan: QueryPlan, response: dict[str, Any]) -> str | None:
    """None when the answer may be rephrased, else the reason it is not (clarifications and refusals keep their
    fixed wording and options)."""
    if plan.action not in _ACTIONS:
        return f"action {plan.action}"
    if response.get("needsClarification") or response.get("isPolicyViolation"):
        return "clarification or refusal"
    if not response.get("summaryText"):
        return "no summary"
    return None


def facts_text(response: dict[str, Any], plan: QueryPlan | None = None, total: int | None = None) -> str:
    """The template answer as a short fact sheet: everything the rewrite may say, and nothing else.

    Count/list answers are given as labelled numbers, not the template's "5 of 88 residents matching …": the model
    read that as "88 residents with that status". Every number says what it counts."""
    if plan is not None and plan.action in ("count", "list") and total is not None and "totalMatched" in response:
        conditions = " · ".join(describe_filters(plan.filters)) or "none (all residents)"
        lines = [
            f"Residents matching the question: {response['count']}",
            f"Conditions: {conditions}",
            f"All residents in the records, of every status (not only matching ones): {total}",
        ]
    else:
        lines = [f"Headline: {response.get('headline', '')}", f"Summary: {response['summaryText']}"]
    for b in response.get("breakdown") or []:
        lines.append(f"- {b['label']}: {b['count']} ({b['percentage']}%)")
    for s in response.get("multiSpotlight") or []:
        detail = f" ({s['supportingDetail']})" if s.get("supportingDetail") else ""
        lines.append(f"- {s['label']}: {s['value']}{detail}")
    if response.get("aggregateValue") is not None:
        lines.append(f"Value: {response['aggregateValue']}")
    rows = response.get("residentsSample") or []
    if rows and not response.get("matchedResident"):
        lines.append(f"A table of {len(rows)} residents is shown below the reply.")
    return "\n".join(lines)


def _numbers(text: str) -> set[str]:
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 1,200 → 1200
    out = set()
    for n in _NUMBER.findall(text):
        out.add(n)
        out.add(n.lstrip("0") or "0")  # 2024-03-07 → "03" also matches "3"
        if "." in n:
            out.add(n.rstrip("0").rstrip("."))
    return out


def check(text: str, facts: str, question: str, response: dict[str, Any]) -> str | None:
    """None when the rewrite is faithful to the facts, else a reason code (never containing the text itself)."""
    if not 1 <= len(text) <= 600:
        return "length"
    if len(text.split()) < 4:  # a bare value repeats the headline and drops what it refers to
        return "too_short"
    if re.search(r"[*#`|_]|\bresident_\d", text) or "<think>" in text:
        return "format"
    allowed_numbers = _numbers(facts)
    if any(n not in allowed_numbers for n in _numbers(text)):
        return "number_not_in_facts"
    if _NUMBER_WORDS.search(text):
        return "number_word"
    if re.search(r"\bone\b", text, re.IGNORECASE) and "1" not in allowed_numbers:
        return "number_word"
    if _GENDERED.search(text):
        return "gendered_pronoun"
    known = set(_WORD.findall(f"{facts}\n{question}".lower())) | _COMMON
    for word in _CAPITALISED.findall(text):
        parts = _WORD.findall(word.lower())
        if parts and not all(p in known or p.rstrip("'’s") in known for p in parts):
            return "unknown_name_or_label"
    headline = str(response.get("headline") or "")
    for polarity in ("Yes", "No"):
        if headline.startswith(f"{polarity} —") and not re.match(rf"\s*{polarity}\b", text, re.IGNORECASE):
            return "polarity"
    if response.get("count") == 0 and not re.search(r"\b(no|none|0|not|nobody)\b", text, re.IGNORECASE):
        return "empty_result_not_stated"
    return None


def _clean(text: str) -> str:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = text.replace("<think>", "").strip().strip('"').strip()
    return re.sub(r"\s+", " ", text)


class ResponsePhraser:
    """Rewrites `summaryText` with the local model; keeps the template on any failure or fact-check miss."""

    def __init__(self, translator: Any, timeout_s: float, max_tokens: int):
        self._model = translator
        self._timeout = timeout_s
        self._max_tokens = max_tokens

    async def phrase(
        self, question: str, plan: QueryPlan, response: dict[str, Any], total: int | None = None
    ) -> tuple[dict[str, Any], Phrasing]:
        if not hasattr(self._model, "complete_text"):
            return response, Phrasing("skipped", "model cannot write text")
        if reason := eligible(plan, response):
            return response, Phrasing("skipped", reason)
        facts = facts_text(response, plan, total)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"QUESTION: {question.strip()}\n\nANSWER FACTS:\n{facts}"},
        ]
        started = time.perf_counter()
        try:
            raw = await self._model.complete_text(messages, max_tokens=self._max_tokens, timeout=self._timeout)
        except TranslatorUnavailable as exc:
            return response, Phrasing("failed", str(exc) or "unavailable", _ms(started))
        except Exception as exc:  # never let phrasing break an answer that is already correct
            log.warning("answer phrasing error: %s", type(exc).__name__)
            return response, Phrasing("failed", type(exc).__name__, _ms(started))
        text = _clean(raw)
        if problem := check(text, facts, question, response):
            return response, Phrasing("rejected", problem, _ms(started))
        return {**response, "summaryText": text, "templateSummaryText": response["summaryText"]}, Phrasing(
            "phrased", None, _ms(started)
        )


def _ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)
