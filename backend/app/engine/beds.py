"""Questions about the facility's beds, answered without any model.

Data: the Care App `beds` collection (one document per room, synced from PointClickCare): `roomId` and its `beds`, each
{description, isOccupied, isMedicareCertified, isMedicaidCertified, isCompanionBed}. Admins read it through snf_mcp's
run_query (SnfMcpRepository.bed_rooms); staff have no API tool for beds, so their bed questions are "not supported".
Same numbers as GET /api/beds/occupancy, from the last PCC sync (that route re-syncs first; reading here never writes).

Every word of a bed question must be explained by the word lists below (bed and room words, statuses, certifications,
room numbers, actions, fillers). Anything else ("beds in the memory care wing") is answered "please rephrase", never
with a number that ignored part of the question.
"""

import datetime as dt
import re
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

_BED = {"bed", "beds", "occupancy", "vacancy", "vacancies"}
_ROOM = {"room", "rooms"}
_PEOPLE = {"resident", "residents", "patient", "patients", "staff", "who", "whom", "whose"}
_AVAILABLE = {"available", "vacant", "free", "empty", "open", "unoccupied", "vacancy", "vacancies", "unused"}
_OCCUPIED = {"occupied", "taken", "filled", "full", "used", "booked"}
_FLAGS = {"medicare": "medicare", "medicaid": "medicaid", "companion": "companion"}
_FLAG_LABELS = {"medicare": "Medicare-certified", "medicaid": "Medicaid-certified", "companion": "companion"}
_LIST = {"list", "show", "which", "name", "names", "display"}
_RATE = {"rate", "percentage", "percent", "%"}
_GROUP = {"per", "each", "by", "breakdown", "wise"}
_NOT = {"not", "non", "no", "without"}
_FILLER = {
    "how", "many", "much", "what", "whats", "is", "are", "the", "our", "we", "us", "do", "does", "have", "has", "there",
    "in", "of", "a", "an", "total", "number", "count", "all", "me", "give", "tell", "please", "currently", "current",
    "right", "now", "today", "facility", "this", "at", "with", "any", "for", "and", "or", "status", "certified",
    "certification", "can", "you", "i", "see", "get", "overall", "left", "remaining", "still", "out",
    "s", "it", "be", "being", "that", "those", "these", "them", "summary", "details", "info", "information", "about",
}  # fmt: skip
_WORD = re.compile(r"\d+|%|[a-z]+")


@dataclass
class BedQuestion:
    status: bool | None = None  # True = occupied, False = available, None = any
    flags: dict[str, bool] = field(default_factory=dict)  # medicare / medicaid / companion → wanted value
    rooms: list[int] = field(default_factory=list)
    action: str = "count"  # count / list / rate / summary
    by_room: bool = False
    about_rooms: bool = False  # the question counts or lists ROOMS (e.g. "rooms with an available bed")


def is_bed_question(question: str, names_someone: bool, person_name: str | None = None) -> bool:
    """Beds are the subject: a bed or room word, no person named, and not about residents ("What bed is Ann in?" is a
    resident's room). `person_name`: what the one-person grammar took for a name; a name with a word the bed words do
    not explain ("In which room is Sagar staying?") makes it a question about that person."""
    words = set(_WORD.findall(question.lower()))
    if names_someone or not words & (_BED | _ROOM) or words & _PEOPLE:
        return False
    return person_name is None or parse(person_name) is not None


def parse(question: str) -> BedQuestion | None:
    """The question as a BedQuestion, or None when a word is not understood."""
    words = _WORD.findall(question.lower().replace("-", " ").replace("’", "'").replace("'", " "))
    q, negate, first_noun = BedQuestion(), False, None
    for i, w in enumerate(words):
        if w.isdigit():
            if i and words[i - 1] in _ROOM or q.rooms:
                q.rooms.append(int(w))
                continue
            return None  # a number that is not a room ("more than 5 beds"): not understood
        if w in _NOT:
            negate = True
            continue
        if w in _ROOM or w in {"bed", "beds"}:
            first_noun = first_noun or ("room" if w in _ROOM else "bed")
        elif w in _FLAGS:
            q.flags[_FLAGS[w]] = not negate
        elif w in _AVAILABLE or w in _OCCUPIED:
            q.status = (w in _OCCUPIED) != negate
        elif w == "occupancy":
            q.action = "rate" if words[i + 1 : i + 2] in (["rate"], ["percentage"]) else q.action if q.action != "count" else "summary"
        elif w in _RATE:
            q.action = "rate"
        elif w in _LIST:
            q.action = "list" if q.action in ("count", "summary") else q.action
        elif w in _GROUP:
            q.by_room = q.by_room or (i + 1 < len(words) and words[i + 1] in _ROOM) or w == "wise"
        elif w not in _FILLER and w not in _BED:
            return None
        negate = negate and w in _FILLER  # "not" applies to the next status or certification word
    q.about_rooms = first_noun == "room" and not q.by_room and q.action in ("count", "list") and not q.rooms
    if q.action == "count" and not (q.status is not None or q.flags or q.rooms or q.by_room or q.about_rooms):
        q.action = "summary"  # "how many beds" / "bed occupancy": the full picture
    return q


def flatten(rooms: list[dict]) -> list[dict]:
    """beds documents → one row per bed: {room, bed, occupied, medicare, medicaid, companion}."""
    return [{"room": r.get("roomId"), "bed": str(b.get("description") or b.get("id") or ""),
             "occupied": b.get("isOccupied") is True, "medicare": b.get("isMedicareCertified") is True,
             "medicaid": b.get("isMedicaidCertified") is True, "companion": b.get("isCompanionBed") is True}
            for r in rooms for b in (r.get("beds") if isinstance(r.get("beds"), list) else [])
            if isinstance(b, dict)]  # fmt: skip


def _as_of(rooms: list[dict], timezone: str) -> str:
    stamps = [d.get("$date") if isinstance(d, dict) else d for d in (r.get("updatedAt") for r in rooms)]
    stamps = [s for s in stamps if isinstance(s, str)]
    if not stamps:
        return ""
    when = dt.datetime.fromisoformat(max(stamps).replace("Z", "+00:00")).astimezone(ZoneInfo(timezone))
    return f" (as of the last PointClickCare sync, {when:%b %-d, %Y %H:%M})"


def describe(q: BedQuestion) -> str:
    parts = [("occupied" if q.status else "available") if q.status is not None else ""]
    parts += [("" if v else "non-") + _FLAG_LABELS[f] for f, v in q.flags.items()]
    return " ".join(p for p in parts if p)


def answer(q: BedQuestion, rooms: list[dict], timezone: str) -> dict:
    """The response card for one bed question (same shape as the resident answers)."""
    beds, as_of = flatten(rooms), _as_of(rooms, timezone)
    known_rooms = {r.get("roomId") for r in rooms}
    missing = [n for n in q.rooms if n not in known_rooms]
    if missing:
        text = f"Room {', '.join(map(str, missing))} is not in the bed records{as_of}."
        return {"headline": "Room not found", "summaryText": text, "count": 0, "needsClarification": True}
    scope = [b for b in beds if not q.rooms or b["room"] in q.rooms]
    match = [b for b in scope if (q.status is None or b["occupied"] == q.status)
             and all(b[f] == v for f, v in q.flags.items())]  # fmt: skip
    what, where = describe(q), (f" in room {', '.join(map(str, q.rooms))}" if q.rooms else "")
    noun = f"{what} beds".strip()

    def n_beds(n: int) -> str:  # "1 available bed", "7 available beds"
        return f"{n} {noun[:-1] if n == 1 else noun}"

    occupied = sum(b["occupied"] for b in scope)
    if q.action == "summary":
        rate = occupied / len(scope) * 100 if scope else 0
        text = (f"{len(scope)} beds{where} in {len({b['room'] for b in scope})} rooms: {occupied} occupied, "
                f"{len(scope) - occupied} available, occupancy {rate:.1f}%{as_of}.")  # fmt: skip
        return {"headline": f"Bed occupancy: {rate:.1f}%", "summaryText": text, "count": len(scope),
                "aggregateValue": round(rate, 1)}  # fmt: skip
    if q.action == "rate":
        base = [b for b in scope if all(b[f] == v for f, v in q.flags.items())]
        busy = sum(b["occupied"] for b in base)
        rate = busy / len(base) * 100 if base else 0
        label = f"{describe(BedQuestion(flags=q.flags))} beds".strip()
        text = f"{busy} of {len(base)} {label}{where} are occupied: {rate:.1f}%{as_of}."
        return {"headline": f"Occupancy rate: {rate:.1f}%", "summaryText": text, "count": len(base),
                "aggregateValue": round(rate, 1)}  # fmt: skip
    if q.by_room:
        per: dict[int, int] = {}
        for b in match:
            per[b["room"]] = per.get(b["room"], 0) + 1
        breakdown = [{"key": str(r), "label": f"Room {r}", "count": c, "percentage": round(c / len(match) * 100, 1)}
                     for r, c in sorted(per.items())]  # fmt: skip
        text = f"{n_beds(len(match))}{where} in {len(per)} rooms{as_of}."
        return {"headline": f"{noun.capitalize()} by room", "summaryText": text, "count": len(match), "breakdown": breakdown}
    if q.about_rooms:
        rooms_hit = sorted({b["room"] for b in match})
        text = f"{len(rooms_hit)} rooms" + (f" with {noun}" if what else "") + f"{as_of}."
        if q.action == "list" and rooms_hit:
            text += " Rooms: " + ", ".join(map(str, rooms_hit[:100])) + ("…" if len(rooms_hit) > 100 else "")
        return {"headline": f"{len(rooms_hit)} rooms", "summaryText": text, "count": len(rooms_hit)}
    text = f"{n_beds(len(match))}{where} of {len(scope)} beds{as_of}."
    if q.action == "list" and match:
        by_room: dict[int, list[str]] = {}
        for b in match:
            by_room.setdefault(b["room"], []).append(b["bed"])
        text += " " + "; ".join(f"Room {r}: {', '.join(sorted(v))}" for r, v in sorted(by_room.items())[:100])
    return {"headline": n_beds(len(match)), "summaryText": text, "count": len(match)}


HELP = ("I can answer bed questions about totals, occupancy and the occupancy rate, available or occupied beds, rooms "
        "and room numbers, and Medicare-certified, Medicaid-certified or companion beds. Please rephrase.")  # fmt: skip
