"""The fast path's answers to simple questions about ONE named person (app/nlu/person_question.py): the candidates the
entity index knows → one lookup plan (resident, staff member or family member) or a clarification whose options each
name exactly one person. Plans use the ordinary QueryPlan and are executed and answered like every other plan."""

from app.models.plan import QueryPlan
from app.nlu.person_question import SHARED, PersonQuestion
from app.nlu.vocabulary import EntityIndex

KIND_LABELS = {"residents": "resident", "staff": "staff member", "family": "family member"}
_KIND_OF = {"residents": "resident", "staff": "staff", "family": "family"}
_STAFF_SHARED = {"facility": "staff_facility", "status": "staff_status"}
# one question word, several plan attributes ("contact details" = phone and email)
_STAFF_EXPAND = {"staff_contact": ["staff_phone", "staff_email"]}
# attributes of ONE person only: several matching staff members → ask which one
ONE_PERSON_ATTRIBUTES = ("staff_email", "staff_phone", "staff_residents", "staff_contact")


def family_detail(m: dict, index: EntityIndex) -> str:
    return f"{m.get('relation') or 'Family member'} of {index.display_by_id.get(m['residentId'], 'a resident')}"


def candidates(asked: PersonQuestion, index: EntityIndex) -> dict[str, list[dict]]:
    """Everyone the name can be, limited to the kinds the question can be about, narrowed by a "(detail)"."""
    people = index.people_named(asked.name, asked.kind)
    allowed = asked.kinds
    people = {k: (v if _KIND_OF[k] in allowed else []) for k, v in people.items()}
    if not asked.detail:
        return people
    d = asked.detail.strip().lower().removesuffix(", inactive")
    return {
        "residents": [r for r in people["residents"] if r["id"].lower() == d.removeprefix("id ").strip()],
        "staff": [s for s in people["staff"] if s["designation"].lower() == d],
        "family": [m for m in people["family"] if family_detail(m, index).lower() == d],
    }


def plan_for(asked: PersonQuestion, kind: str, people: list[dict]) -> QueryPlan:
    """The lookup plan for one resident / family member, or the named staff members (same name, several records)."""
    attrs = [a for a, _ in asked.attributes]
    if kind == "residents":
        filters = {"status": {"include": [asked.check]}} if asked.check else {}
        fields = None if asked.check else [a for a in attrs if not a.startswith("staff_")] or None
        return QueryPlan.model_validate({"action": "lookup", "lookup": {"residents": [people[0]["id"]]}, "filters": filters,
                                         **({"fields": fields} if fields else {})})  # fmt: skip
    if kind == "staff":
        shown = [_STAFF_SHARED.get(a, a) for a, scope in asked.attributes if scope in ("staff", SHARED)]
        shown = list(dict.fromkeys(x for a in shown for x in _STAFF_EXPAND.get(a, [a])))
        if asked.check:
            shown = list(dict.fromkeys([*shown, "staff_status"]))
        names = list(dict.fromkeys(p["name"] for p in people))[:5]
        return QueryPlan.model_validate({
            "action": "lookup", "entity": "staff", "lookup": {"staff": names},
            **({"fields": shown} if shown else {}),
            **({"filters": {"staff_status": {"include": [asked.check]}}} if asked.check else {}),
        })  # fmt: skip
    member = people[0]
    return QueryPlan.model_validate({"action": "lookup", "fields": ["family_relationship"],
                                     "lookup": {"residents": [member["residentId"]]},
                                     "filters": {"family_member": {"include": [member["id"]]}}})  # fmt: skip


def access_answer(asked: PersonQuestion, person: dict, permissions: list | None) -> dict:
    """"Can Jone access the transport calendar?": yes / no per section asked, for ONE staff member. Only the asked
    sections are answered (never the whole permission set, never write access)."""
    from app.data.staff import can_access

    label = person["name"] + (f" ({person['designation']})" if person.get("designation") else "")
    facts = []
    for section in asked.sections:
        ok = can_access(permissions, section)
        words = section.replace("-", " ")
        words = words[:1].upper() + words[1:]
        verdict = "not recorded" if ok is None else ("Yes" if ok else "No")
        facts.append({"label": words, "value": verdict,
                      "sentence": f"{label} {'can' if ok else 'cannot'} access {words}." if ok is not None
                      else f"{label}'s permissions don't include {words}."})  # fmt: skip
    one = facts[0] if len(facts) == 1 else None
    return {
        "headline": f"{one['value']} — {person['name']}" if one else f"Access — {person['name']}",
        "summaryText": " ".join(f["sentence"] for f in facts),
        "count": 1,
        **({"attributeSpotlight": {"label": f"Access: {one['label']}", "value": one["value"]}} if one else
           {"multiSpotlight": [{"label": f"Access: {f['label']}", "value": f["value"]} for f in facts]}),
    }  # fmt: skip


def message(asked: PersonQuestion, people: dict[str, list[dict]]) -> str:
    parts = [f"{len(v)} {KIND_LABELS[k]}{'' if len(v) == 1 else 's'}" for k, v in people.items() if v]
    found = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return (f'"{asked.name}" matches {found}. Which one do you mean? Adding the surname, or saying resident, staff '
            "member or family member, narrows it down.")  # fmt: skip


def options(asked: PersonQuestion, people: dict[str, list[dict]], index: EntityIndex, kind_words: dict[str, str],
            limit: int) -> list[dict[str, str]]:  # fmt: skip
    """Up to `limit` candidates, taken in turn from each kind. Each option is the SAME question about exactly one person:
    the name gets its kind word and a "(detail)" — record id for same-named residents, designation, relation."""
    text, (start, end) = asked.text, asked.name_span
    lead = asked.kind_span[0] if asked.kind_span else start
    possessive = "'s" if text[start:end].lower().replace("’", "'").endswith("'s") else ""
    # same words in any order ("Dodia Sagar" / "Sagar Dodia") are the same name to the search: tell them apart by id
    def words(name: str) -> tuple[str, ...]:
        return tuple(sorted(name.lower().split()))

    names = [words(r["name"]) for r in people["residents"]]

    # the kind word ("staff member Jone") only when the candidates are of several kinds, or the question named one
    several_kinds = sum(1 for v in people.values() if v) > 1 or asked.kind_span is not None

    def ask(kind: str, name: str, detail: str) -> str:
        word = f"{kind_words.get(kind, kind)} " if several_kinds else ""
        return text[:lead] + f"{word}{name}{detail}{possessive}" + text[end:]

    per_kind = [
        [{"label": f"Resident · {r['name']}" + (d := (f" (ID {r['id']})" if names.count(words(r["name"])) > 1 else "")),
          "query": ask("resident", r["name"], d)} for r in people["residents"]],
        [{"label": f"Staff · {s['name']}" + (f" ({s['designation']}{', inactive' if s['status'] == 'inactive' else ''})" if s["designation"] else ""),
          "query": ask("staff", s["name"], f" ({s['designation']})" if s["designation"] else "")} for s in people["staff"]],
        [{"label": f"Family · {m['name']} ({family_detail(m, index)})", "query": ask("family", m["name"], f" ({family_detail(m, index)})")}
         for m in people["family"] if index.display_by_id.get(m["residentId"])],
    ]  # fmt: skip
    out: list[dict[str, str]] = []
    while len(out) < limit and any(per_kind):
        for opts in per_kind:
            if opts and len(out) < limit:
                option = opts.pop(0)
                if option not in out:
                    out.append(option)
    return out
