"""Answer phrasing by the local model: shown only when it passes the fact check; template kept otherwise."""

import pytest

from app.engine.phraser import ResponsePhraser, check, facts_text
from app.models.plan import QueryPlan
from app.nlu.translator import TranslatorUnavailable
from tests.conftest import FakeTranslator, PinnedDateSettings

COUNT_RESPONSE = {
    "headline": "12 residents",
    "summaryText": "12 of 40 residents matching Status: Active · Care Type: Memory Care.",
    "count": 12,
}
CHECK_RESPONSE = {
    "headline": "No — Eleanor Vance · Assisted Living",
    "summaryText": "Eleanor Vance (R-001) does not match Care Type: Memory Care. Current record: Active, Assisted Living.",
    "count": 1,
}


def _check(text, response=COUNT_RESPONSE, question="how many active memory care residents?"):
    return check(text, facts_text(response), question, response)


@pytest.mark.parametrize(
    "text",
    [
        "There are 12 active residents in Memory Care, out of 40 in total.",
        "12 of the 40 residents are active and in Memory Care.",
    ],
)
def test_faithful_rewrites_pass(text):
    assert _check(text) is None


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("There are 13 active residents in Memory Care.", "number_not_in_facts"),
        ("That is 30% of all residents.", "number_not_in_facts"),
        ("Twelve residents are active in Memory Care.", "number_word"),
        ("There are 12 active residents in the Rosewood Wing.", "unknown_name_or_label"),
        ("**12** active residents in Memory Care.", "format"),
        ("12 residents; she is the oldest.", "gendered_pronoun"),
        ("", "length"),
        ("12", "too_short"),
    ],
)
def test_unfaithful_rewrites_are_rejected(text, reason):
    assert _check(text) == reason


def test_yes_no_polarity_must_match():
    q = "is Eleanor Vance in memory care?"
    assert _check("No, Eleanor Vance is in Assisted Living, not Memory Care.", CHECK_RESPONSE, q) is None
    assert _check("Yes, Eleanor Vance is in Memory Care.", CHECK_RESPONSE, q) == "polarity"


def test_empty_result_must_be_stated():
    empty = {"headline": "No matching residents", "summaryText": "0 of 40 residents matching Care Type: Respite.", "count": 0}
    assert _check("There are no residents in Respite care right now.", empty) is None
    assert _check("Respite care residents are listed below.", empty) == "empty_result_not_stated"


class TextModel:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.messages = reply, error, None

    async def complete_text(self, messages, max_tokens, timeout):
        self.messages = messages
        if self.error:
            raise self.error
        return self.reply


PLAN = QueryPlan.model_validate({"action": "count"})


async def test_phraser_replaces_summary_and_keeps_template():
    model = TextModel("There are 12 active residents in Memory Care, out of 40 in total.")
    out, p = await ResponsePhraser(model, 5, 100).phrase("how many active memory care residents?", PLAN, COUNT_RESPONSE)
    assert p.status == "phrased"
    assert out["summaryText"].startswith("There are 12")
    assert out["templateSummaryText"] == COUNT_RESPONSE["summaryText"]
    assert out["headline"] == COUNT_RESPONSE["headline"] and out["count"] == 12
    assert "how many active memory care residents?" in model.messages[1]["content"]


@pytest.mark.parametrize(
    ("model", "status"),
    [
        (TextModel("There are 14 residents."), "rejected"),
        (TextModel(error=TranslatorUnavailable("timeout")), "failed"),
        (TextModel(error=RuntimeError("boom")), "failed"),
    ],
)
async def test_phraser_keeps_template_on_any_problem(model, status):
    out, p = await ResponsePhraser(model, 5, 100).phrase("q", PLAN, COUNT_RESPONSE)
    assert p.status == status
    assert out is COUNT_RESPONSE


async def test_clarifications_are_never_rephrased():
    model = TextModel("anything")
    clarify = QueryPlan.model_validate({"action": "clarify", "message": "Which resident?"})
    _, p = await ResponsePhraser(model, 5, 100).phrase("q", clarify, {"summaryText": "Which resident?"})
    assert p.status == "skipped" and model.messages is None


# ---- end to end through the service ---------------------------------------------------------------------------

Q = "Tell me the count of active residents whose care taker is Nurse Sarah Jenkins?"
PLAN_JSON = {
    "action": "count",
    "filters": {"status": {"include": ["active"]}, "primary_caregiver": {"include": ["Sarah Jenkins"]}},
}


class PhrasingFake(FakeTranslator):
    def __init__(self, plans, reply):
        super().__init__(plans)
        self.reply = reply
        self.text_calls = 0

    async def complete_text(self, messages, max_tokens, timeout):
        self.text_calls += 1
        return self.reply


async def _service(repo, reply, enabled=True):
    from app.engine.service import QueryService

    settings = PinnedDateSettings(
        _env_file=None, environment="test", topic_gate_enabled=False, semantic_router_enabled=False,
        llm_response_phrasing=enabled,
    )  # fmt: skip
    fake = PhrasingFake({Q: PLAN_JSON}, reply)
    service = QueryService(repo, fake, settings)
    await service.refresh_vocabulary()
    return service, fake


async def test_service_shows_checked_rewrite(repo):
    service, fake = await _service(repo, "9 active residents have Nurse Sarah Jenkins, RN as their caregiver.")
    result = await service.handle(Q)
    r = result["response"]
    assert r["count"] == 9 and fake.text_calls == 1
    assert r["summaryText"] == "9 active residents have Nurse Sarah Jenkins, RN as their caregiver."
    assert r["templateSummaryText"].startswith("9 of ")


async def test_service_keeps_template_when_rewrite_changes_a_number(repo):
    service, _ = await _service(repo, "10 active residents have Nurse Sarah Jenkins as their caregiver.")
    r = (await service.handle(Q))["response"]
    assert r["summaryText"].startswith("9 of ") and "templateSummaryText" not in r


async def test_phrasing_is_off_by_default(repo):
    service, fake = await _service(repo, "unused", enabled=False)
    r = (await service.handle(Q))["response"]
    assert fake.text_calls == 0 and r["summaryText"].startswith("9 of ")
