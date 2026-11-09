"""End-to-end pipeline tests with a fake translator in place of Claude."""

from tests.conftest import TODAY, FakeTranslator

JENKINS_ACTIVE = {
    "action": "count",
    "filters": {"status": {"include": ["active"]}, "primary_caregiver": {"include": ["Sarah Jenkins"]}},
}


async def test_question_that_failed_before_now_answers_9(make_service):
    q = "Tell me the count of active residents whose care taker is Nurse Sarah Jenkins?"
    fake = FakeTranslator({q: JENKINS_ACTIVE})
    service = await make_service(fake)
    result = await service.handle(q)
    assert result["response"]["count"] == 9
    assert result["tier"] == "llm"
    assert "Caregiver: Nurse Sarah Jenkins, RN" in result["interpretation"]
    assert fake.calls[0]["today"] == TODAY


async def test_resident_names_never_sent_to_model(make_service):
    q = "Who is the caretaker of Eleanor Vance?"
    fake = FakeTranslator(
        {
            "Who is the caretaker of resident_1?": {
                "action": "lookup",
                "lookup": {"residents": ["resident_1"], "attributes": ["primary_caregiver"]},
            }
        }
    )
    service = await make_service(fake)
    result = await service.handle(q)
    assert "Eleanor" not in fake.calls[0]["question"] and "Vance" not in fake.calls[0]["question"]
    assert result["response"]["headline"] == "Nurse Sarah Jenkins, RN — Primary Caregiver"


async def test_masked_template_is_cached_across_residents(make_service):
    fake = FakeTranslator(
        {
            "What is resident_1's room?": {
                "action": "lookup",
                "lookup": {"residents": ["resident_1"], "attributes": ["room_number"]},
            }
        }
    )
    service = await make_service(fake)
    first = await service.handle("What is Eleanor Vance's room?")
    second = await service.handle("What is Arthur Pendleton's room?")
    assert len(fake.calls) == 1, "second question must be served from the cache"
    assert second["tier"] == "cache" and second["tokensConsumed"] == 0
    assert first["response"]["headline"].startswith("MC-101")
    assert second["response"]["headline"].startswith("MC-104")


async def test_follow_up_uses_previous_plan(make_service):
    q2 = "And how many of them are in memory care?"
    fake = FakeTranslator(
        {
            "How many active residents does Sarah Jenkins look after?": JENKINS_ACTIVE,
            q2: {
                "action": "count",
                "filters": {
                    "status": {"include": ["active"]},
                    "primary_caregiver": {"include": ["Nurse Sarah Jenkins, RN"]},
                    "care_type": {"include": ["memory_care"]},
                },
                "uses_previous": True,
            },
        }
    )
    service = await make_service(fake)
    await service.handle("How many active residents does Sarah Jenkins look after?", session_id="s1")
    result = await service.handle(q2, session_id="s1")
    assert fake.calls[1]["previous"] is not None and "Nurse Sarah Jenkins, RN" in fake.calls[1]["previous"]
    assert result["response"]["count"] == 9

    # A follow-up answer is tied to its conversation: the same words in a fresh chat must not reuse it.
    await service.handle(q2, session_id="other-chat")
    assert len(fake.calls) == 3 and fake.calls[2]["previous"] is None


async def test_repeated_question_in_same_chat_uses_cache(make_service):
    q = "Who is the primary caretaker of Eleanor Vance ?"
    fake = FakeTranslator(
        {
            "Who is the primary caretaker of resident_1 ?": {
                "action": "lookup",
                "lookup": {"residents": ["resident_1"], "attributes": ["primary_caregiver"]},
            },
            "How many residents are active?": {"action": "count", "filters": {"status": {"include": ["active"]}}},
        }
    )
    service = await make_service(fake)
    tiers = [
        (await service.handle(x, session_id="tab-1"))["tier"] for x in [q, q, q, "How many residents are active?", q]
    ]
    assert tiers == ["llm", "cache", "cache", "llm", "cache"]
    assert len(fake.calls) == 2


async def test_no_previous_plan_without_session(make_service):
    fake = FakeTranslator(
        {"how many residents are active?": {"action": "count", "filters": {"status": {"include": ["active"]}}}}
    )
    service = await make_service(fake)
    await service.handle("how many residents are active?")
    assert fake.calls[0]["previous"] is None


async def test_unknown_caregiver_asks_instead_of_guessing(make_service):
    q = "How many residents does Nurse Joy look after?"
    service = await make_service(
        FakeTranslator({q: {"action": "count", "filters": {"primary_caregiver": {"include": ["Nurse Joy"]}}}})
    )
    result = await service.handle(q)
    assert result["response"]["needsClarification"]
    assert "Nurse Joy" in result["response"]["summaryText"]


async def test_unknown_wing_is_reported(make_service):
    q = "Residents in West Wing"
    service = await make_service(
        FakeTranslator({q: {"action": "count", "filters": {"wing": {"include": ["West Wing"]}}}})
    )
    result = await service.handle(q)
    assert result["response"]["needsClarification"] and "West Wing" in result["response"]["summaryText"]


async def test_ambiguous_resident_offers_choices(make_service):
    q = "Tell me about Roosevelt"
    service = await make_service(
        FakeTranslator({"Tell me about resident_1": {"action": "lookup", "lookup": {"residents": ["resident_1"]}}})
    )
    result = await service.handle(q)
    labels = [o["label"] for o in result["response"]["clarificationOptions"]]
    assert any("Franklin Roosevelt" in label for label in labels) and any(
        "Theodore Roosevelt" in label for label in labels
    )
    assert result["response"]["clarificationOptions"][0]["query"] in (
        "Tell me about Franklin Roosevelt",
        "Tell me about Theodore Roosevelt",
    )


async def test_unknown_resident_not_found(make_service):
    q = "Tell me about John Smith"
    service = await make_service(FakeTranslator({q: {"action": "lookup", "lookup": {"residents": ["John Smith"]}}}))
    result = await service.handle(q)
    assert "John Smith" in result["response"]["summaryText"]


async def test_invalid_model_output_is_not_executed(make_service):
    q = "How many residents are alive?"
    service = await make_service(
        FakeTranslator({q: {"action": "count", "filters": {"status": {"include": ["alive"]}}}})
    )
    result = await service.handle(q)
    assert result["response"]["needsClarification"] and result["mcpCall"] is None


async def test_unsupported_is_passed_through(make_service):
    q = "What medication is Eleanor Vance on?"
    service = await make_service(
        FakeTranslator(
            {
                "What medication is resident_1 on?": {
                    "action": "unsupported",
                    "message": "Medication data is not available.",
                }
            }
        )
    )
    result = await service.handle(q)
    assert result["response"]["isPolicyViolation"] and result["policy"]["passed"] is False


async def test_model_outage_returns_unavailable(make_service):
    service = await make_service(FakeTranslator(fail=True))
    result = await service.handle("How many residents are active?")
    assert result["tier"] == "unavailable" and result["response"]["headline"] == "Assistant unavailable"


async def test_no_model_configured(make_service):
    service = await make_service(None)
    result = await service.handle("How many residents are active?")
    assert "not configured" in result["response"]["summaryText"]


async def test_token_usage_reported(make_service):
    q = "How many residents are active?"
    service = await make_service(
        FakeTranslator({q: {"action": "count", "filters": {"status": {"include": ["active"]}}}})
    )
    result = await service.handle(q)
    assert result["tokenUsage"]["cache_read_input_tokens"] == 600
    assert service.stats.output_tokens == 40
