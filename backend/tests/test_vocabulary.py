import pytest

from app.nlu.vocabulary import EntityIndex


@pytest.fixture
async def index(repo):
    return EntityIndex(await repo.vocabulary())


@pytest.mark.parametrize(
    ("question", "masked", "ids"),
    [
        ("Tell me about Eleanor Vance", "Tell me about resident_1", [["RES-1001"]]),
        ("What is the status of Eleanor Vance?", "What is the status of resident_1?", [["RES-1001"]]),
        ("Elenor Vanse caregiver", "resident_1 caregiver", [["RES-1001"]]),
        ("Louisa May Alcott status", "resident_1 status", [["RES-1058"]]),
        ("tell me about vincent van gogh", "tell me about resident_1", [["RES-1037"]]),
        (
            "Compare Eleanor Vance and Arthur Pendleton",
            "Compare resident_1 and resident_2",
            [["RES-1001"], ["RES-1002"]],
        ),
        ("Tell me about Roosevelt", "Tell me about resident_1", [["RES-1018", "RES-1031"]]),
    ],
)
async def test_resident_names_are_masked(index, question, masked, ids):
    m = index.mask(question)
    assert m.text == masked
    assert list(m.placeholders.values()) == ids


@pytest.mark.parametrize(
    "question",
    [
        "May I see the active residents",  # "May" is not Louisa May-Alcott
        "active residents of Sarah Jenkins",  # staff names are not masked
        "Which residents does Maya Patel look after?",
        "how many residents are active",
    ],
)
async def test_no_false_masking(index, question):
    assert index.mask(question).text == question


async def test_caregiver_resolution(index):
    assert index.resolve_caregiver("Sarah Jenkins").values == ["Nurse Sarah Jenkins, RN"]
    assert index.resolve_caregiver("Jenkins").values == ["Nurse Sarah Jenkins, RN"]
    maya = index.resolve_caregiver("Maya Patel")
    assert len(maya.values) == 2 and not maya.ambiguous  # one person under two job titles
    assert index.resolve_caregiver("Dr Who").values == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("garden", ["Garden Terrace"]),
        ("Garden Terrance", ["Garden Terrace"]),
        ("north wing", ["North Wing"]),
        ("West Wing", []),
    ],
)
async def test_wing_resolution(index, text, expected):
    assert index.resolve_wing(text).values == expected


async def test_resident_resolution(index):
    assert index.resolve_resident("RES-1001", {}).values == ["RES-1001"]
    assert index.resolve_resident("RES-9999", {}).values == []
    assert index.resolve_resident("John Smith", {}).values == []
    assert index.resolve_resident("Roosevelt", {}).ambiguous


def test_facility_names_that_differ_only_by_number():
    """Real Care App facilities are named "(train) FACILITY_n": the number is the whole identity."""
    from app.data.repository import Vocabulary
    from app.nlu.vocabulary import EntityIndex

    wings = ["(train) FACILITY_1", "(train) FACILITY_4", "(train) FACILITY_12", "(train) FACILITY_42", "North Wing"]
    idx = EntityIndex(Vocabulary(wings=wings))
    # {text as written: [wing]}
    assert idx.mentioned_wings("how many residents are in (train) FACILITY_42") == {"(train) FACILITY_42": ["(train) FACILITY_42"]}
    assert list(idx.mentioned_wings("residents of train facility 12").values()) == [["(train) FACILITY_12"]]
    # the bracketed qualifier is optional; the number still has to match exactly
    assert idx.mentioned_wings("residents assigned to FACILITY_12") == {"FACILITY_12": ["(train) FACILITY_12"]}
    assert idx.mentioned_wings("residents assigned to FACILITY_99") == {}
    assert idx.resolve_wing("FACILITY_12").values == ["(train) FACILITY_12"]
    assert idx.resolve_wing("facility 4").values == ["(train) FACILITY_4"]
    assert idx.resolve_wing("FACILITY_99").values == []  # unknown number → not found, never a neighbour
    no_number = idx.resolve_wing("train facility")  # no number → genuinely ambiguous → the user is asked
    assert no_number.ambiguous and len(no_number.values) == 4
    assert idx.resolve_wing("north").values == ["North Wing"]  # letter names keep working


def test_surname_first_names_are_one_person():
    """Regression: "Jain Aarya" (surname first) was split into two names, the planner gave up and the model reused
    the previous question's resident. Surname-first must resolve to the same single resident."""
    import asyncio

    from app.data.memory_repository import MemoryResidentRepository
    from app.nlu.vocabulary import EntityIndex

    idx = EntityIndex(asyncio.run(MemoryResidentRepository().vocabulary()))
    for q in ("What is the admission date of Vance Eleanor?", "What is the admission date of Vance, Eleanor?"):
        m = idx.mask(q)
        assert list(m.placeholders.values()) == [["RES-1001"]], (q, m.text)
    assert idx.resolve_resident("Vance Eleanor", {}).values == ["RES-1001"]
    assert idx.resolve_resident("Eleanor Vance", {}).values == ["RES-1001"]
