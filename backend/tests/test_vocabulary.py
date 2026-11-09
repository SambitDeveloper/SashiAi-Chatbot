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
