"""QuestionStructure: subjects vs references, the comparison asked for, and contextual inclusion of the reference.

Questions are written with resident_N placeholders (what the analyser sees after masking). The rules under test are
grammatical, never question-specific: a resident is a reference only as the operand of a comparison construction.
"""

import pytest
import yaml

from app.nlu.intent_router import DEFAULT_ROUTES_FILE
from app.nlu.structure import StructureAnalyzer

A = StructureAnalyzer(yaml.safe_load(DEFAULT_ROUTES_FILE.read_text()))


def comps(q):
    return [(c.field, c.op, c.reference) for c in A.analyse(q).comparisons]


def refs(q):
    return [(r.key, r.inclusion) for r in A.analyse(q).references]


# ---------------------------------------------------------------- roles
@pytest.mark.parametrize(
    ("question", "subjects", "references"),
    [
        ("What is resident_1's room number?", ["resident_1"], []),
        ("Who is resident_1's caregiver and what is her room?", ["resident_1"], []),
        ("How many residents were admitted on the same day as resident_1?", [], ["resident_1"]),
        ("Is resident_1 older than resident_2?", ["resident_1"], ["resident_2"]),  # one of each
        ("Find residents who have Vik or Jerry listed as their primary caregiver.", [], []),  # "as" without "same"
        ("How many residents were admitted after March?", [], []),  # "after" + a date, not a resident
        ("residents in the same facility", [], []),  # no operand: not a reference comparison
    ],
)
def test_roles(question, subjects, references):
    s = A.analyse(question)
    assert s.subjects == subjects and [r.key for r in s.references] == references


# ---------------------------------------------------------------- comparisons per field kind
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("residents admitted on the same day as resident_1", [("admission_date", "same_day", "resident_1")]),
        ("residents admitted in the same month and year as resident_1", [("admission_date", "same_month", "resident_1")]),
        ("residents admitted in the same year as resident_1", [("admission_date", "same_year", "resident_1")]),
        ("residents admitted before resident_1", [("admission_date", "less", "resident_1")]),
        ("residents admitted earlier than resident_1", [("admission_date", "less", "resident_1")]),
        ("residents older than resident_1", [("age", "greater", "resident_1")]),  # the field is implied
        ("residents younger than resident_1", [("age", "less", "resident_1")]),
        ("residents of the same age as resident_1", [("age", "equal", "resident_1")]),
        ("residents in the same care level and facility as resident_1",
         [("care_type", "equal", "resident_1"), ("facility", "equal", "resident_1")]),
        ("residents with the same room number as resident_1", [("room_number", "equal", "resident_1")]),
    ],
)  # fmt: skip
def test_comparison_per_kind(question, expected):
    assert comps(question) == expected


def test_caregiver_set_comparisons_are_four_distinct_meanings():
    """overlaps (≥1 in common), same_set (exactly the same), disjoint (none in common), not_equal (not exactly the
    same) — "different" is NOT "no overlap"."""
    assert comps("residents who share a caregiver with resident_1") == [("primary_caregiver", "overlaps", "resident_1")]
    assert comps("residents who have the same caregiver as resident_1") == [("primary_caregiver", "overlaps", "resident_1")]
    assert comps("residents whose caregiver list overlaps with the caregiver list of resident_1") == [
        ("primary_caregiver", "overlaps", "resident_1")
    ]
    assert comps("residents with exactly the same caregivers as resident_1") == [("primary_caregiver", "same_set", "resident_1")]
    assert comps("residents who share no caregiver with resident_1") == [("primary_caregiver", "disjoint", "resident_1")]
    assert comps("residents with a different caregiver list from resident_1") == [("primary_caregiver", "not_equal", "resident_1")]


def test_pronoun_operand_stands_for_the_resident_named_before_it():
    s = A.analyse("Who are the other residents in the same facility as resident_1 who were admitted after her?")
    assert [(c.field, c.op, c.reference) for c in s.comparisons] == [
        ("facility", "equal", "resident_1"),
        ("admission_date", "greater", "resident_1"),
    ]


def test_comparison_without_a_field_is_undetermined():
    s = A.analyse("residents who are the same as resident_1")  # compared — on what is not said
    assert s.references and s.undetermined


@pytest.mark.parametrize(
    "question",
    [
        "who looks after resident_1",  # "after" belongs to the field phrase "looks after"
        "which nurse looks after resident_1 now",
        "how many residents were admitted after resident_1",  # no: this IS a comparison — see the next test
    ],
)
def test_comparison_words_inside_other_phrases_are_not_comparisons(question):
    s = A.analyse(question)
    if "admitted after" in question:  # the same word with a compared field is a comparison
        assert [(c.field, c.op) for c in s.comparisons] == [("admission_date", "greater")]
    else:
        assert not s.references and not s.undetermined


# ---------------------------------------------------------------- inclusion is contextual
@pytest.mark.parametrize(
    ("question", "inclusion"),
    [
        ("How many residents were admitted on the same day as resident_1?", "include"),
        ("List all other residents admitted on the same day as resident_1", "exclude"),  # "other" + entity noun
        ("Find residents sharing a caregiver with resident_1", "include"),
        ("Who else is in the same facility as resident_1?", "exclude"),  # "who else"
        ("residents besides resident_1 admitted on the same day as resident_1", "exclude"),  # preposition + reference
        ("the rest of the residents in the same facility as resident_1", "exclude"),
        # "other" that does not modify the answer set leaves it unchanged
        ("residents in the same facility as resident_1 with no other caregiver", "include"),
        ("residents who look after each other in the same facility as resident_1", "include"),
    ],
)
def test_inclusion(question, inclusion):
    assert refs(question)[0][1] == inclusion


# ---------------------------------------------------------------- comparisons among residents, nobody named
def test_cross_record_comparison():
    assert A.analyse("List all residents who share the same room number").cross_record
    assert A.analyse("show duplicate residents").cross_record
    assert not A.analyse("show the same for memory care").cross_record  # a follow-up, no field compared
    assert not A.analyse("how many residents share the same room number as resident_1").cross_record  # a reference


# ---------------------------------------------------------------- operation
@pytest.mark.parametrize(
    ("question", "operations"),
    [
        ("How many residents are active?", ["count"]),
        ("Which residents were admitted before resident_1, and how many are they?", ["count", "list"]),
        ("residents per care type", ["breakdown"]),
        ("average age of residents", ["aggregate"]),
    ],
)
def test_operations(question, operations):
    assert sorted(A.analyse(question).operations) == sorted(operations)
