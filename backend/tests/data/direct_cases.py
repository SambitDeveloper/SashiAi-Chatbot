"""Labeled questions for the direct (no-LLM) route. None = must go to the language model."""

A = {"status": {"include": ["active"]}}
CASES = [
    # should be answerable directly
    ("How many residents are there?", {"action": "count"}),
    ("How many residents are active?", {"action": "count", "filters": A}),
    (
        "how many residents are in memory care",
        {"action": "count", "filters": {"care_type": {"include": ["memory_care"]}}},
    ),
    ("How many female residents are there?", {"action": "count", "filters": {"gender": {"include": ["Female"]}}}),
    (
        "how many active residents does Care Aide Elena Rostova look after",
        {"action": "count", "filters": {**A, "primary_caregiver": {"include": ["Care Aide Elena Rostova"]}}},
    ),
    ("how many residents in the North Wing", {"action": "count", "filters": {"wing": {"include": ["North Wing"]}}}),
    (
        "list the active residents in assisted living",
        {"action": "list", "filters": {**A, "care_type": {"include": ["assisted_living"]}}},
    ),
    ("show me the discharged residents", {"action": "list", "filters": {"status": {"include": ["discharged"]}}}),
    ("residents by status", {"action": "breakdown", "group_by": ["status"]}),
    ("how many residents in each wing", {"action": "breakdown", "group_by": ["wing"]}),
    ("residents per caregiver", {"action": "breakdown", "group_by": ["primary_caregiver"]}),
    ("what is the average age of residents", {"action": "aggregate", "aggregate": {"op": "avg", "field": "age"}}),
    ("who is the oldest resident", {"action": "list", "sort": {"field": "age", "direction": "desc"}, "limit": 1}),
    # comparisons, negation and age ranges are resolved locally (dates are covered in test_direct_grammar.py)
    ("how many residents are not active", {"action": "count", "filters": {"status": {"exclude": ["active"]}}}),
    ("how many residents are over 80", {"action": "count", "filters": {"age": {"min": 81}}}),
    ("residents who are not in memory care", {"action": "list", "filters": {"care_type": {"exclude": ["memory_care"]}}}),
    ("residents aged between 70 and 80", {"action": "list", "filters": {"age": {"min": 70, "max": 80}}}),
    (
        "Tell me about Robert MacKenzi category and staus",
        {"action": "lookup", "lookup": {"residents": ["RES-1006"], "attributes": ["care_type", "status"]}},
    ),
    (
        "Who is the primary caretaker of Eleanor Vance ?",
        {"action": "lookup", "lookup": {"residents": ["RES-1001"], "attributes": ["primary_caregiver"]}},
    ),
    ("hi", {"action": "help"}),
    # must go to the language model
    (
        "How many active residents are available who joined october month and undergoes Care Aide Elena Rostova care taker?",
        None,
    ),
    ("And how many of them are in memory care?", None),
    ("active residents with diabetes", None),
    ("Tell me about Roosevelt", None),
    ("how many dementia patients", None),
    ("the 3 oldest residents", None),
    ("Compare Eleanor Vance and Arthur Pendleton", None),
    ("how many residents in the north", None),
    ("residents of nurse joy", None),
]
