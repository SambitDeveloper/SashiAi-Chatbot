"""Held-out evaluation set: 25 simple + 25 very complex resident questions (2026-11-13).

Written AFTER the prompt, few-shot examples and grounding rules were finalised, and never used to tune them, so it
measures generalisation rather than fit. Same format as gold_questions.py (one phrasing per item); expected plans
use canonical database values and are verified by executing them on tests/data/residents.json. Today = 2026-11-09.
"""

TODAY = "2026-11-09"
A = {"status": {"include": ["active"]}}
JENKINS = "Nurse Sarah Jenkins, RN"
MAYA = ["Wellness Coordinator Maya Patel", "Care Coordinator Maya Patel"]

EVAL: list[dict] = [
    # ------------------------------------------------------------------ 25 simple
    {"id": 101, "kind": "simple", "q": ["How many residents are inactive?"], "plan": {"action": "count", "filters": {"status": {"include": ["inactive"]}}}},
    {"id": 102, "kind": "simple", "q": ["How many male residents are there?"], "plan": {"action": "count", "filters": {"gender": {"include": ["Male"]}}}},
    {"id": 103, "kind": "simple", "q": ["Count residents in skilled nursing"], "plan": {"action": "count", "filters": {"care_type": {"include": ["skilled_nursing"]}}}},
    {"id": 104, "kind": "simple", "q": ["List the pending residents"], "plan": {"action": "list", "filters": {"status": {"include": ["pending"]}}}},
    {"id": 105, "kind": "simple", "q": ["How many residents are in the East Wing?"], "plan": {"action": "count", "filters": {"wing": {"include": ["East Wing"]}}}},
    {"id": 106, "kind": "simple", "q": ["Show residents in respite care"], "plan": {"action": "list", "filters": {"care_type": {"include": ["respite_care"]}}}},
    {"id": 107, "kind": "simple", "q": ["residents by care type"], "plan": {"action": "breakdown", "group_by": ["care_type"]}},
    {"id": 108, "kind": "simple", "q": ["How many residents does Marcus Chen look after?"], "plan": {"action": "count", "filters": {"primary_caregiver": {"include": ["Care Aide Marcus Chen"]}}}},
    {"id": 109, "kind": "simple", "q": ["average age of active residents"], "plan": {"action": "aggregate", "aggregate": {"op": "avg", "field": "age"}, "filters": A}},
    {"id": 110, "kind": "simple", "q": ["who is the youngest resident"], "plan": {"action": "list", "sort": {"field": "age", "direction": "asc"}, "limit": 1}},
    {"id": 111, "kind": "simple", "q": ["How many residents were admitted this month?"], "plan": {"action": "count", "filters": {"admission_date": {"start": "2026-11-01", "end": "2026-11-30"}}}},
    {"id": 112, "kind": "simple", "q": ["How many female residents are in assisted living?"], "plan": {"action": "count", "filters": {"gender": {"include": ["Female"]}, "care_type": {"include": ["assisted_living"]}}}},
    {"id": 113, "kind": "simple", "q": ["list residents in the South Wing"], "plan": {"action": "list", "filters": {"wing": {"include": ["South Wing"]}}}},
    {"id": 114, "kind": "simple", "q": ["Tell me about Harold Finch"], "plan": {"action": "lookup", "lookup": {"residents": ["RES-1004"]}}},
    {"id": 115, "kind": "simple", "q": ["how many residents are in independent living"], "plan": {"action": "count", "filters": {"care_type": {"include": ["independent_living"]}}}},
    {"id": 116, "kind": "simple", "q": ["residents per care type in North Wing"], "plan": {"action": "breakdown", "group_by": ["care_type"], "filters": {"wing": {"include": ["North Wing"]}}}},
    {"id": 117, "kind": "simple", "q": ["How many active residents are in Garden Terrace?"], "plan": {"action": "count", "filters": {**A, "wing": {"include": ["Garden Terrace"]}}}},
    {"id": 118, "kind": "simple", "q": ["show residents looked after by David O'Connor"], "plan": {"action": "list", "filters": {"primary_caregiver": {"include": ["Head Nurse David O'Connor, RN"]}}}},
    {"id": 119, "kind": "simple", "q": ["how many residents were admitted in October"], "plan": {"action": "count", "filters": {"admission_date": {"start": "2026-10-01", "end": "2026-10-31"}}}},
    {"id": 120, "kind": "simple", "q": ["who is the oldest resident in memory care"], "plan": {"action": "list", "filters": {"care_type": {"include": ["memory_care"]}}, "sort": {"field": "age", "direction": "desc"}, "limit": 1}},
    {"id": 121, "kind": "simple", "q": ["total number of residents in the North Wing"], "plan": {"action": "count", "filters": {"wing": {"include": ["North Wing"]}}}},
    {"id": 122, "kind": "simple", "q": ["How many residents are discharged?"], "plan": {"action": "count", "filters": {"status": {"include": ["discharged"]}}}},
    {"id": 123, "kind": "simple", "q": ["average age of female residents"], "plan": {"action": "aggregate", "aggregate": {"op": "avg", "field": "age"}, "filters": {"gender": {"include": ["Female"]}}}},
    {"id": 124, "kind": "simple", "q": ["thanks"], "plan": {"action": "help"}},
    {"id": 125, "kind": "simple", "q": ["How many residents does Liam Gallagher look after?"], "plan": {"action": "count", "filters": {"primary_caregiver": {"include": ["Nurse Liam Gallagher, LPN"]}}}},
    # ------------------------------------------------------------------ 25 very complex
    {"id": 201, "kind": "complex", "q": ["How many memory care residents in Garden Terrace are not active?"],
     "plan": {"action": "count", "filters": {"care_type": {"include": ["memory_care"]}, "wing": {"include": ["Garden Terrace"]}, "status": {"exclude": ["active"]}}}},
    {"id": 202, "kind": "complex", "q": ["List female residents over 85 in assisted living"],
     "plan": {"action": "list", "filters": {"gender": {"include": ["Female"]}, "age": {"min": 86}, "care_type": {"include": ["assisted_living"]}}}},
    {"id": 203, "kind": "complex", "q": ["How many residents aged between 75 and 85 are in skilled nursing or memory care?"],
     "plan": {"action": "count", "filters": {"age": {"min": 75, "max": 85}, "care_type": {"include": ["skilled_nursing", "memory_care"]}}}},
    {"id": 204, "kind": "complex", "q": ["Which male residents in the North Wing were admitted in the last 6 months?"],
     "plan": {"action": "list", "filters": {"gender": {"include": ["Male"]}, "wing": {"include": ["North Wing"]}, "admission_date": {"start": "2026-05-09", "end": "2026-11-09"}}}},
    {"id": 205, "kind": "complex", "q": ["How many residents in memory care are not looked after by Sarah Jenkins?"],
     "plan": {"action": "count", "filters": {"care_type": {"include": ["memory_care"]}, "primary_caregiver": {"exclude": [JENKINS]}}}},
    {"id": 206, "kind": "complex", "q": ["breakdown of female residents by wing"],
     "plan": {"action": "breakdown", "group_by": ["wing"], "filters": {"gender": {"include": ["Female"]}}}},
    {"id": 207, "kind": "complex", "q": ["average age per care type for active residents"],
     "plan": {"action": "aggregate", "aggregate": {"op": "avg", "field": "age"}, "group_by": ["care_type"], "filters": A}},
    {"id": 208, "kind": "complex", "q": ["the 5 oldest active residents"],
     "plan": {"action": "list", "filters": A, "sort": {"field": "age", "direction": "desc"}, "limit": 5}},
    {"id": 209, "kind": "complex", "q": ["How many residents in the East Wing or South Wing are under 75?"],
     "plan": {"action": "count", "filters": {"wing": {"include": ["East Wing", "South Wing"]}, "age": {"max": 74}}}},
    {"id": 210, "kind": "complex", "q": ["Is Eleanor Vance active and in memory care?"],
     "plan": {"action": "lookup", "lookup": {"residents": ["RES-1001"]}, "filters": {**A, "care_type": {"include": ["memory_care"]}}}},
    {"id": 211, "kind": "complex", "q": ["list residents admitted since August who are not active"],
     "plan": {"action": "list", "filters": {"admission_date": {"start": "2026-08-01"}, "status": {"exclude": ["active"]}}}},
    {"id": 212, "kind": "complex", "q": ["How many assisted living residents were discharged this year?"],
     "plan": {"action": "count", "filters": {"status": {"include": ["discharged"]}, "discharge_date": {"start": "2026-01-01", "end": "2026-12-31"}, "care_type": {"include": ["assisted_living"]}}}},
    {"id": 213, "kind": "complex", "q": ["maximum age in the South Wing"],
     "plan": {"action": "aggregate", "aggregate": {"op": "max", "field": "age"}, "filters": {"wing": {"include": ["South Wing"]}}}},
    {"id": 214, "kind": "complex", "q": ["how many memory care residents does each caregiver have"],
     "plan": {"action": "breakdown", "group_by": ["primary_caregiver"], "filters": {"care_type": {"include": ["memory_care"]}}}},
    {"id": 215, "kind": "complex", "q": ["count male residents in independent living excluding the North Wing"],
     "plan": {"action": "count", "filters": {"gender": {"include": ["Male"]}, "care_type": {"include": ["independent_living"]}, "wing": {"exclude": ["North Wing"]}}}},
    {"id": 216, "kind": "complex", "q": ["which pending or admitted residents are in skilled nursing"],
     "plan": {"action": "list", "filters": {"status": {"include": ["pending", "admitted"]}, "care_type": {"include": ["skilled_nursing"]}}}},
    {"id": 217, "kind": "complex", "q": ["the 3 most recently admitted residents in the East Wing"],
     "plan": {"action": "list", "filters": {"wing": {"include": ["East Wing"]}}, "sort": {"field": "admission_date", "direction": "desc"}, "limit": 3}},
    {"id": 218, "kind": "complex", "q": ["residents aged 90 or older"],
     "plan": {"action": "list", "filters": {"age": {"min": 90}}}},
    {"id": 219, "kind": "complex", "q": ["how many of Maya Patel's active residents are female"],
     "plan": {"action": "count", "filters": {**A, "primary_caregiver": {"include": MAYA}, "gender": {"include": ["Female"]}}}},
    {"id": 220, "kind": "complex", "q": ["breakdown by status of residents in respite care"],
     "plan": {"action": "breakdown", "group_by": ["status"], "filters": {"care_type": {"include": ["respite_care"]}}}},
    {"id": 221, "kind": "complex", "q": ["average age of residents admitted this year"],
     "plan": {"action": "aggregate", "aggregate": {"op": "avg", "field": "age"}, "filters": {"admission_date": {"start": "2026-01-01", "end": "2026-12-31"}}}},
    {"id": 222, "kind": "complex", "followup_of": "How many active residents are in assisted living?", "q": ["and how many of them are male?"],
     "plan": {"action": "count", "filters": {**A, "care_type": {"include": ["assisted_living"]}, "gender": {"include": ["Male"]}}}},
    {"id": 223, "kind": "complex", "followup_of": "list residents in Garden Terrace", "q": ["which of those are in memory care?"],
     "plan": {"action": "list", "filters": {"wing": {"include": ["Garden Terrace"]}, "care_type": {"include": ["memory_care"]}}}},
    {"id": 224, "kind": "complex", "q": ["how many residents in the North Wing are neither active nor pending"],
     "plan": {"action": "count", "filters": {"wing": {"include": ["North Wing"]}, "status": {"exclude": ["active", "pending"]}}}},
    {"id": 225, "kind": "complex", "q": ["Compare Harold Finch and Walter Bishop"],
     "plan": {"action": "lookup", "lookup": {"residents": ["RES-1004", "RES-1008"]}}},
]  # fmt: skip

assert len(EVAL) == 50 and sum(e["kind"] == "simple" for e in EVAL) == 25
