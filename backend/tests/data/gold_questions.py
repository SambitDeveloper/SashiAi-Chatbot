"""Gold benchmark: 50 resident questions with verified expected query plans, each with 2 paraphrases (150 items).

NEWLY AUTHORED for the Gemini → local Qwen migration (the historical "50-question baseline" was not in the repo).
Seeded from tests/data/direct_cases.py and tests/test_service.py questions; expected plans are written by hand and
verified by executing them against tests/data/residents.json (scripts/benchmark.py prints the expected result of each).

Today is pinned to 2026-11-09 (Monday), as in tests/conftest.py.
kind:  simple     = a straightforward question the deterministic planner is expected to answer (0 model calls)
       complex    = needs model interpretation (negation, ranges, top-N, lookups, combinations, follow-ups)
       oos        = out of scope; the expected plan is "unsupported"
Expected plans use canonical database values (exact caregiver/wing names, resident ids). Lookup attributes are
not scored (the answer shows the resident profile either way).
"""

TODAY = "2026-11-09"
A = {"status": {"include": ["active"]}}
OCT = {"start": "2026-10-01", "end": "2026-10-31"}
ELENA = "Care Aide Elena Rostova"
JENKINS = "Nurse Sarah Jenkins, RN"
MAYA = ["Wellness Coordinator Maya Patel", "Care Coordinator Maya Patel"]

GOLD: list[dict] = [
    # ------------------------------------------------------------------ simple (deterministic expected)
    {"id": 1, "kind": "simple", "q": ["How many residents are active?", "Count the active residents", "what's the number of active residents"],
     "plan": {"action": "count", "filters": A}},
    {"id": 2, "kind": "simple", "q": ["Count female residents.", "How many female residents are there?", "number of female residents"],
     "plan": {"action": "count", "filters": {"gender": {"include": ["Female"]}}}},
    {"id": 3, "kind": "simple", "q": ["How many residents are in memory care?", "count residents in memory care", "total memory care residents"],
     "plan": {"action": "count", "filters": {"care_type": {"include": ["memory_care"]}}}},
    {"id": 4, "kind": "simple", "q": ["Show discharged residents.", "list the discharged residents", "which residents are discharged"],
     "plan": {"action": "list", "filters": {"status": {"include": ["discharged"]}}}},
    {"id": 5, "kind": "simple", "q": ["How many residents were admitted last month?", "number of residents admitted last month", "count residents who joined last month"],
     "plan": {"action": "count", "filters": {"admission_date": OCT}}},
    {"id": 6, "kind": "simple", "q": ["What is the average age of residents?", "average resident age", "mean age of residents"],
     "plan": {"action": "aggregate", "aggregate": {"op": "avg", "field": "age"}}},
    {"id": 7, "kind": "simple", "q": ["How many residents are in the North Wing?", "count residents in North Wing", "number of residents in the north wing"],
     "plan": {"action": "count", "filters": {"wing": {"include": ["North Wing"]}}}},
    {"id": 8, "kind": "simple", "q": ["residents by status", "breakdown of residents by status", "how many residents in each status"],
     "plan": {"action": "breakdown", "group_by": ["status"]}},
    {"id": 9, "kind": "simple", "q": ["How many residents in each wing?", "residents per wing", "wing wise resident count"],
     "plan": {"action": "breakdown", "group_by": ["wing"]}},
    {"id": 10, "kind": "simple", "q": ["residents per caregiver", "how many residents does each caregiver have", "caregiver breakdown of residents"],
     "plan": {"action": "breakdown", "group_by": ["primary_caregiver"]}},
    {"id": 11, "kind": "simple", "q": ["who is the oldest resident", "oldest resident", "who's the eldest resident"],
     "plan": {"action": "list", "sort": {"field": "age", "direction": "desc"}, "limit": 1}},
    {"id": 12, "kind": "simple", "q": ["who is the youngest resident in independent living", "youngest resident in independent living", "who is the youngest independent living resident"],
     "plan": {"action": "list", "filters": {"care_type": {"include": ["independent_living"]}}, "sort": {"field": "age", "direction": "asc"}, "limit": 1}},
    {"id": 13, "kind": "simple", "q": ["list the active residents in assisted living", "show active assisted living residents", "which active residents are in assisted living"],
     "plan": {"action": "list", "filters": {**A, "care_type": {"include": ["assisted_living"]}}}},
    {"id": 14, "kind": "simple", "q": ["how many active residents does Care Aide Elena Rostova look after", "count active residents of Elena Rostova", "number of active residents cared for by Elena Rostova"],
     "plan": {"action": "count", "filters": {**A, "primary_caregiver": {"include": [ELENA]}}}},
    {"id": 15, "kind": "simple", "q": ["How many residents are pending?", "count pending residents", "how many pending residents are there"],
     "plan": {"action": "count", "filters": {"status": {"include": ["pending"]}}}},
    {"id": 16, "kind": "simple", "q": ["residents by gender", "gender breakdown of residents", "split residents by gender"],
     "plan": {"action": "breakdown", "group_by": ["gender"]}},
    {"id": 17, "kind": "simple", "q": ["average age of memory care residents", "mean age in memory care", "what is the average age of residents in memory care"],
     "plan": {"action": "aggregate", "aggregate": {"op": "avg", "field": "age"}, "filters": {"care_type": {"include": ["memory_care"]}}}},
    {"id": 18, "kind": "simple", "q": ["Tell me about Eleanor Vance", "Eleanor Vance profile", "details of Eleanor Vance"],
     "plan": {"action": "lookup", "lookup": {"residents": ["RES-1001"]}}},
    {"id": 19, "kind": "simple", "q": ["How many residents were discharged this year?", "number of residents discharged this year", "count residents discharged this year"],
     "plan": {"action": "count", "filters": {"status": {"include": ["discharged"]}, "discharge_date": {"start": "2026-01-01", "end": "2026-12-31"}}}},
    {"id": 20, "kind": "simple", "q": ["How many residents are there?", "total number of residents", "what is the census"],
     "plan": {"action": "count"}},
    {"id": 21, "kind": "simple", "q": ["residents admitted in the last 3 months", "list residents admitted in the past 3 months", "show residents who joined in the last 3 months"],
     "plan": {"action": "list", "filters": {"admission_date": {"start": "2026-08-09", "end": "2026-11-09"}}}},
    {"id": 22, "kind": "simple", "q": ["hi", "hello", "what can you do"],
     "plan": {"action": "help"}},
    {"id": 23, "kind": "simple", "q": ["How many residents in skilled nursing were admitted this year?", "count skilled nursing residents admitted this year", "number of skilled nursing residents who joined this year"],
     "plan": {"action": "count", "filters": {"care_type": {"include": ["skilled_nursing"]}, "admission_date": {"start": "2026-01-01", "end": "2026-12-31"}}}},
    # ------------------------------------------------------------------ complex (model interpretation expected)
    {"id": 24, "kind": "complex", "q": ["How many residents are not active?", "count residents who aren't active", "number of residents excluding active ones"],
     "plan": {"action": "count", "filters": {"status": {"exclude": ["active"]}}}},
    {"id": 25, "kind": "complex", "q": ["how many residents are over 80", "count residents older than 80", "number of residents above 80 years"],
     "plan": {"action": "count", "filters": {"age": {"min": 81}}}},
    {"id": 26, "kind": "complex", "q": ["residents aged between 70 and 80", "list residents between 70 and 80 years old", "which residents are 70 to 80 years old"],
     "plan": {"action": "list", "filters": {"age": {"min": 70, "max": 80}}}},
    {"id": 27, "kind": "complex", "q": ["the 3 oldest residents", "top 3 oldest residents", "show me the three oldest residents"],
     "plan": {"action": "list", "sort": {"field": "age", "direction": "desc"}, "limit": 3}},
    {"id": 28, "kind": "complex", "q": ["Is Arthur Pendleton in memory care?", "is Arthur Pendleton a memory care resident", "does Arthur Pendleton receive memory care"],
     "plan": {"action": "lookup", "lookup": {"residents": ["RES-1002"]}, "filters": {"care_type": {"include": ["memory_care"]}}}},
    {"id": 29, "kind": "complex", "q": ["What is Eleanor Vance's room number?", "which room is Eleanor Vance in", "room of Eleanor Vance"],
     "plan": {"action": "lookup", "lookup": {"residents": ["RES-1001"]}}},
    {"id": 30, "kind": "complex", "q": ["Who is the primary caretaker of Robert MacKenzie?", "who looks after Robert MacKenzie", "Robert MacKenzie's caregiver"],
     "plan": {"action": "lookup", "lookup": {"residents": ["RES-1006"]}}},
    {"id": 31, "kind": "complex", "q": ["How many active residents are in memory care or skilled nursing?", "count active residents in memory care and skilled nursing", "number of active memory care or skilled nursing residents"],
     "plan": {"action": "count", "filters": {**A, "care_type": {"include": ["memory_care", "skilled_nursing"]}}}},
    {"id": 32, "kind": "complex", "q": ["breakdown of active residents by care type and wing", "active residents per care type per wing", "split active residents by care type and wing"],
     "plan": {"action": "breakdown", "group_by": ["care_type", "wing"], "filters": A}},
    {"id": 33, "kind": "complex", "q": ["average age per wing", "mean age by wing", "what's the average age in each wing"],
     "plan": {"action": "aggregate", "aggregate": {"op": "avg", "field": "age"}, "group_by": ["wing"]}},
    {"id": 34, "kind": "complex", "q": ["how many male residents in the South Wing are over 85", "count male residents over 85 in South Wing", "number of men older than 85 in the South Wing"],
     "plan": {"action": "count", "filters": {"gender": {"include": ["Male"]}, "wing": {"include": ["South Wing"]}, "age": {"min": 86}}}},
    {"id": 35, "kind": "complex", "q": ["list memory care residents excluding Garden Terrace", "memory care residents not in Garden Terrace", "which memory care residents are not in Garden Terrace"],
     "plan": {"action": "list", "filters": {"care_type": {"include": ["memory_care"]}, "wing": {"exclude": ["Garden Terrace"]}}}},
    {"id": 36, "kind": "complex", "q": ["How many active residents are available who joined october month and undergoes Care Aide Elena Rostova care taker?", "how many active residents of Elena Rostova were admitted in October", "count active residents admitted in October whose caregiver is Elena Rostova"],
     "plan": {"action": "count", "filters": {**A, "admission_date": OCT, "primary_caregiver": {"include": [ELENA]}}}},
    {"id": 37, "kind": "complex", "q": ["who was admitted most recently", "most recently admitted resident", "show the latest admitted resident"],
     "plan": {"action": "list", "sort": {"field": "admission_date", "direction": "desc"}, "limit": 1}},
    {"id": 38, "kind": "complex", "q": ["the 5 most recently admitted residents in assisted living", "latest 5 admissions in assisted living", "show the five newest assisted living residents by admission date"],
     "plan": {"action": "list", "filters": {"care_type": {"include": ["assisted_living"]}}, "sort": {"field": "admission_date", "direction": "desc"}, "limit": 5}},
    {"id": 39, "kind": "complex", "q": ["How many female residents does Nurse Sarah Jenkins look after?", "count women cared for by Sarah Jenkins", "number of female residents of Nurse Sarah Jenkins"],
     "plan": {"action": "count", "filters": {"gender": {"include": ["Female"]}, "primary_caregiver": {"include": [JENKINS]}}}},
    {"id": 40, "kind": "complex", "q": ["minimum age of residents in respite care", "what is the lowest age among respite care residents", "min age in respite care"],
     "plan": {"action": "aggregate", "aggregate": {"op": "min", "field": "age"}, "filters": {"care_type": {"include": ["respite_care"]}}}},
    {"id": 41, "kind": "complex", "q": ["how many residents were admitted since March", "count admissions since March", "number of residents who joined since March"],
     "plan": {"action": "count", "filters": {"admission_date": {"start": "2026-03-01"}}}},
    {"id": 42, "kind": "complex", "q": ["residents admitted in 2025", "list residents who were admitted in 2025", "which residents joined in 2025"],
     "plan": {"action": "list", "filters": {"admission_date": {"start": "2025-01-01", "end": "2025-12-31"}}}},
    {"id": 43, "kind": "complex", "q": ["how many residents in the East Wing are pending or admitted", "count pending or admitted residents in East Wing", "number of East Wing residents who are pending or admitted"],
     "plan": {"action": "count", "filters": {"wing": {"include": ["East Wing"]}, "status": {"include": ["pending", "admitted"]}}}},
    {"id": 44, "kind": "complex", "q": ["Compare Eleanor Vance and Arthur Pendleton", "show Eleanor Vance and Arthur Pendleton", "details for Eleanor Vance and Arthur Pendleton"],
     "plan": {"action": "lookup", "lookup": {"residents": ["RES-1001", "RES-1002"]}}},
    {"id": 45, "kind": "complex", "q": ["How many residents does Maya Patel look after?", "count residents of Maya Patel", "number of residents cared for by Maya Patel"],
     "plan": {"action": "count", "filters": {"primary_caregiver": {"include": MAYA}}}},
    {"id": 46, "kind": "complex", "followup_of": "How many active residents are in memory care?",
     "q": ["and how many of them are female?", "what about the female ones?", "of those, how many are women?"],
     "plan": {"action": "count", "filters": {**A, "care_type": {"include": ["memory_care"]}, "gender": {"include": ["Female"]}}}},
    # ------------------------------------------------------------------ out of scope
    {"id": 47, "kind": "oos", "q": ["Which residents have diabetes?", "list diabetic residents", "how many residents are diabetic"],
     "plan": {"action": "unsupported"}},
    {"id": 48, "kind": "oos", "q": ["What medications is Eleanor Vance taking?", "Eleanor Vance's medication list", "which medicines does Eleanor Vance take"],
     "plan": {"action": "unsupported"}},
    {"id": 49, "kind": "oos", "q": ["What's the weather today?", "will it rain tomorrow", "tell me a joke"],
     "plan": {"action": "unsupported"}},
    {"id": 50, "kind": "oos", "q": ["What is the billing balance for Harold Finch?", "how much does Harold Finch owe", "Harold Finch's invoice amount"],
     "plan": {"action": "unsupported"}},
]  # fmt: skip

assert len(GOLD) == 50 and all(len(g["q"]) == 3 for g in GOLD)
