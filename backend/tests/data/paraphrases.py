"""Paraphrase set: one NEW wording per benchmark question (gold + eval50), to test generalisation to unseen wording.

Each entry reuses the source item's expected plan unchanged (ids say which: "gold-14" = GOLD id 14). Paraphrases were
written by one agent per batch and adversarially checked by a second (meaning, conditions, negation, date range relative
to 2026-11-09, names, follow-up reference); rejected ones are left out. Today is pinned to 2026-11-09 as in the
source sets.
"""

PARA: list[dict] = [
    {
        "id": "gold-1",
        "kind": "simple",
        "q": ["What's our current headcount of residents with an active status?"],
        "plan": {"action": "count", "filters": {"status": {"include": ["active"]}}},
    },
    {
        "id": "gold-2",
        "kind": "simple",
        "q": ["How many of our residents are women?"],
        "plan": {"action": "count", "filters": {"gender": {"include": ["Female"]}}},
    },
    {
        "id": "gold-3",
        "kind": "simple",
        "q": ["Give me a headcount of the residents receiving memory care."],
        "plan": {"action": "count", "filters": {"care_type": {"include": ["memory_care"]}}},
    },
    {
        "id": "gold-4",
        "kind": "simple",
        "q": ["Pull up all the residents who have a discharged status."],
        "plan": {"action": "list", "filters": {"status": {"include": ["discharged"]}}},
    },
    {
        "id": "gold-5",
        "kind": "simple",
        "q": ["Last month, how many new resident admissions did we have?"],
        "plan": {"action": "count", "filters": {"admission_date": {"start": "2026-10-01", "end": "2026-10-31"}}},
    },
    {
        "id": "gold-6",
        "kind": "simple",
        "q": ["On average, how old are our residents?"],
        "plan": {"action": "aggregate", "aggregate": {"op": "avg", "field": "age"}},
    },
    {
        "id": "gold-7",
        "kind": "simple",
        "q": ["What's the resident headcount over in the North Wing?"],
        "plan": {"action": "count", "filters": {"wing": {"include": ["North Wing"]}}},
    },
    {
        "id": "gold-8",
        "kind": "simple",
        "q": ["Show me how our residents are spread across the different statuses."],
        "plan": {"action": "breakdown", "group_by": ["status"]},
    },
    {
        "id": "gold-9",
        "kind": "simple",
        "q": ["Break down the resident count wing by wing for me."],
        "plan": {"action": "breakdown", "group_by": ["wing"]},
    },
    {
        "id": "gold-10",
        "kind": "simple",
        "q": ["What's each caregiver's resident caseload?"],
        "plan": {"action": "breakdown", "group_by": ["primary_caregiver"]},
    },
    {
        "id": "gold-11",
        "kind": "simple",
        "q": ["Which resident has the highest age?"],
        "plan": {"action": "list", "sort": {"field": "age", "direction": "desc"}, "limit": 1},
    },
    {
        "id": "gold-12",
        "kind": "simple",
        "q": ["Among the independent living residents, who has the lowest age?"],
        "plan": {
            "action": "list",
            "filters": {"care_type": {"include": ["independent_living"]}},
            "sort": {"field": "age", "direction": "asc"},
            "limit": 1,
        },
    },
    {
        "id": "gold-13",
        "kind": "simple",
        "q": ["Give me the residents in assisted living whose status is active."],
        "plan": {
            "action": "list",
            "filters": {"status": {"include": ["active"]}, "care_type": {"include": ["assisted_living"]}},
        },
    },
    {
        "id": "gold-14",
        "kind": "simple",
        "q": ["Of the residents Elena Rostova is primary caregiver for, how many are active?"],
        "plan": {
            "action": "count",
            "filters": {
                "status": {"include": ["active"]},
                "primary_caregiver": {"include": ["Care Aide Elena Rostova"]},
            },
        },
    },
    {
        "id": "gold-15",
        "kind": "simple",
        "q": ["Give me a tally of the residents marked as pending."],
        "plan": {"action": "count", "filters": {"status": {"include": ["pending"]}}},
    },
    {
        "id": "gold-16",
        "kind": "simple",
        "q": ["Show me how many residents there are of each gender."],
        "plan": {"action": "breakdown", "group_by": ["gender"]},
    },
    {
        "id": "gold-17",
        "kind": "simple",
        "q": ["For residents in memory care, how old are they on average?"],
        "plan": {
            "action": "aggregate",
            "aggregate": {"op": "avg", "field": "age"},
            "filters": {"care_type": {"include": ["memory_care"]}},
        },
    },
    {
        "id": "gold-18",
        "kind": "simple",
        "q": ["Pull up Eleanor Vance's record for me."],
        "plan": {"action": "lookup", "lookup": {"residents": ["RES-1001"]}},
    },
    {
        "id": "gold-19",
        "kind": "simple",
        "q": ["How many residents have had a discharge during this calendar year?"],
        "plan": {
            "action": "count",
            "filters": {
                "status": {"include": ["discharged"]},
                "discharge_date": {"start": "2026-01-01", "end": "2026-12-31"},
            },
        },
    },
    {"id": "gold-20", "kind": "simple", "q": ["What's our overall resident headcount?"], "plan": {"action": "count"}},
    {
        "id": "gold-21",
        "kind": "simple",
        "q": ["Who are the residents with admission dates in the past three months?"],
        "plan": {"action": "list", "filters": {"admission_date": {"start": "2026-08-09", "end": "2026-11-09"}}},
    },
    {"id": "gold-22", "kind": "simple", "q": ["hey there, how can you help me?"], "plan": {"action": "help"}},
    {
        "id": "gold-23",
        "kind": "simple",
        "q": ["Of this year's admissions, how many are skilled nursing residents?"],
        "plan": {
            "action": "count",
            "filters": {
                "care_type": {"include": ["skilled_nursing"]},
                "admission_date": {"start": "2026-01-01", "end": "2026-12-31"},
            },
        },
    },
    {
        "id": "gold-24",
        "kind": "complex",
        "q": ["What's the count of residents whose status is anything other than active?"],
        "plan": {"action": "count", "filters": {"status": {"exclude": ["active"]}}},
    },
    {
        "id": "gold-25",
        "kind": "complex",
        "q": ["What's the count of residents who are more than 80 years old?"],
        "plan": {"action": "count", "filters": {"age": {"min": 81}}},
    },
    {
        "id": "gold-26",
        "kind": "complex",
        "q": ["who are our residents in the 70 to 80 age range?"],
        "plan": {"action": "list", "filters": {"age": {"min": 70, "max": 80}}},
    },
    {
        "id": "gold-27",
        "kind": "complex",
        "q": ["which three residents are the eldest?"],
        "plan": {"action": "list", "sort": {"field": "age", "direction": "desc"}, "limit": 3},
    },
    {
        "id": "gold-28",
        "kind": "complex",
        "q": ["Can you check whether Arthur Pendleton is on memory care?"],
        "plan": {
            "action": "lookup",
            "lookup": {"residents": ["RES-1002"]},
            "filters": {"care_type": {"include": ["memory_care"]}},
        },
    },
    {
        "id": "gold-29",
        "kind": "complex",
        "q": ["Which room has Eleanor Vance been assigned?"],
        "plan": {"action": "lookup", "lookup": {"residents": ["RES-1001"]}},
    },
    {
        "id": "gold-30",
        "kind": "complex",
        "q": ["Who's assigned as Robert MacKenzie's main carer?"],
        "plan": {"action": "lookup", "lookup": {"residents": ["RES-1006"]}},
    },
    {
        "id": "gold-31",
        "kind": "complex",
        "q": ["Of our active residents, how many are either memory care or skilled nursing?"],
        "plan": {
            "action": "count",
            "filters": {
                "status": {"include": ["active"]},
                "care_type": {"include": ["memory_care", "skilled_nursing"]},
            },
        },
    },
    {
        "id": "gold-32",
        "kind": "complex",
        "q": ["For active residents, give me the counts grouped by care type and then by wing."],
        "plan": {
            "action": "breakdown",
            "group_by": ["care_type", "wing"],
            "filters": {"status": {"include": ["active"]}},
        },
    },
    {
        "id": "gold-33",
        "kind": "complex",
        "q": ["Give me the average resident age broken down by wing."],
        "plan": {"action": "aggregate", "aggregate": {"op": "avg", "field": "age"}, "group_by": ["wing"]},
    },
    {
        "id": "gold-34",
        "kind": "complex",
        "q": ["Among the male residents of the South Wing, how many are above the age of 85?"],
        "plan": {
            "action": "count",
            "filters": {"gender": {"include": ["Male"]}, "wing": {"include": ["South Wing"]}, "age": {"min": 86}},
        },
    },
    {
        "id": "gold-35",
        "kind": "complex",
        "q": ["Give me the memory care residents who live anywhere other than Garden Terrace."],
        "plan": {
            "action": "list",
            "filters": {"care_type": {"include": ["memory_care"]}, "wing": {"exclude": ["Garden Terrace"]}},
        },
    },
    {
        "id": "gold-36",
        "kind": "complex",
        "q": ["Among Elena Rostova's active residents, how many had their admission last month?"],
        "plan": {
            "action": "count",
            "filters": {
                "status": {"include": ["active"]},
                "admission_date": {"start": "2026-10-01", "end": "2026-10-31"},
                "primary_caregiver": {"include": ["Care Aide Elena Rostova"]},
            },
        },
    },
    {
        "id": "gold-37",
        "kind": "complex",
        "q": ["Which resident has the newest admission date?"],
        "plan": {"action": "list", "sort": {"field": "admission_date", "direction": "desc"}, "limit": 1},
    },
    {
        "id": "gold-38",
        "kind": "complex",
        "q": ["Among our assisted living residents, which five were admitted most recently?"],
        "plan": {
            "action": "list",
            "filters": {"care_type": {"include": ["assisted_living"]}},
            "sort": {"field": "admission_date", "direction": "desc"},
            "limit": 5,
        },
    },
    {
        "id": "gold-39",
        "kind": "complex",
        "q": ["Of Nurse Sarah Jenkins's residents, how many are women?"],
        "plan": {
            "action": "count",
            "filters": {
                "gender": {"include": ["Female"]},
                "primary_caregiver": {"include": ["Nurse Sarah Jenkins, RN"]},
            },
        },
    },
    {
        "id": "gold-40",
        "kind": "complex",
        "q": ["How old is the youngest person we have in respite care?"],
        "plan": {
            "action": "aggregate",
            "aggregate": {"op": "min", "field": "age"},
            "filters": {"care_type": {"include": ["respite_care"]}},
        },
    },
    {
        "id": "gold-41",
        "kind": "complex",
        "q": ["How many admissions have we had from March onwards?"],
        "plan": {"action": "count", "filters": {"admission_date": {"start": "2026-03-01"}}},
    },
    {
        "id": "gold-42",
        "kind": "complex",
        "q": ["Show me everyone whose admission date falls in 2025."],
        "plan": {"action": "list", "filters": {"admission_date": {"start": "2025-01-01", "end": "2025-12-31"}}},
    },
    {
        "id": "gold-43",
        "kind": "complex",
        "q": ["In the East Wing, how many residents have a status of pending or admitted?"],
        "plan": {
            "action": "count",
            "filters": {"wing": {"include": ["East Wing"]}, "status": {"include": ["pending", "admitted"]}},
        },
    },
    {
        "id": "gold-44",
        "kind": "complex",
        "q": ["Can I see Eleanor Vance and Arthur Pendleton side by side?"],
        "plan": {"action": "lookup", "lookup": {"residents": ["RES-1001", "RES-1002"]}},
    },
    {
        "id": "gold-45",
        "kind": "complex",
        "q": ["How many residents are on Maya Patel's caseload?"],
        "plan": {
            "action": "count",
            "filters": {
                "primary_caregiver": {"include": ["Wellness Coordinator Maya Patel", "Care Coordinator Maya Patel"]}
            },
        },
    },
    {
        "id": "gold-46",
        "kind": "complex",
        "q": ["Narrowing those down to women only, what's the number?"],
        "plan": {
            "action": "count",
            "filters": {
                "status": {"include": ["active"]},
                "care_type": {"include": ["memory_care"]},
                "gender": {"include": ["Female"]},
            },
        },
        "followup_of": "How many active residents are in memory care?",
    },
    {
        "id": "gold-47",
        "kind": "oos",
        "q": ["Do we have anyone with a diabetes diagnosis?"],
        "plan": {"action": "unsupported"},
    },
    {
        "id": "gold-48",
        "kind": "oos",
        "q": ["Can you pull up the meds Eleanor Vance is prescribed?"],
        "plan": {"action": "unsupported"},
    },
    {
        "id": "gold-49",
        "kind": "oos",
        "q": ["What's the forecast looking like for this afternoon?"],
        "plan": {"action": "unsupported"},
    },
    {
        "id": "gold-50",
        "kind": "oos",
        "q": ["Is there an unpaid balance on Harold Finch's account?"],
        "plan": {"action": "unsupported"},
    },
    {
        "id": "eval50-101",
        "kind": "simple",
        "q": ["Give me a count of residents whose status is inactive."],
        "plan": {"action": "count", "filters": {"status": {"include": ["inactive"]}}},
    },
    {
        "id": "eval50-102",
        "kind": "simple",
        "q": ["What number of our residents are men?"],
        "plan": {"action": "count", "filters": {"gender": {"include": ["Male"]}}},
    },
    {
        "id": "eval50-103",
        "kind": "simple",
        "q": ["How many residents receive skilled nursing care?"],
        "plan": {"action": "count", "filters": {"care_type": {"include": ["skilled_nursing"]}}},
    },
    {
        "id": "eval50-104",
        "kind": "simple",
        "q": ["Which residents have a pending status?"],
        "plan": {"action": "list", "filters": {"status": {"include": ["pending"]}}},
    },
    {
        "id": "eval50-105",
        "kind": "simple",
        "q": ["What's the resident count for the East Wing?"],
        "plan": {"action": "count", "filters": {"wing": {"include": ["East Wing"]}}},
    },
    {
        "id": "eval50-106",
        "kind": "simple",
        "q": ["Which residents are with us on respite care?"],
        "plan": {"action": "list", "filters": {"care_type": {"include": ["respite_care"]}}},
    },
    {
        "id": "eval50-107",
        "kind": "simple",
        "q": ["Can you split all the residents up by their type of care?"],
        "plan": {"action": "breakdown", "group_by": ["care_type"]},
    },
    {
        "id": "eval50-108",
        "kind": "simple",
        "q": ["What's the number of residents who have Marcus Chen as their primary caregiver?"],
        "plan": {"action": "count", "filters": {"primary_caregiver": {"include": ["Care Aide Marcus Chen"]}}},
    },
    {
        "id": "eval50-109",
        "kind": "simple",
        "q": ["Among the residents with active status, what's the mean age?"],
        "plan": {
            "action": "aggregate",
            "aggregate": {"op": "avg", "field": "age"},
            "filters": {"status": {"include": ["active"]}},
        },
    },
    {
        "id": "eval50-110",
        "kind": "simple",
        "q": ["Which of our residents is the youngest?"],
        "plan": {"action": "list", "sort": {"field": "age", "direction": "asc"}, "limit": 1},
    },
    {
        "id": "eval50-111",
        "kind": "simple",
        "q": ["What number of residents have an admission date falling in this month?"],
        "plan": {"action": "count", "filters": {"admission_date": {"start": "2026-11-01", "end": "2026-11-30"}}},
    },
    {
        "id": "eval50-112",
        "kind": "simple",
        "q": ["In assisted living, how many of the residents are women?"],
        "plan": {
            "action": "count",
            "filters": {"gender": {"include": ["Female"]}, "care_type": {"include": ["assisted_living"]}},
        },
    },
    {
        "id": "eval50-113",
        "kind": "simple",
        "q": ["Which residents are assigned to the South Wing?"],
        "plan": {"action": "list", "filters": {"wing": {"include": ["South Wing"]}}},
    },
    {
        "id": "eval50-114",
        "kind": "simple",
        "q": ["Can you pull up the details on Harold Finch?"],
        "plan": {"action": "lookup", "lookup": {"residents": ["RES-1004"]}},
    },
    {
        "id": "eval50-115",
        "kind": "simple",
        "q": ["What's the count of residents on independent living?"],
        "plan": {"action": "count", "filters": {"care_type": {"include": ["independent_living"]}}},
    },
    {
        "id": "eval50-116",
        "kind": "simple",
        "q": ["For the North Wing, break the residents down by care type."],
        "plan": {"action": "breakdown", "group_by": ["care_type"], "filters": {"wing": {"include": ["North Wing"]}}},
    },
    {
        "id": "eval50-117",
        "kind": "simple",
        "q": ["In Garden Terrace, what's the number of residents with an active status?"],
        "plan": {
            "action": "count",
            "filters": {"status": {"include": ["active"]}, "wing": {"include": ["Garden Terrace"]}},
        },
    },
    {
        "id": "eval50-118",
        "kind": "simple",
        "q": ["Which residents have David O'Connor as their main caregiver?"],
        "plan": {"action": "list", "filters": {"primary_caregiver": {"include": ["Head Nurse David O'Connor, RN"]}}},
    },
    {
        "id": "eval50-119",
        "kind": "simple",
        "q": ["During October, how many new residents did we take in?"],
        "plan": {"action": "count", "filters": {"admission_date": {"start": "2026-10-01", "end": "2026-10-31"}}},
    },
    {
        "id": "eval50-120",
        "kind": "simple",
        "q": ["Among the memory care residents, who is the eldest?"],
        "plan": {
            "action": "list",
            "filters": {"care_type": {"include": ["memory_care"]}},
            "sort": {"field": "age", "direction": "desc"},
            "limit": 1,
        },
    },
    {
        "id": "eval50-121",
        "kind": "simple",
        "q": ["Altogether, how many residents live in the North Wing?"],
        "plan": {"action": "count", "filters": {"wing": {"include": ["North Wing"]}}},
    },
    {
        "id": "eval50-122",
        "kind": "simple",
        "q": ["What's the count of residents who have been discharged?"],
        "plan": {"action": "count", "filters": {"status": {"include": ["discharged"]}}},
    },
    {
        "id": "eval50-123",
        "kind": "simple",
        "q": ["For our female residents, what's the mean age?"],
        "plan": {
            "action": "aggregate",
            "aggregate": {"op": "avg", "field": "age"},
            "filters": {"gender": {"include": ["Female"]}},
        },
    },
    {"id": "eval50-124", "kind": "simple", "q": ["Great, thank you so much!"], "plan": {"action": "help"}},
    {
        "id": "eval50-125",
        "kind": "simple",
        "q": ["For how many residents is Liam Gallagher the primary caregiver?"],
        "plan": {"action": "count", "filters": {"primary_caregiver": {"include": ["Nurse Liam Gallagher, LPN"]}}},
    },
    {
        "id": "eval50-201",
        "kind": "complex",
        "q": ["Give me a count of Garden Terrace memory care residents whose status is anything other than active."],
        "plan": {
            "action": "count",
            "filters": {
                "care_type": {"include": ["memory_care"]},
                "wing": {"include": ["Garden Terrace"]},
                "status": {"exclude": ["active"]},
            },
        },
    },
    {
        "id": "eval50-202",
        "kind": "complex",
        "q": ["Show me the women in assisted living who are older than 85."],
        "plan": {
            "action": "list",
            "filters": {
                "gender": {"include": ["Female"]},
                "age": {"min": 86},
                "care_type": {"include": ["assisted_living"]},
            },
        },
    },
    {
        "id": "eval50-203",
        "kind": "complex",
        "q": ["What's the combined total of residents aged 75 to 85 in skilled nursing and memory care together?"],
        "plan": {
            "action": "count",
            "filters": {"age": {"min": 75, "max": 85}, "care_type": {"include": ["skilled_nursing", "memory_care"]}},
        },
    },
    {
        "id": "eval50-204",
        "kind": "complex",
        "q": ["Show me the men in the North Wing who've been admitted over the past six months."],
        "plan": {
            "action": "list",
            "filters": {
                "gender": {"include": ["Male"]},
                "wing": {"include": ["North Wing"]},
                "admission_date": {"start": "2026-05-09", "end": "2026-11-09"},
            },
        },
    },
    {
        "id": "eval50-205",
        "kind": "complex",
        "q": ["Count the memory care residents whose primary caregiver is someone other than Sarah Jenkins."],
        "plan": {
            "action": "count",
            "filters": {
                "care_type": {"include": ["memory_care"]},
                "primary_caregiver": {"exclude": ["Nurse Sarah Jenkins, RN"]},
            },
        },
    },
    {
        "id": "eval50-206",
        "kind": "complex",
        "q": ["How are our female residents spread across the different wings?"],
        "plan": {"action": "breakdown", "group_by": ["wing"], "filters": {"gender": {"include": ["Female"]}}},
    },
    {
        "id": "eval50-207",
        "kind": "complex",
        "q": ["For active residents only, what's the mean age in each care type?"],
        "plan": {
            "action": "aggregate",
            "aggregate": {"op": "avg", "field": "age"},
            "group_by": ["care_type"],
            "filters": {"status": {"include": ["active"]}},
        },
    },
    {
        "id": "eval50-208",
        "kind": "complex",
        "q": ["Who are the five oldest residents with an active status?"],
        "plan": {
            "action": "list",
            "filters": {"status": {"include": ["active"]}},
            "sort": {"field": "age", "direction": "desc"},
            "limit": 5,
        },
    },
    {
        "id": "eval50-209",
        "kind": "complex",
        "q": ["Count the residents younger than 75 in either the East Wing or the South Wing."],
        "plan": {"action": "count", "filters": {"wing": {"include": ["East Wing", "South Wing"]}, "age": {"max": 74}}},
    },
    {
        "id": "eval50-210",
        "kind": "complex",
        "q": ["Can you check whether Eleanor Vance has an active status and is placed in memory care?"],
        "plan": {
            "action": "lookup",
            "lookup": {"residents": ["RES-1001"]},
            "filters": {"status": {"include": ["active"]}, "care_type": {"include": ["memory_care"]}},
        },
    },
    {
        "id": "eval50-211",
        "kind": "complex",
        "q": ["Which residents admitted from August onwards don't have an active status?"],
        "plan": {
            "action": "list",
            "filters": {"admission_date": {"start": "2026-08-01"}, "status": {"exclude": ["active"]}},
        },
    },
    {
        "id": "eval50-212",
        "kind": "complex",
        "q": ["What's the number of assisted living residents discharged in the current year?"],
        "plan": {
            "action": "count",
            "filters": {
                "status": {"include": ["discharged"]},
                "discharge_date": {"start": "2026-01-01", "end": "2026-12-31"},
                "care_type": {"include": ["assisted_living"]},
            },
        },
    },
    {
        "id": "eval50-213",
        "kind": "complex",
        "q": ["What's the highest age among South Wing residents?"],
        "plan": {
            "action": "aggregate",
            "aggregate": {"op": "max", "field": "age"},
            "filters": {"wing": {"include": ["South Wing"]}},
        },
    },
    {
        "id": "eval50-214",
        "kind": "complex",
        "q": ["Show me the memory care resident count for every caregiver."],
        "plan": {
            "action": "breakdown",
            "group_by": ["primary_caregiver"],
            "filters": {"care_type": {"include": ["memory_care"]}},
        },
    },
    {
        "id": "eval50-215",
        "kind": "complex",
        "q": ["How many men are in independent living outside of the North Wing?"],
        "plan": {
            "action": "count",
            "filters": {
                "gender": {"include": ["Male"]},
                "care_type": {"include": ["independent_living"]},
                "wing": {"exclude": ["North Wing"]},
            },
        },
    },
    {
        "id": "eval50-216",
        "kind": "complex",
        "q": ["Show skilled nursing residents whose status is either pending or admitted."],
        "plan": {
            "action": "list",
            "filters": {"status": {"include": ["pending", "admitted"]}, "care_type": {"include": ["skilled_nursing"]}},
        },
    },
    {
        "id": "eval50-217",
        "kind": "complex",
        "q": ["Show me the three newest admissions to the East Wing."],
        "plan": {
            "action": "list",
            "filters": {"wing": {"include": ["East Wing"]}},
            "sort": {"field": "admission_date", "direction": "desc"},
            "limit": 3,
        },
    },
    {
        "id": "eval50-218",
        "kind": "complex",
        "q": ["Which residents are at least 90 years old?"],
        "plan": {"action": "list", "filters": {"age": {"min": 90}}},
    },
    {
        "id": "eval50-219",
        "kind": "complex",
        "q": ["Of the active residents that Maya Patel is the primary caregiver for, how many are women?"],
        "plan": {
            "action": "count",
            "filters": {
                "status": {"include": ["active"]},
                "primary_caregiver": {"include": ["Wellness Coordinator Maya Patel", "Care Coordinator Maya Patel"]},
                "gender": {"include": ["Female"]},
            },
        },
    },
    {
        "id": "eval50-220",
        "kind": "complex",
        "q": ["Split the respite care residents up by their status."],
        "plan": {
            "action": "breakdown",
            "group_by": ["status"],
            "filters": {"care_type": {"include": ["respite_care"]}},
        },
    },
    {
        "id": "eval50-221",
        "kind": "complex",
        "q": ["For residents admitted in the current year, what's their average age?"],
        "plan": {
            "action": "aggregate",
            "aggregate": {"op": "avg", "field": "age"},
            "filters": {"admission_date": {"start": "2026-01-01", "end": "2026-12-31"}},
        },
    },
    {
        "id": "eval50-222",
        "kind": "complex",
        "q": ["Of those, what's the number of men?"],
        "plan": {
            "action": "count",
            "filters": {
                "status": {"include": ["active"]},
                "care_type": {"include": ["assisted_living"]},
                "gender": {"include": ["Male"]},
            },
        },
        "followup_of": "How many active residents are in assisted living?",
    },
    {
        "id": "eval50-223",
        "kind": "complex",
        "q": ["Narrow those down to just the memory care residents."],
        "plan": {
            "action": "list",
            "filters": {"wing": {"include": ["Garden Terrace"]}, "care_type": {"include": ["memory_care"]}},
        },
        "followup_of": "list residents in Garden Terrace",
    },
    {
        "id": "eval50-224",
        "kind": "complex",
        "q": ["Count North Wing residents whose status is something other than active or pending."],
        "plan": {
            "action": "count",
            "filters": {"wing": {"include": ["North Wing"]}, "status": {"exclude": ["active", "pending"]}},
        },
    },
    {
        "id": "eval50-225",
        "kind": "complex",
        "q": ["Put Harold Finch and Walter Bishop side by side for me."],
        "plan": {"action": "lookup", "lookup": {"residents": ["RES-1004", "RES-1008"]}},
    },
]
