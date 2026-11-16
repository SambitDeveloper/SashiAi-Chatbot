"""Deterministic date and numeric-range grammar (app/nlu/grammar.py): pure parsing, no model, no database."""

from datetime import date

import pytest

from app.nlu.grammar import parse_dates, parse_number_ranges, skeleton

TODAY = date(2026, 11, 13)


def _range(text: str) -> dict | None:
    matches = parse_dates(text, TODAY)
    return matches[0].range if len(matches) == 1 else None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("admitted in the last 3 months", {"start": "2026-08-13", "end": "2026-11-13"}),  # the production example
        ("admitted in the last 90 days", {"start": "2026-08-15", "end": "2026-11-13"}),
        ("admitted in the past 2 weeks", {"start": "2026-10-30", "end": "2026-11-13"}),
        ("admitted this month", {"start": "2026-11-01", "end": "2026-11-30"}),
        ("admitted last month", {"start": "2026-10-01", "end": "2026-10-31"}),  # whole previous calendar month
        ("admitted previous month", {"start": "2026-10-01", "end": "2026-10-31"}),
        ("admitted this year", {"start": "2026-01-01", "end": "2026-12-31"}),
        ("admitted last year", {"start": "2025-01-01", "end": "2025-12-31"}),
        ("admitted in october this year", {"start": "2026-10-01", "end": "2026-10-31"}),
        ("admitted in october", {"start": "2026-10-01", "end": "2026-10-31"}),  # most recent past october
        ("admitted in december", {"start": "2025-12-01", "end": "2025-12-31"}),  # december has not happened in 2026 yet
        ("registered today", {"start": "2026-11-13", "end": "2026-11-13"}),
        ("registered yesterday", {"start": "2026-11-12", "end": "2026-11-12"}),
        ("admitted since 2026-01-01", {"start": "2026-01-01"}),
        ("admitted before 2026-03-01", {"end": "2026-02-28"}),  # "before" excludes the day itself
        ("admitted by 2026-03-01", {"end": "2026-03-01"}),  # "by"/"until" include it
        ("admitted after 2026-03-01", {"start": "2026-03-02"}),  # "after" starts once the day is over
        ("admitted since 2026-03-01", {"start": "2026-03-01"}),
        # day dates written out, in either order, with or without a year
        ("admitted after August 1, 2026", {"start": "2026-08-02"}),
        ("admitted before January 1, 2020", {"end": "2019-12-31"}),
        ("admitted on 1st of june 2026", {"start": "2026-06-01", "end": "2026-06-01"}),
        ("admitted on aug 3", {"start": "2026-08-03", "end": "2026-08-03"}),
        # a period after a comparison word: the whole period is before/after
        ("admission date before 2026", {"end": "2025-12-31"}),
        ("admitted after 2024", {"start": "2025-01-01"}),
        ("admitted after june", {"start": "2026-07-01"}),
        ("admitted until 2025", {"end": "2025-12-31"}),
        ("admitted since march", {"start": "2026-03-01"}),
        ("admitted between 2026-01-01 and 2026-03-31", {"start": "2026-01-01", "end": "2026-03-31"}),
    ],
)
def test_date_expressions(text, expected):
    assert _range(text) == expected


def test_leap_year_month_clamping():
    # "last 1 month" from the 31st lands in a 30-day month without raising.
    assert parse_dates("last 1 month", date(2026, 3, 31))[0].range == {"start": "2026-02-28", "end": "2026-03-31"}


def test_impossible_day_is_not_a_date():
    assert all(d.range.get("start") != "2026-02-30" for d in parse_dates("admitted on february 30 2026", TODAY))


def test_no_date_expression():
    assert parse_dates("active residents in memory care", TODAY) == []


@pytest.mark.parametrize(
    ("text", "bounds"),
    [
        ("over 85", {"min": 86}),
        ("above 85", {"min": 86}),
        ("older than 90", {"min": 91}),
        ("at least 80", {"min": 80}),
        ("80 or older", {"min": 80}),
        ("under 70", {"max": 69}),
        ("younger than 75", {"max": 74}),
        ("at most 90", {"max": 90}),
        ("between 60 and 70", {"min": 60, "max": 70}),
        ("aged 70 to 80", {"min": 70, "max": 80}),
        ("aged 70", {"min": 70, "max": 70}),
        ("70 years old", {"min": 70, "max": 70}),
    ],
)
def test_number_ranges(text, bounds):
    matches = parse_number_ranges(text)
    assert len(matches) == 1 and matches[0].bounds == bounds


def test_skeleton_strips_dates_and_numbers():
    assert skeleton("residents admitted in the last 3 months", TODAY) == "residents admitted in the"
    assert skeleton("residents aged between 60 and 70 in skilled nursing", TODAY) == "residents aged in skilled nursing"
    assert skeleton("active residents in memory care", TODAY) == "active residents in memory care"
