"""Typed placeholders that stand for people in a masked question (vocabulary.EntityIndex.mask).

  resident_N  a resident (always, as before)
  family_N    a family member (familymembers), when no other kind of person has that name
  person_N    a name that matches more than one kind of person (staff and family member, or a resident who is also
              a family member of another resident named in the question); the backend decides which from how the plan
              uses it

The names behind them never leave the server: the model only sees these tokens. This module is the one definition
of their spelling, so every tokenizer and check treats them the same way (one token, never a word plus a number).
"""

import re

RESIDENT = "resident"
FAMILY = "family"
PERSON = "person"
KINDS = (RESIDENT, FAMILY, PERSON)

TOKEN = r"(?:resident|family|person)_\d+"  # for tokenizers: one placeholder = one token
ANY_IN_TEXT = re.compile(r"\b" + TOKEN + r"\b", re.IGNORECASE)
_FULL = re.compile(r"^(resident|family|person)_(\d+)$", re.IGNORECASE)


def kind_of(text: str) -> str | None:
    """'family_2' → 'family'; anything that is not exactly one placeholder → None."""
    m = _FULL.match(str(text).strip())
    return m.group(1).lower() if m else None


def is_placeholder(text: str, kinds: tuple[str, ...] = KINDS) -> bool:
    return kind_of(text) in kinds


def unmask(text: str, originals: dict[str, str]) -> str:
    """Every placeholder → the text the user typed for it (server side only, e.g. a clarification quote)."""
    return ANY_IN_TEXT.sub(lambda m: originals.get(m.group(0).lower(), m.group(0)), text)
