"""Read-only access to app/data/resident_schema.yaml for the parts the filter semantics need.

Both filter engines use this one source: `resident_view.to_mongo_filter` (MongoDB) and `repository.matches`
(in-memory test repository), so "what counts as missing" can never differ between them.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

SCHEMA_FILE = Path(__file__).with_name("resident_schema.yaml")


@lru_cache
def schema_config() -> dict:
    return yaml.safe_load(SCHEMA_FILE.read_text(encoding="utf-8")) or {}


def list_fields() -> frozenset[str]:
    """View fields that hold a list (e.g. all assigned staff names)."""
    return frozenset(schema_config().get("list_fields") or [])


def missing_markers(view_field: str) -> list[Any]:
    """Values meaning "not recorded" for this view field: null, "" and the field's declared placeholder values."""
    declared = (schema_config().get("missing_values") or {}).get(view_field) or []
    return [None, "", *declared]


def is_missing(value: Any, view_field: str) -> bool:
    """True when the field has no real value: absent/None, "", a declared placeholder, or a list with no real value
    (an empty list included). Mirrors the MongoDB expressions built in resident_view._missing_clause."""
    markers = missing_markers(view_field)
    values = value if isinstance(value, list) else [value]
    return all(any(v == m for m in markers) for v in values)


def output_fields() -> dict[str, str]:
    """Plan attribute → view field for every attribute the schema can return, limited to view fields the mapping
    actually defines (an attribute whose field is not mapped is not supported)."""
    cfg = schema_config()
    mapped = set((cfg.get("fields") or {}).keys())
    if (cfg.get("relations") or {}).get("family"):  # fields the family relation derives (app/data/relations.py)
        mapped |= {"familyCount", "familyMembers", "familyContacts"}
    return {a: f for a, f in (cfg.get("output_fields") or {}).items() if f in mapped}


def identity_fields() -> list[str]:
    return list(schema_config().get("identity_fields") or [])


def single_record_only() -> frozenset[str]:
    return frozenset(schema_config().get("single_record_only") or [])


def comparable_fields() -> dict[str, dict]:
    """Plan field → {kind, ...} for fields that can be compared with another record (resident_schema.yaml `comparable`)."""
    return {k: dict(v or {}) for k, v in (schema_config().get("comparable") or {}).items()}
