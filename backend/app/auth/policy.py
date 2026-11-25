"""The access policy (access_policy.yaml): capabilities, the modules they need, the fields they cover, the pre-model
gate's word rules and the user-facing messages. Loaded once; everything else in app/auth reads it from here."""

import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DIR = Path(__file__).resolve().parent
_DATA = _DIR.parent / "data"
_SOURCES = {"schema": _DATA / "resident_schema.yaml", "routes": _DATA / "intent_routes.yaml"}


def norm_module(name: Any) -> str:
    """Module names as compared: case, hyphens, underscores and spacing do not matter ("transport-calendar")."""
    return " ".join(str(name or "").replace("-", " ").replace("_", " ").lower().split())


@lru_cache
def _source(name: str) -> dict:
    return yaml.safe_load(_SOURCES[name].read_text(encoding="utf-8")) or {}


def resolve_ref(value: Any) -> Any:
    """"@schema:relations.staff.cue_words" → that value of resident_schema.yaml (routes = intent_routes.yaml)."""
    if not (isinstance(value, str) and value.startswith("@")):
        return value
    source, _, path = value[1:].partition(":")
    node: Any = _source(source)
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"access_policy.yaml: {value} does not exist")
        node = node[part]
    return node


def _words(entries) -> list[str]:
    """Word lists; a referenced mapping contributes its keys and values ("roles": role names and designations)."""
    out: list[str] = []
    for e in entries or []:
        v = resolve_ref(e)
        if isinstance(v, dict):
            v = [*v.keys(), *(x for vals in v.values() for x in (vals if isinstance(vals, list) else [vals]))]
        out.extend(str(w).replace("_", " ") for w in (v if isinstance(v, list) else [v]) if w)
    return out


@dataclass(frozen=True)
class Capability:
    name: str
    label: str
    modules: tuple[str, ...]  # normalised
    fields: frozenset[str]
    base_of: str | None
    actions: frozenset[str] = frozenset()  # plan actions that need this capability (resident plans)


@dataclass
class GateRule:
    any_of: tuple[str, ...]
    pattern: re.Pattern | None
    names_residents: bool = False
    app_sections: bool = False
    unless: re.Pattern | None = None  # the rule does not apply when these words occur
    unless_names_resident: bool = False  # ... or when the question names a resident


@dataclass
class AccessPolicy:
    raw: dict
    capabilities: dict[str, Capability]
    gate: list[GateRule]
    messages: dict[str, str]
    allowed_when: tuple[str, ...]
    principal: dict = field(default_factory=dict)

    @property
    def version(self) -> str:
        """Changes whenever the policy file changes: part of every principal's policy_version."""
        return hashlib.sha256(repr(self.raw).encode()).hexdigest()[:12]

    def capability_for_field(self, name: str, entity: str) -> str:
        for cap in self.capabilities.values():
            if name in cap.fields:
                return cap.name
        return self.base_capability(entity)

    def capabilities_for_action(self, action: str) -> list[str]:
        return [c.name for c in self.capabilities.values() if action in c.actions]

    def base_capability(self, entity: str) -> str:
        for cap in self.capabilities.values():
            if cap.base_of == entity:
                return cap.name
        raise ValueError(f"access_policy.yaml: no capability is base_of {entity}")

    def message(self, key: str, **values) -> str:
        return self.messages[key].format(**values)

    def setting(self, key: str, default=None):
        return resolve_ref(self.principal.get(key, default))


def _pattern(words: list[str]) -> re.Pattern | None:
    words = sorted({w.lower() for w in words}, key=len, reverse=True)
    # a simple plural matches too ("physicians", "nurses")
    return re.compile(r"(?<![a-z])(?:" + "|".join(re.escape(w) for w in words) + r")(?:s|es)?(?![a-z])") if words else None


def load_policy(path: Path | None = None) -> AccessPolicy:
    raw = yaml.safe_load((path or _DIR / "access_policy.yaml").read_text(encoding="utf-8")) or {}
    caps = {}
    for name, c in (raw.get("capabilities") or {}).items():
        caps[name] = Capability(
            name=name, label=str(c.get("label") or name), modules=tuple(norm_module(m) for m in c.get("modules") or []),
            fields=frozenset(c.get("fields") or []), base_of=c.get("base_of"), actions=frozenset(c.get("actions") or []),
        )  # fmt: skip
    gate = []
    for rule in raw.get("gate") or []:
        unknown = [n for n in rule.get("any_of") or [] if n not in caps]
        if unknown:
            raise ValueError(f"access_policy.yaml: gate names unknown capabilities {unknown}")
        gate.append(GateRule(tuple(rule["any_of"]), _pattern(_words(rule.get("words"))), bool(rule.get("names_residents")),
                             bool(rule.get("app_sections")), _pattern(_words(rule.get("unless_words"))),
                             bool(rule.get("unless_names_resident"))))  # fmt: skip
    return AccessPolicy(
        raw=raw, capabilities=caps, gate=gate, messages=dict(raw.get("messages") or {}),
        allowed_when=tuple((raw.get("modules") or {}).get("allowed_when") or ["allowed", "isRead"]),
        principal=dict(raw.get("principal") or {}),
    )  # fmt: skip


@lru_cache
def access_policy() -> AccessPolicy:
    return load_policy()
