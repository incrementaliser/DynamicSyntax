"""Predicate metavariables (Java ``MetaPredicate``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from dylan.formula.predicate_argument import Predicate


@dataclass(frozen=True, slots=True)
class MetaPredicate(Predicate):
    """Predicate metavariable with pooled identity by name."""

    _pool: ClassVar[dict[str, "MetaPredicate"]] = {}

    @classmethod
    def get(cls, name: str) -> "MetaPredicate":
        """Return a pooled predicate metavariable."""
        if name not in cls._pool:
            cls._pool[name] = MetaPredicate(name)
        return cls._pool[name]
