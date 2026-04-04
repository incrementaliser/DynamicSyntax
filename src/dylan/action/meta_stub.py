"""Minimal meta-type placeholder until full meta machinery is ported."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from dylan.type.dstype import DSType


@dataclass(frozen=True, slots=True)
class MetaType(DSType):
    """Upper-case type metavariable (Java `MetaType`) — stub."""

    name: str

    _pool: ClassVar[dict[str, MetaType]] = {}

    @classmethod
    def get(cls, name: str) -> MetaType:
        if name not in cls._pool:
            cls._pool[name] = MetaType(name)
        return cls._pool[name]

    def __str__(self) -> str:
        return self.name


class MetaElement:
    """Static reset hook used by IfThenElse (Java `MetaElement`)."""

    @staticmethod
    def reset_bound_metas() -> None:
        """Clear metavar bindings between action applications."""
        return


def reset_bound_metas() -> None:
    MetaElement.reset_bound_metas()
