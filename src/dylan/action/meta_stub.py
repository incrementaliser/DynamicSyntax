"""Meta-type placeholder and pool reset hook (Java ``MetaElement`` / ``MetaType``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from dylan.action.meta.element import MetaElement, reset_meta_element_pool
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


def reset_bound_metas() -> None:
    """Clear metavariable bindings between action applications (Java ``MetaElement.resetBoundMetas``)."""
    reset_meta_element_pool()


# Back-compat alias
MetaElementStub = MetaElement
