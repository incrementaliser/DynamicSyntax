"""Java-aligned ``MetaType`` wrapping ``MetaElement[DSType]`` (``qmul.ds.action.meta.MetaType``)."""

from __future__ import annotations

from typing import Any

from dylan.action.meta.element import MetaElement, reset_meta_element_pool
from dylan.type.dstype import DSType


class MetaType(DSType):
    """Upper-case type metavariable (e.g. ``X`` in ``?ty(X)``); equality binds via ``MetaElement``."""

    __slots__ = ("_meta",)

    def __init__(self, meta: MetaElement[Any]) -> None:
        super().__init__()
        self._meta = meta

    @classmethod
    def get(cls, name: str) -> MetaType:
        """Return a new wrapper around the pooled ``MetaElement`` for *name* (Java ``MetaType.get``)."""
        return cls(MetaElement.get(name, DSType))

    def get_value(self) -> DSType | None:
        """Bound ``DSType`` if any (Java ``MetaType.getValue``)."""
        return self._meta.get_value()

    def get_meta(self) -> MetaElement[Any]:
        """Underlying meta cell (Java ``MetaType.getMeta``)."""
        return self._meta

    def instantiate(self) -> DSType:
        """Instantiate bound value or return self if still unbound (Java ``MetaType.instantiate``)."""
        v = self._meta.get_value()
        if v is None:
            return self
        return v.instantiate()

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other is None:
            return False
        if not isinstance(other, DSType):
            return False
        if isinstance(other, MetaType):
            ov = other._meta.get_value()
            return self._meta == ov
        return self._meta == other

    def __hash__(self) -> int:
        """Stable hash by metavariable name only (bound value must not affect hash)."""
        return hash((MetaType, self._meta.name))

    def __str__(self) -> str:
        return str(self._meta)


def reset_bound_metas() -> None:
    """Clear metavariable bindings between action applications (Java ``MetaElement.resetBoundMetas``)."""
    reset_meta_element_pool()


MetaElementStub = MetaElement
