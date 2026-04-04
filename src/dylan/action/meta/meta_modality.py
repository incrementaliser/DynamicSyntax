"""Modality metavariable (Java ``MetaModality``)."""

from __future__ import annotations

from typing import Any

from dylan.action.meta.element import MetaElement
from dylan.tree.modality import EXIST_LEFT, EXIST_RIGHT, Modality


class MetaModality(Modality):
    """Modality meta ``<Z>`` bound via side-effecting equality (Java ``MetaModality``)."""

    def __init__(self, meta_el: MetaElement[Modality]) -> None:
        super().__init__([], required=False)
        self._meta_el = meta_el

    @staticmethod
    def get(name: str) -> MetaModality:
        """Return the shared meta-modality for *name* (Java ``MetaModality.get``)."""
        return MetaModality(MetaElement.get(name, Modality))

    def instantiate(self) -> Modality:
        """Resolve to the bound modality when set (Java ``MetaModality.instantiate``)."""
        v = self._meta_el.get_value()
        if v is None:
            return self
        return v.instantiate()

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other is None:
            return False
        if isinstance(other, MetaModality):
            return self._meta_el == other._meta_el.get_value()
        if not isinstance(other, Modality):
            return False
        return self._meta_el == other

    def __hash__(self) -> int:
        return hash((MetaModality, self._meta_el.name))

    def __str__(self) -> str:
        return f"{EXIST_LEFT}{self._meta_el!s}{EXIST_RIGHT}"

    def get_meta(self) -> MetaElement[Modality]:
        """Return the underlying :class:`MetaElement` (Java ``getMeta``)."""
        return self._meta_el
