"""Named action-sequence metavariables (Java ``MetaActionSequence``)."""

from __future__ import annotations

from typing import Any

from dylan.action.action import Action
from dylan.action.action_sequence import ActionSequence
from dylan.action.meta.element import MetaElement


class MetaActionSequence(ActionSequence):
    """Meta wrapper ``<<A>>`` bound to a concrete :class:`ActionSequence` (Java ``MetaActionSequence``)."""

    def __init__(self, meta_el: MetaElement[ActionSequence]) -> None:
        super().__init__()
        self._meta_el = meta_el

    @staticmethod
    def get(name: str) -> MetaActionSequence:
        """Return the shared meta-sequence for *name* (Java ``MetaActionSequence.get``)."""
        return MetaActionSequence(MetaElement.get(name, ActionSequence))

    def get_meta(self) -> MetaElement[ActionSequence]:
        """Return the underlying :class:`MetaElement` (Java ``getMeta``)."""
        return self._meta_el

    def instantiate(self) -> ActionSequence:
        """Resolve to the bound sequence when set (Java ``MetaActionSequence.instantiate``)."""
        v = self._meta_el.get_value()
        if v is None:
            return self
        return v.instantiate()

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other is None:
            return False
        if isinstance(other, MetaActionSequence):
            return self._meta_el == other._meta_el.get_value()
        if not isinstance(other, ActionSequence):
            return False
        return self._meta_el == other

    def __hash__(self) -> int:
        return hash((MetaActionSequence, self._meta_el.name))


def register_action_sequence(name: str, actions: list[Action]) -> None:
    """Bind meta-sequence *name* to a concrete list of actions (grammar loader hook)."""
    seq = ActionSequence(actions)
    MetaElement.get(name, ActionSequence).set_value(seq)
