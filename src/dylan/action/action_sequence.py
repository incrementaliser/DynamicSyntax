"""Ordered list of :class:`Action` (Java ``ActionSequence``)."""

from __future__ import annotations

from dylan.action.action import Action


class ActionSequence(list[Action]):
    """Sequence of actions executed in order (Java ``ActionSequence``)."""

    def instantiate(self) -> ActionSequence:
        """Return a copy with instantiated effects (Java ``ActionSequence.instantiate``)."""
        return ActionSequence([a.instantiate() for a in self])
