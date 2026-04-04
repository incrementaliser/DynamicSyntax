"""Computational action (Java `ComputationalAction`)."""

from __future__ import annotations

from dylan.action.action import Action
from dylan.action.atomic.effect import Effect
from dylan.action.atomic.effect_factory import EffectFactory


class ComputationalAction(Action):
    """Grammar rule from ``computational-actions.txt``."""

    HYP_ADJUNCTION_PREFIX = "hyp-adj"

    def __init__(
        self,
        name: str,
        lines: list[str],
        always_good: bool = False,
        backtrack_on_success: bool = False,
    ) -> None:
        self._source_lines = list(lines)
        eff = EffectFactory.create_lines(lines)
        super().__init__(name, eff, backtrack_on_success)
        self.always_good = always_good

    def is_always_good(self) -> bool:
        return self.always_good

    def set_always_good(self, v: bool) -> None:
        self.always_good = v

    def instantiate(self) -> ComputationalAction:
        """Return a fresh copy; hyp-adj / link names skip full effect instantiation (Java parity)."""
        if self.name.startswith(self.HYP_ADJUNCTION_PREFIX) or self.name.startswith("link"):
            assert self.effect is not None
            return ComputationalAction(
                self.name, self._source_lines, self.always_good, self.backtrack_on_success
            )
        assert self.effect is not None
        _ = self.effect.instantiate()
        return ComputationalAction(
            self.name, self._source_lines, self.always_good, self.backtrack_on_success
        )

    def __lt__(self, other: ComputationalAction) -> bool:
        if self.always_good and not other.always_good:
            return True
        if not self.always_good and other.always_good:
            return False
        return id(self) < id(other)
