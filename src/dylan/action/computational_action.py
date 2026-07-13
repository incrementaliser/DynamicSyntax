"""Computational action (Java `ComputationalAction`)."""

from __future__ import annotations

from typing import Any

from dylan.action.action import Action
from dylan.action.atomic.effect import Effect
from dylan.action.atomic.effect_factory import EffectFactory
from dylan.tree.tree import Tree


class ComputationalAction(Action):
    """Grammar rule from ``computational-actions.txt``."""

    HYP_ADJUNCTION_PREFIX = "hyp-adj"

    def __init__(
        self,
        name: str,
        lines_or_effect: list[str] | Effect,
        always_good: bool = False,
        backtrack_on_success: bool = False,
    ) -> None:
        """Construct from source *lines* or a ready *Effect* (Java constructor overloads)."""
        if isinstance(lines_or_effect, Effect):
            self._source_lines: list[str] = []
            eff: Effect | None = lines_or_effect
        else:
            self._source_lines = list(lines_or_effect)
            eff = EffectFactory.create_lines(self._source_lines)
        super().__init__(name, eff, backtrack_on_success)
        self.always_good = always_good

    def is_always_good(self) -> bool:
        """Return whether this action belongs to the non-optional star grammar."""
        return self.always_good

    def set_always_good(self, v: bool) -> None:
        """Set non-optional star-grammar membership."""
        self.always_good = v

    def instantiate(self) -> ComputationalAction:
        """Return a copy whose effect has metavariables resolved (Java ``instantiate``).

        Hyp-adj / link keep the same effect object. Others use ``effect.instantiate()``
        so DAG edges store concrete IF labels (e.g. ``Ty(e>t)`` not ``Ty(X)``).
        """
        assert self.effect is not None
        if self.name.startswith(self.HYP_ADJUNCTION_PREFIX) or self.name.startswith("link"):
            # Java: ``new ComputationalAction(name, this.action)`` — flags default false.
            return ComputationalAction(self.name, self.effect)
        # Java: ``new ComputationalAction(name, this.action.instantiate())``.
        return ComputationalAction(self.name, self.effect.instantiate())

    def exec_exhaustively(
        self,
        tree: Tree,
        context: Any = None,
    ) -> list[tuple[ComputationalAction, Tree]] | None:
        """Delegate exhaustive metavar search to nested IF (Java ``ComputationalAction.execExhaustively``)."""
        from dylan.action.atomic.if_then_else import IfThenElse

        eff = self.effect
        if not isinstance(eff, IfThenElse):
            return None
        nested = eff.exec_exhaustively(tree, context)
        if not nested:
            return None
        results: list[tuple[ComputationalAction, Tree]] = []
        for ite, t in nested:
            rebuilt = ComputationalAction(
                self.name,
                ite,
                self.always_good,
                self.backtrack_on_success,
            )
            results.append((rebuilt, t))
        return results

    def __eq__(self, other: object) -> bool:
        """Java ``equals``: effect equality (name check in Java is a no-op)."""
        if not isinstance(other, ComputationalAction):
            return False
        return self.effect == other.effect

    def __hash__(self) -> int:
        """Hash by name and effect (pairs with ``__eq__``)."""
        return hash((self.name, self.effect))

    def __lt__(self, other: ComputationalAction) -> bool:
        """Sort always-good actions before optional ones."""
        if self.always_good and not other.always_good:
            return True
        if not self.always_good and other.always_good:
            return False
        return id(self) < id(other)


ComputationalAction.isAlwaysGood = ComputationalAction.is_always_good  # type: ignore[attr-defined]
ComputationalAction.setAlwaysGood = ComputationalAction.set_always_good  # type: ignore[attr-defined]
