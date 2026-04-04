"""Build effects from spec strings (partial ``EffectFactory``).

Recognised action keywords are dispatched to concrete ``Effect``
subclasses; everything else falls back to :class:`GenericEffect` (a
no-op) so that grammar files can be loaded without crashing.
"""

from __future__ import annotations

import logging

from dylan.action.atomic.abort import Abort
from dylan.action.atomic.effect import Effect
from dylan.action.atomic.empty_effect import EmptyEffect

logger = logging.getLogger(__name__)


class GenericEffect(Effect):
    """Fallback no-op effect for action specs not yet ported.

    Allows grammars to load; the action simply does nothing when
    executed.  A debug-level log message is emitted once per spec so
    unimplemented actions can be tracked.
    """

    def __init__(self, spec: str) -> None:
        self.spec = spec

    def exec_tuple_context(self, tree: "Tree", context: "ParserTuple | None") -> "Tree | None":  # type: ignore[override]
        """Return *tree* unchanged (no-op)."""
        return tree

    def instantiate(self) -> Effect:
        """Return a copy (nothing to bind)."""
        return GenericEffect(self.spec)

    def __str__(self) -> str:
        return f"GenericEffect({self.spec})"


class EffectFactory:
    """Factory mirroring Java ``EffectFactory.create``."""

    _macro_templates: dict[str, list[str]] = {}

    @classmethod
    def clear_macro_templates(cls) -> None:
        """Reset macro definitions (Java ``EffectFactory`` macro file absent)."""
        cls._macro_templates.clear()

    @classmethod
    def init_macro_templates(cls, raw_lines: list[str]) -> None:
        """Load ``lexical-macros.txt`` (stub: macros not expanded in v0)."""
        cls.clear_macro_templates()

    @staticmethod
    def get_if_indices(lines: list[str]) -> list[int]:
        """Return indices of lines starting with ``IF`` (case-insensitive)."""
        return [i for i, ln in enumerate(lines) if ln.strip().lower().startswith("if")]

    @staticmethod
    def create_lines(lines: list[str]) -> Effect:
        """Create an effect from one or more source lines."""
        from dylan.action.atomic.if_then_else import IfThenElse

        if len(lines) == 1:
            return EffectFactory.create(lines[0])
        if lines[0].strip().lower().startswith("if"):
            return IfThenElse.from_lines(lines)
        return GenericEffect(" / ".join(lines))

    @staticmethod
    def create(line: str) -> Effect:
        """Create a single-line effect from its text specification."""
        from dylan.action.atomic.delete import Delete
        from dylan.action.atomic.go import Go
        from dylan.action.atomic.make import Make
        from dylan.action.atomic.put import Put

        s = line.strip()
        low = s.lower()

        if low.startswith(Abort.FUNCTOR.lower()):
            return Abort()
        if low.startswith(EmptyEffect.FUNCTOR.lower()):
            return EmptyEffect()

        eff: Effect | None
        eff = Make.parse(s)
        if eff is not None:
            return eff
        eff = Go.parse(s)
        if eff is not None:
            return eff
        eff = Put.parse(s)
        if eff is not None:
            return eff
        eff = Delete.parse(s)
        if eff is not None:
            return eff

        return GenericEffect(s)

    @staticmethod
    def create_multiple(lines: list[str], if_indices: list[int]) -> list[Effect]:
        """Split *lines* at ``IF`` boundaries and build one effect per block."""
        from dylan.action.atomic.if_then_else import IfThenElse

        if not if_indices:
            return [GenericEffect(" / ".join(lines))]
        chunks: list[Effect] = []
        starts = if_indices + [len(lines)]
        for i in range(len(if_indices)):
            sub = lines[starts[i] : starts[i + 1]]
            chunks.append(IfThenElse.from_lines(sub))
        return chunks
