"""IF / THEN / ELSE effect (Java ``IfThenElse``).

Supports arbitrary nesting depth as per the Java implementation.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from dylan.action.atomic.effect import Effect
from dylan.action.atomic.effect_factory import EffectFactory
from dylan.action.meta.element import reset_all_meta_bindings
from dylan.action.meta_stub import reset_bound_metas
from dylan.tree.label.labels import Label, label_factory_create
from dylan.tree.tree import Tree

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_ITE_PATTERN = re.compile(r"(?i)(IF|THEN|ELSE)?\s*(.*)")
_IF_RE = re.compile(r"(?i)\bIF\b")
_ELSE_RE = re.compile(r"(?i)\bELSE\b")


def _find_end_of_embedded_ite(lines: list[str], start: int) -> int:
    """Return the index just past the end of a nested IF/THEN/ELSE block starting at *start*.

    Mirrors Java ``findEndIndexOfEmbeddedITE``.
    """
    embed = 1
    j = start + 1
    while j < len(lines):
        cur = lines[j].lower()
        if "else" in cur:
            embed -= 1
        if embed < 0:
            break
        if "if" in cur:
            embed += 1
        j += 1
    return j


class Backtracker:
    """Controls metavar backtracking during label checking (stub)."""

    def __init__(self) -> None:
        self.index = 0

    def set_index(self, i: int) -> None:
        """Set the current backtrack index."""
        self.index = i

    def can_backtrack_tuple_context(self, tree: Tree, context: ParserTuple | None) -> bool:
        """Always ``False`` in stub implementation."""
        return False


class IfThenElse(Effect):
    """Conditional effect used by lexical and computational actions."""

    IF_FUNCTOR = "IF"
    THEN_FUNCTOR = "THEN"
    ELSE_FUNCTOR = "ELSE"

    def __init__(
        self,
        if_labels: list[Label],
        then_effects: list[Effect],
        else_effects: list[Effect],
        embedding_level: int = 0,
        parent: IfThenElse | None = None,
    ) -> None:
        self.if_labels = if_labels
        self.then_effects = then_effects
        self.else_effects = else_effects
        self.embedding_level = embedding_level
        self.parent = parent
        self.backtracker = Backtracker()

    @classmethod
    def from_lines(
        cls,
        lines: list[str],
        embedding_level: int = 0,
        parent: IfThenElse | None = None,
    ) -> IfThenElse:
        """Parse IF/THEN/ELSE blocks with full nesting support (mirrors Java constructor)."""
        src = [str(l) for l in lines]
        if_labels: list[Label] = []
        then_effects: list[Effect] = []
        else_effects: list[Effect] = []
        phase = cls.IF_FUNCTOR
        i = 0
        while i < len(src):
            line = src[i].strip()
            m = _ITE_PATTERN.match(line)
            if not m:
                i += 1
                continue
            kw = m.group(1)
            rest = (m.group(2) or "").strip()

            if kw is not None:
                ku = kw.upper()
                if ku == cls.THEN_FUNCTOR and phase == cls.IF_FUNCTOR:
                    phase = cls.THEN_FUNCTOR
                elif ku == cls.ELSE_FUNCTOR and phase == cls.THEN_FUNCTOR:
                    phase = cls.ELSE_FUNCTOR

            if phase == cls.IF_FUNCTOR:
                payload = rest if kw else line
                if payload:
                    if_labels.append(label_factory_create(payload, None))
                i += 1

            elif phase == cls.THEN_FUNCTOR:
                idx_if = line.lower().find("if")
                if idx_if >= 0:
                    j = _find_end_of_embedded_ite(src, i)
                    sub = src[i:j]
                    sub[0] = line[idx_if:]
                    then_effects.append(
                        cls.from_lines(sub, embedding_level + 1, None),
                    )
                    i = j
                else:
                    payload = rest if kw else line
                    if payload:
                        then_effects.append(EffectFactory.create(payload))
                    i += 1

            elif phase == cls.ELSE_FUNCTOR:
                idx_if = line.lower().find("if")
                if idx_if >= 0:
                    j = _find_end_of_embedded_ite(src, i)
                    sub = src[i:j]
                    sub[0] = line[idx_if:]
                    else_effects.append(
                        cls.from_lines(sub, embedding_level + 1, None),
                    )
                    i = j
                else:
                    payload = rest if kw else line
                    if payload:
                        else_effects.append(EffectFactory.create(payload))
                    i += 1
            else:
                i += 1

        result = cls(if_labels, then_effects, else_effects, embedding_level, parent)
        return result

    def setup_backtrackers(self, exceptions: list[Any] | None = None) -> None:
        self.backtracker = Backtracker()

    def exec_tuple_context(self, tree: Tree, context: Any) -> Tree | None:
        """Execute this IF/THEN/ELSE on *tree*.

        Matches Java ``IfThenElse.execTupleContext``: when the chosen branch
        (THEN or ELSE) is **empty**, return ``None`` (action fails), not the
        original tree.  Java initialises ``result = null`` and only overwrites
        it when an effect executes; an empty branch therefore yields ``null``.
        """
        if self.embedding_level == 0:
            reset_all_meta_bindings()
            self.setup_backtrackers([])
        else:
            reset_bound_metas()
        self.backtracker.set_index(0)
        success = True
        for lab in self.if_labels:
            if not lab.check_with_tuple_as_context(tree, context):
                success = False
                break
        branch = self.then_effects if success else self.else_effects
        if not branch:
            return None
        result: Tree | None = tree
        for eff in branch:
            result = eff.exec_tuple_context(result, context)
            if result is None:
                return None
        return result

    def exec(self, tree: Tree, context: Any) -> Tree | None:
        return self.exec_tuple_context(tree, context)

    def instantiate(self) -> Effect:
        return IfThenElse(
            list(self.if_labels),
            [e.instantiate() for e in self.then_effects],
            [e.instantiate() for e in self.else_effects],
            self.embedding_level,
            self.parent,
        )

    def __str__(self) -> str:
        return (
            f"IF {self.if_labels} THEN {self.then_effects} ELSE {self.else_effects}"
        )
