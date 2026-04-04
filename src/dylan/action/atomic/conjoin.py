"""``conjoin`` — merge a formula onto the node's Fo via ``conjoin`` (Java ``Conjoin``)."""

from __future__ import annotations

import logging
import re
from typing import Any

from dylan.action.atomic.effect import Effect
from dylan.formula.formula import Formula
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.formula.ttr_formula import TTRFormula
from dylan.tree.label.labels import FormulaLabel
from dylan.tree.tree import Tree

logger = logging.getLogger(__name__)

_PATTERN = re.compile(r"(?i)conjoin\((.+)\)")


class Conjoin(Effect):
    """Replace or create ``Fo`` with ``f.conjoin(argument)`` (Java ``Conjoin``)."""

    FUNCTOR = "conjoin"

    def __init__(self, formula: Formula) -> None:
        self.formula = formula

    @classmethod
    def parse(cls, string: str) -> Conjoin | None:
        m = _PATTERN.fullmatch(string.strip())
        if not m:
            return None
        inner = m.group(1).strip()
        f = Formula.create(inner)
        if f is None:
            raise ValueError(f"conjoin: could not parse formula {inner!r}")
        return cls(f)

    def exec_tuple_context(self, tree: Tree, context: Any) -> Tree | None:
        node = tree.pointed_node
        l = node.get_formula_label()
        f: Formula
        if l is None:
            if isinstance(self.formula, TTRFormula):
                f = TTRRecordType()
            else:
                return None
        else:
            f = l.get_formula()
        if l is not None:
            node.remove_label(l)
        instance = self.formula.instantiate()
        conjoined = FormulaLabel(f.conjoin(instance))
        node.add_label(conjoined)
        return tree

    def instantiate(self) -> Effect:
        return Conjoin(self.formula.instantiate())

    def __str__(self) -> str:
        return f"{self.FUNCTOR}({self.formula})"
