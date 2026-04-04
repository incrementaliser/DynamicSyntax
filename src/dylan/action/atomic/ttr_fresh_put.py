"""``ttrput`` — add a freshened TTR formula as ``Fo`` (Java ``TTRFreshPut``)."""

from __future__ import annotations

import logging
import re
from typing import Any

from dylan.action.atomic.effect import Effect
from dylan.formula.formula import Formula
from dylan.formula.ttr_formula import TTRFormula
from dylan.tree.label.labels import FormulaLabel
from dylan.tree.tree import Tree

logger = logging.getLogger(__name__)

_PATTERN = re.compile(r"(?i)ttrput\((.+)\)")


class TTRFreshPut(Effect):
    """Put ``Fo(freshen(ttr))`` when no formula label exists (Java ``TTRFreshPut``)."""

    FUNCTOR = "ttrput"

    def __init__(self, ttr: TTRFormula) -> None:
        self.ttr = ttr

    @classmethod
    def parse(cls, string: str) -> TTRFreshPut | None:
        m = _PATTERN.fullmatch(string.strip())
        if not m:
            return None
        inner = m.group(1).strip()
        f = Formula.create(inner)
        if isinstance(f, TTRFormula):
            return cls(f)
        from dylan.formula.opaque_ttr_spec import OpaqueTTRSpec

        return cls(OpaqueTTRSpec(inner))

    def exec_tuple_context(self, tree: Tree, context: Any) -> Tree | None:
        node = tree.pointed_node
        if node.get_formula_label() is not None:
            logger.warning("ttrput: node already has Fo; leaving tree")
            return tree
        fresh = self.ttr.freshen_vars(tree)
        inst = fresh.instantiate().evaluate()
        node.add_label(FormulaLabel(inst))
        return tree

    def instantiate(self) -> Effect:
        return TTRFreshPut(self.ttr.clone())  # type: ignore[arg-type]

    def __str__(self) -> str:
        return f"{self.FUNCTOR}({self.ttr})"
