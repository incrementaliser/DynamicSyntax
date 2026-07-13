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
        """Execute on *tree* freshening from *tree*'s pools (Java ``execTupleContext``).

        Java ``TTRFreshPut.execTupleContext`` calls ``freshenVars(tree)`` on the
        tree being modified (not the parent context), so successive hyp-sem
        puts on a cloned search tree allocate distinct entity labels.
        """
        _ = context
        node = tree.pointed_node
        if node.get_formula_label() is not None:
            logger.warning("ttrput: node already has Fo; leaving tree")
            return tree
        fresh = self.ttr.freshen_vars(tree)
        node.add_label(FormulaLabel(fresh.instantiate()))
        return tree

    def instantiate(self) -> Effect:
        return TTRFreshPut(self.ttr.clone())  # type: ignore[arg-type]

    def __eq__(self, other: object) -> bool:
        """Java ``TTRFreshPut.equals``: mutual TTR subsumption (not string identity)."""
        if not isinstance(other, TTRFreshPut):
            return False
        if self.ttr is None:
            return other.ttr is None
        if other.ttr is None:
            return False
        return bool(self.ttr.subsumes(other.ttr) and other.ttr.subsumes(self.ttr))

    def __hash__(self) -> int:
        """Match Java ``Effect.hashCode`` (``toString``-based); equals may be coarser via subsumption."""
        return hash(str(self))

    def __str__(self) -> str:
        return f"{self.FUNCTOR}({self.ttr})"
