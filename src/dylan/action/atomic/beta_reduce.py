"""``beta-reduce`` — apply lambda at ``down1`` to formula at ``down0`` (Java ``BetaReduce``)."""

from __future__ import annotations

from typing import Any

from dylan.action.atomic.effect import Effect
from dylan.tree.label.labels import FormulaLabel, TypeLabel
from dylan.tree.tree import Tree
from dylan.type.dstype import ConstructedType


class BetaReduce(Effect):
    """Combine functor ``down1`` lambda with argument ``down0`` on the parent (Java ``BetaReduce``)."""

    FUNCTOR = "beta-reduce"

    def exec_tuple_context(self, tree: Tree, context: Any) -> Tree | None:
        node = tree.pointed_node
        d0 = tree.node_at(node.address.down0())
        d1 = tree.node_at(node.address.down1())
        if d0 is None or d1 is None:
            raise RuntimeError("beta-reduce: missing daughters")
        t0 = d0.get_type()
        t1 = d1.get_type()
        f0 = d0.get_formula()
        f1 = d1.get_formula()
        if not isinstance(t1, ConstructedType):
            raise RuntimeError(f"beta-reduce: unsuitable functor type {t1}")
        ct1 = t1
        if t0 is None or ct1.from_type != t0:
            raise RuntimeError(f"beta-reduce: unsuitable types {t0!s} vs {ct1!s}")
        beta = getattr(f1, "beta_reduce", None)
        if f1 is None or not callable(beta):
            raise RuntimeError(f"beta-reduce: formula at down1 is not a lambda: {f1!r}")
        if f0 is None:
            raise RuntimeError("beta-reduce: missing formula at down0")
        reduced = beta(f0)
        node.add_label(TypeLabel(ct1.to_type))
        node.add_label(FormulaLabel(reduced))
        return tree

    def instantiate(self) -> Effect:
        return BetaReduce()

    def __str__(self) -> str:
        return self.FUNCTOR
