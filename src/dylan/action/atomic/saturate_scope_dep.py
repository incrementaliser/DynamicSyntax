"""``saturate_scope_dep`` — extend scope statements on the pointed node (Java ``SaturateScopeDep``)."""

from __future__ import annotations

import logging
import re
from typing import Any

from dylan.action.atomic.effect import Effect
from dylan.action.meta.meta_formula import MetaFormula
from dylan.formula.formula import Formula
from dylan.tree.label.labels import ScopeStatement
from dylan.tree.tree import Tree

logger = logging.getLogger(__name__)

_PATTERN = re.compile(r"(?i)saturate_scope_dep\(\s*(.+)\s*,\s*(.+)\s*\)")


class SaturateScopeDep(Effect):
    """Add refined :class:`ScopeStatement` labels matching Java ``SaturateScopeDep``."""

    FUNCTOR = "saturate_scope_dep"

    def __init__(self, f1: Formula, f2: Formula) -> None:
        self.f1 = f1
        self.f2 = f2

    @classmethod
    def parse(cls, string: str) -> SaturateScopeDep | None:
        m = _PATTERN.fullmatch(string.strip())
        if not m:
            return None
        a, b = Formula.create(m.group(1).strip()), Formula.create(m.group(2).strip())
        if a is None or b is None:
            raise ValueError(f"saturate_scope_dep: bad formula in {string!r}")
        return cls(a, b)

    def exec_tuple_context(self, tree: Tree, context: Any) -> Tree | None:
        n = tree.pointed_node
        f1i = self.f1.instantiate()
        f2i = self.f2.instantiate()
        if isinstance(f1i, MetaFormula) or isinstance(f2i, MetaFormula):
            logger.debug("saturate_scope_dep: metavar in instance; leaving tree")
            return tree
        to_add: list[ScopeStatement] = []
        for lab in n.labels:
            if isinstance(lab, ScopeStatement):
                narrow = lab.get_narrowest()
                wide = lab.get_widest()
                if wide == f1i and f2i != narrow:
                    to_add.append(ScopeStatement(f2i, narrow))
        for ss in to_add:
            tree.put(ss.instantiate())
        return tree

    def instantiate(self) -> Effect:
        return self

    def __str__(self) -> str:
        return f"{self.FUNCTOR}({self.f1},{self.f2})"
