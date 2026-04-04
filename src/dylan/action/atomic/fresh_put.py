"""``freshput`` — bind a meta-formula to a fresh variable (Java ``FreshPut``)."""

from __future__ import annotations

import logging
import re
from typing import Any

from dylan.action.atomic.effect import Effect
from dylan.action.meta.meta_formula import MetaFormula
from dylan.tree.label.labels import FormulaLabel
from dylan.tree.tree import Tree

logger = logging.getLogger(__name__)

_PATTERN = re.compile(r"(?i)freshput\(\s*([S-U])\s*,\s*(\w+)\s*\)")
_TYPES = frozenset({"event", "prop", "entity"})


class FreshPut(Effect):
    """Allocate a tree-fresh variable and bind it to a :class:`MetaFormula` (Java ``FreshPut``)."""

    FUNCTOR = "freshput"

    def __init__(self, var: MetaFormula, var_type: str) -> None:
        self.var = var
        self.var_type = var_type

    @classmethod
    def parse(cls, string: str) -> FreshPut | None:
        m = _PATTERN.fullmatch(string.strip())
        if not m:
            return None
        vname, typ = m.group(1), m.group(2).strip().lower()
        if typ not in _TYPES:
            raise ValueError(f"freshput: unknown variable type {typ!r}")
        return cls(MetaFormula.get(vname), typ)

    def exec_tuple_context(self, tree: Tree, context: Any) -> Tree | None:
        if self.var_type == "event":
            fresh = tree.get_fresh_event_variable()
        elif self.var_type == "prop":
            fresh = tree.get_fresh_proposition_variable()
        else:
            fresh = tree.get_fresh_entity_variable()
        self.var.get_meta().reset()
        _ = self.var == fresh
        tree.put(FormulaLabel(self.var.instantiate()))
        return tree

    def instantiate(self) -> Effect:
        self.var.get_meta().reset()
        return self

    def __str__(self) -> str:
        return f"{self.FUNCTOR}({self.var.get_meta().name},{self.var_type})"
