"""``merge`` — merge a modality target into the pointed node (Java ``Merge``)."""

from __future__ import annotations

import logging
import re
from typing import Any

from dylan.action.atomic.effect import Effect
from dylan.tree.modality import Modality
from dylan.tree.tree import Tree

logger = logging.getLogger(__name__)

_PATTERN = re.compile(r"(?i)merge\((.+)\)")


class Merge(Effect):
    """Splice a non-locally-fixed node into the current fixed node (Java ``Merge``)."""

    FUNCTOR = "merge"

    def __init__(self, modality: Modality) -> None:
        self.modality = modality

    @classmethod
    def parse(cls, string: str) -> Merge | None:
        m = _PATTERN.fullmatch(string.strip())
        if not m:
            return None
        return cls(Modality.parse(m.group(1).strip()))

    def exec_tuple_context(self, tree: Tree, context: Any) -> Tree | None:
        node = tree.pointed_node
        if not node.is_locally_fixed():
            logger.debug("merge: pointed node not locally fixed")
            return None
        mod_i = self.modality.instantiate()
        other = tree.get_node(mod_i)
        if other is None or other.is_locally_fixed():
            logger.debug("merge: other missing or fixed")
            return None
        tyl = node.get_type_label()
        if tyl is not None:
            node.remove_label(tyl)
        thisfol = node.get_formula_label()
        otherfol = other.get_formula_label()
        if otherfol is not None and thisfol is not None:
            node.remove_label(thisfol)
        tree.merge(mod_i)
        return tree

    def instantiate(self) -> Effect:
        return Merge(self.modality.instantiate())

    def __str__(self) -> str:
        return f"{self.FUNCTOR}({self.modality})"
