"""``copy_content`` — copy Ty/Fo from a modality-reached node (Java ``CopyContent``)."""

from __future__ import annotations

import logging
import re
from typing import Any

from dylan.action.atomic.effect import Effect
from dylan.tree.label.labels import FormulaLabel, TypeLabel
from dylan.tree.modality import Modality
from dylan.tree.tree import Tree

logger = logging.getLogger(__name__)

_PATTERN = re.compile(r"(?i)copy_content\((.+)\)")


class CopyContent(Effect):
    """Copy type and formula labels from ``get_node(mod)`` onto the pointer (Java ``CopyContent``)."""

    FUNCTOR = "copy_content"

    def __init__(self, modality: Modality) -> None:
        self.modality = modality

    @classmethod
    def parse(cls, string: str) -> CopyContent | None:
        m = _PATTERN.fullmatch(string.strip())
        if not m:
            return None
        return cls(Modality.parse(m.group(1).strip()))

    def exec_tuple_context(self, tree: Tree, context: Any) -> Tree | None:
        n = tree.get_node(self.modality.instantiate())
        if n is None or not n.has_type():
            return None
        for lab in n.labels:
            if isinstance(lab, (TypeLabel, FormulaLabel)):
                tree.put(lab.instantiate())
        return tree

    def instantiate(self) -> Effect:
        return CopyContent(self.modality.instantiate())

    def __str__(self) -> str:
        return f"{self.FUNCTOR}({self.modality})"
