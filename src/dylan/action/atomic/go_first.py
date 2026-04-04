"""``gofirst`` — move pointer to nearest ancestor carrying a label (Java ``GoFirst``)."""

from __future__ import annotations

import logging
import re
from typing import Any

from dylan.action.atomic.effect import Effect
from dylan.tree.label.labels import Label, label_factory_create
from dylan.tree.tree import Tree

logger = logging.getLogger(__name__)

_PATTERN = re.compile(r"(?i)gofirst\((.+)\)")


class GoFirst(Effect):
    """Walk up until ``label.check`` holds, then set pointer (Java ``GoFirst``)."""

    FUNCTOR = "gofirst"

    def __init__(self, label: Label) -> None:
        self.label = label

    @classmethod
    def parse(cls, string: str) -> GoFirst | None:
        m = _PATTERN.fullmatch(string.strip())
        if not m:
            return None
        return cls(label_factory_create(m.group(1).strip()))

    def exec_tuple_context(self, tree: Tree, context: Any) -> Tree | None:
        cur_addr = tree.pointer
        cur_node = tree.pointed_node
        while not self.label.check(cur_node):
            parent = cur_addr.up()
            if parent is None:
                logger.debug("GoFirst: no matching label above; failing")
                return None
            cur_addr = parent
            cur_node = tree[cur_addr]
        tree.set_pointer(cur_addr)
        return tree

    def instantiate(self) -> Effect:
        return GoFirst(self.label.instantiate())

    def __str__(self) -> str:
        return f"{self.FUNCTOR}({self.label})"
