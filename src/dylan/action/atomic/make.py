"""``make(modality)`` effect — create a daughter node (Java ``Make``)."""

from __future__ import annotations

import logging
import re

from typing import Any

from dylan.action.atomic.effect import Effect
from dylan.tree.basic_operator import BasicOperator
from dylan.tree.label.labels import BottomLabel
from dylan.tree.tree import Tree

logger = logging.getLogger(__name__)

_MAKE_RE = re.compile(r"(?i)make\((.+)\)")


class Make(Effect):
    """Create a new daughter node below the pointed node."""

    FUNCTOR = "make"

    def __init__(self, op: BasicOperator) -> None:
        self.op = op

    @classmethod
    def parse(cls, string: str) -> Make | None:
        """Parse ``make(\\/0)`` etc.; return ``None`` if no match or unparseable operator."""
        m = _MAKE_RE.fullmatch(string.strip())
        if not m:
            return None
        try:
            return cls(BasicOperator.parse(m.group(1).strip()))
        except ValueError:
            return None

    def exec_tuple_context(self, tree: Tree, context: Any) -> Tree | None:
        if any(isinstance(lab, BottomLabel) for lab in tree.pointed_node.labels) and not self.op.is_link():
            logger.debug("make: BottomLabel on node and op is not link — fail")
            return None
        addr = tree.pointer.go_op(self.op)
        if addr is None:
            return None
        if addr not in tree:
            tree.make(self.op)
        else:
            logger.debug("make: node %s already exists", addr)
        return tree

    def instantiate(self) -> Effect:
        return Make(self.op)

    def __str__(self) -> str:
        return f"make({self.op})"
