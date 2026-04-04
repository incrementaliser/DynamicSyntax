"""``put(label)`` effect — add a label at the pointed node (Java ``Put``)."""

from __future__ import annotations

import logging
import re

from dylan.action.atomic.effect import Effect
from dylan.dag.parser_tuple import ParserTuple
from dylan.tree.label.labels import Label, label_factory_create
from dylan.tree.tree import Tree

logger = logging.getLogger(__name__)

_PUT_RE = re.compile(r"(?i)put\((.+)\)")


class Put(Effect):
    """Add a label to the pointed node."""

    FUNCTOR = "put"

    def __init__(self, label: Label) -> None:
        self.label = label

    @classmethod
    def parse(cls, string: str) -> Put | None:
        """Parse ``put(?ty(e))`` etc.; return ``None`` if no match."""
        m = _PUT_RE.fullmatch(string.strip())
        if not m:
            return None
        lab = label_factory_create(m.group(1).strip())
        return cls(lab)

    def exec_tuple_context(self, tree: Tree, context: ParserTuple | None) -> Tree | None:
        node = tree.pointed_node
        if node.contains(self.label):
            logger.debug("put: label %s already present at %s", self.label, node.address)
            return tree
        tree.put_label(self.label)
        return tree

    def instantiate(self) -> Effect:
        return Put(self.label)

    def __str__(self) -> str:
        return f"put({self.label})"
