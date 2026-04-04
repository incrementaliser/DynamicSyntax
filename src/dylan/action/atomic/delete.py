"""``delete(label)`` effect — remove a label from the pointed node (Java ``Delete``)."""

from __future__ import annotations

import logging
import re

from dylan.action.atomic.effect import Effect
from typing import Any
from dylan.tree.label.labels import Label, label_factory_create
from dylan.tree.tree import Tree

logger = logging.getLogger(__name__)

_DELETE_RE = re.compile(r"(?i)delete\((.+)\)")


class Delete(Effect):
    """Remove a label from the pointed node."""

    FUNCTOR = "delete"

    def __init__(self, label: Label) -> None:
        self.label = label

    @classmethod
    def parse(cls, string: str) -> Delete | None:
        """Parse ``delete(?ty(e))`` etc.; return ``None`` if no match."""
        m = _DELETE_RE.fullmatch(string.strip())
        if not m:
            return None
        lab = label_factory_create(m.group(1).strip())
        return cls(lab)

    def exec_tuple_context(self, tree: Tree, context: Any) -> Tree | None:
        tree.delete_label(self.label)
        return tree

    def instantiate(self) -> Effect:
        """Fresh effect with label metavariables resolved (Java ``Delete.instantiate``)."""
        return Delete(self.label.instantiate())

    def __str__(self) -> str:
        return f"delete({self.label})"
