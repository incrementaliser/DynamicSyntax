"""``go(modality)`` effect — move the tree pointer (Java ``Go``)."""

from __future__ import annotations

import logging
import re

from dylan.action.atomic.effect import Effect
from typing import Any
from dylan.tree.modality import Modality
from dylan.tree.tree import Tree

logger = logging.getLogger(__name__)

_GO_RE = re.compile(r"(?i)go\((.+)\)")


class Go(Effect):
    """Move the tree pointer along a modality path."""

    FUNCTOR = "go"

    def __init__(self, modality: Modality) -> None:
        self.modality = modality

    @classmethod
    def parse(cls, string: str) -> Go | None:
        """Parse ``go(\\/0)`` etc.; return ``None`` if no match or if the modality is unparseable."""
        m = _GO_RE.fullmatch(string.strip())
        if not m:
            return None
        try:
            return cls(Modality.parse(m.group(1).strip()))
        except ValueError:
            return None

    def exec_tuple_context(self, tree: Tree, context: Any) -> Tree | None:
        addr = tree.pointer.go_modality(self.modality)
        if addr is None or addr not in tree:
            logger.debug("go: cannot reach node via %s from %s", self.modality, tree.pointer)
            return None
        tree.pointer = addr
        return tree

    def instantiate(self) -> Effect:
        return Go(self.modality)

    def __str__(self) -> str:
        return f"go({self.modality})"
