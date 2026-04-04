"""``do`` — run a named meta action sequence (Java ``Do``)."""

from __future__ import annotations

import logging
import re
from typing import Any

from dylan.action.atomic.effect import Effect
from dylan.action.meta.meta_action_sequence import MetaActionSequence
from dylan.tree.tree import Tree

logger = logging.getLogger(__name__)

_PATTERN = re.compile(r"(?i)do\(([A-Z][A-Z0-9]*)\)")


class Do(Effect):
    """Instantiate and execute every action in a registered :class:`MetaActionSequence` (Java ``Do``)."""

    FUNCTOR = "do"

    def __init__(self, meta_action_sequence_name: str) -> None:
        self.meta_action_sequence_name = meta_action_sequence_name

    @classmethod
    def parse(cls, string: str) -> Do | None:
        m = _PATTERN.fullmatch(string.strip())
        if not m:
            return None
        return cls(m.group(1))

    def exec_tuple_context(self, tree: Tree, context: Any) -> Tree | None:
        seq = MetaActionSequence.get(self.meta_action_sequence_name).instantiate()
        cur: Tree | None = tree
        for a in seq:
            assert cur is not None
            cur = a.exec(cur, context) if hasattr(a, "exec") else None
            if cur is None:
                logger.debug("do: action %s failed", a)
                return None
        return cur

    def instantiate(self) -> Effect:
        return Do(self.meta_action_sequence_name)

    def __str__(self) -> str:
        return f"{self.FUNCTOR}({self.meta_action_sequence_name})"
