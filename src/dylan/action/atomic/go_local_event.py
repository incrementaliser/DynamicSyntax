"""``goLocalEvent`` — jump to local event node and bind a meta-modality (Java ``GoLocalEvent``)."""

from __future__ import annotations

import logging
import re
from typing import Any

from dylan.action.atomic.effect import Effect
from dylan.tree.basic_operator import ARROW_UP, BasicOperator, DOWN_0
from dylan.tree.modality import Modality
from dylan.tree.tree import Tree
from dylan.type.dstype import DSType

logger = logging.getLogger(__name__)

_PATTERN = re.compile(r"(?i)goLocalEvent\((.+)\)")


class GoLocalEvent(Effect):
    """Climb to a ``t``-typed node, go down ``0``, and bind inverse modality (Java ``GoLocalEvent``)."""

    FUNCTOR = "goLocalEvent"

    def __init__(self, meta_modality: Any) -> None:
        self.meta_modality = meta_modality

    @classmethod
    def parse(cls, string: str) -> GoLocalEvent | None:
        m = _PATTERN.fullmatch(string.strip())
        if not m:
            return None
        inner = m.group(1).strip()
        try:
            mod = Modality.parse(inner)
        except ValueError as ex:
            raise ValueError(f"unrecognised goLocalEvent string: {string}") from ex
        from dylan.action.meta.meta_modality import MetaModality

        if not isinstance(mod, MetaModality):
            raise ValueError(f"goLocalEvent expects a modality metavar, got {inner!r}")
        return cls(mod)

    def exec_tuple_context(self, tree: Tree, context: Any) -> Tree | None:
        cur_node = tree.pointed_node
        cur_addr = tree.pointer
        ops: list[BasicOperator] = []
        req_t = cur_node.get_required_type()
        typ = cur_node.get_type()
        ty: DSType | None = req_t if req_t is not None else typ
        while ty != DSType.t:
            addr_s = cur_addr.address
            if addr_s.endswith("L"):
                return None
            ops.append(BasicOperator(ARROW_UP, addr_s[-1]))
            parent = cur_addr.up()
            if parent is None:
                return None
            cur_addr = parent
            cur_node = tree[cur_addr]
            req_t = cur_node.get_required_type()
            typ = cur_node.get_type()
            ty = req_t if req_t is not None else typ
        ops.append(DOWN_0)
        cur_addr = cur_addr.down0()
        m = Modality(ops, required=False)
        inverse = m.inverse()
        _ = self.meta_modality == inverse
        tree.set_pointer(cur_addr)
        return tree

    def instantiate(self) -> Effect:
        self.meta_modality.get_meta().reset()
        return self

    def __str__(self) -> str:
        return f"{self.FUNCTOR}({self.meta_modality})"
