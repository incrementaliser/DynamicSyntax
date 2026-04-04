"""DS tree (partial Java ``Tree``)."""

from __future__ import annotations

import logging
from typing import Any

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.tree.basic_operator import BasicOperator
from dylan.tree.label.labels import Label, Requirement, TypeLabel
from dylan.tree.modality import Modality
from dylan.tree.node import Node
from dylan.tree.node_address import NodeAddress

logger = logging.getLogger(__name__)


class Tree(dict[NodeAddress, Node]):
    """Maps node addresses to nodes (Java extends TreeMap)."""

    def __init__(self, other: Tree | None = None) -> None:
        super().__init__()
        if other is None:
            self.root_addr = NodeAddress()
            self.pointer = self.root_addr
            root = Node(self.root_addr)
            root.add_label(Requirement(TypeLabel.t))
            self[self.root_addr] = root
        else:
            self.root_addr = other.root_addr
            self.pointer = other.pointer
            for k, v in other.items():
                n = Node(v.address)
                n.labels = list(v.labels)
                self[k] = n

    @property
    def pointed_node(self) -> Node:
        return self[self.pointer]

    def clone(self) -> Tree:
        return Tree(self)

    # ── tree-modifying operations (Java Tree.make / go / put / delete) ──

    def make(self, op: BasicOperator) -> None:
        """Create a new daughter node below the pointed node (Java ``Tree.make``)."""
        if not op.is_down():
            raise RuntimeError(f"Can't make non-daughter node {op}")
        addr = self.pointer.go_op(op)
        if addr is not None and addr not in self:
            self[addr] = Node(addr)

    def go(self, mod: Modality) -> None:
        """Move the pointer along *mod* (Java ``Tree.go(Modality)``)."""
        addr = self.pointer.go_modality(mod)
        if addr is None or addr not in self:
            raise RuntimeError(f"Can't go to non-existent node from {self.pointer} via {mod}")
        self.pointer = addr

    def go_op(self, op: BasicOperator) -> None:
        """Move the pointer one step (Java ``Tree.go(BasicOperator)``)."""
        addr = self.pointer.go_op(op)
        if addr is not None:
            self.pointer = addr

    def put_label(self, label: Label) -> None:
        """Add a label at the pointed node (Java ``Tree.put``)."""
        self.pointed_node.add_label(label)

    def delete_label(self, label: Label) -> None:
        """Remove a label from the pointed node (Java ``Tree.delete``)."""
        self.pointed_node.remove_label(label)

    # ── semantics stubs ──

    def get_maximal_semantics(self, context: Any = None) -> TTRRecordType:
        """Maximal TTR semantics (stub: empty record if root has no formula)."""
        return TTRRecordType()

    def get_maximal_semantics_with_context(self, context: Any) -> TTRRecordType:
        return self.get_maximal_semantics(context)
