"""DS tree (partial Java ``Tree``)."""

from __future__ import annotations

import logging
from typing import Any

from dylan.formula.variable import Variable
from dylan.tree.basic_operator import BasicOperator
from dylan.tree.label.labels import Label, Requirement, TypeLabel
from dylan.tree.modality import Modality
from dylan.tree.node import Node
from dylan.tree.node_address import NodeAddress

logger = logging.getLogger(__name__)

ENTITY_VARIABLE_ROOT = "x"
EVENT_VARIABLE_ROOT = "e"
PROPOSITION_VARIABLE_ROOT = "p"


class Tree(dict[NodeAddress, Node]):
    """Maps node addresses to nodes (Java extends TreeMap)."""

    def __init__(self, other: Tree | None = None) -> None:
        super().__init__()
        if other is None:
            self.root_addr = NodeAddress()
            self.pointer = self.root_addr
            self._entity_pool: list[Variable] = []
            self._event_pool: list[Variable] = []
            self._proposition_pool: list[Variable] = []
            root = Node(self.root_addr)
            root.add_label(Requirement(TypeLabel.t))
            self[self.root_addr] = root
        else:
            self.root_addr = other.root_addr
            self.pointer = other.pointer
            self._entity_pool = list(other._entity_pool)
            self._event_pool = list(other._event_pool)
            self._proposition_pool = list(other._proposition_pool)
            for k, v in other.items():
                self[k] = Node(v.address, list(v.labels))

    @property
    def pointed_node(self) -> Node:
        return self[self.pointer]

    def clone(self) -> Tree:
        """Deep copy nodes, pointer, and variable pools (Java ``Tree.clone`` sketch)."""
        return Tree(self)

    def set_pointer(self, addr: NodeAddress) -> None:
        """Set the pointer address (Java ``setPointer``)."""
        self.pointer = addr

    def get_fresh_entity_variable(self) -> Variable:
        """Allocate ``x1``, ``x2``, … (Java ``getFreshEntityVariable``)."""
        v = Variable(ENTITY_VARIABLE_ROOT + str(len(self._entity_pool) + 1))
        self._entity_pool.append(v)
        return v

    def get_fresh_event_variable(self) -> Variable:
        """Allocate ``e1``, ``e2``, … (Java ``getFreshEventVariable``)."""
        v = Variable(EVENT_VARIABLE_ROOT + str(len(self._event_pool) + 1))
        self._event_pool.append(v)
        return v

    def get_fresh_proposition_variable(self) -> Variable:
        """Allocate ``p1``, ``p2``, … (Java ``getFreshPropositionVariable``)."""
        v = Variable(PROPOSITION_VARIABLE_ROOT + str(len(self._proposition_pool) + 1))
        self._proposition_pool.append(v)
        return v

    def node_at(self, addr: NodeAddress) -> Node | None:
        """Return the node at *addr* or ``None`` (Java ``get``)."""
        return self.get(addr)

    def get_node(self, modality: Modality) -> Node | None:
        """Node reached from the pointer via *modality* (Java ``getNode(Modality)``)."""
        dest = self.pointer.go_modality(modality)
        if dest is None:
            return None
        return self.get(dest)

    def get_daughters(self, node: Node, order: str | None = None) -> list[Node]:
        """Daughter nodes in fixed order, optionally filtered by *order* chars (Java ``getDaughters``)."""
        addr = node.address
        seq: list[NodeAddress] = []
        if order is None:
            seq = [
                addr.down0(),
                addr.down1(),
                addr.down_link(),
                addr.down_star(),
                addr.down_local_unfixed(),
            ]
        else:
            for ch in order:
                seq.append(addr.down_char(ch))
        out: list[Node] = []
        for a in seq:
            n = self.get(a)
            if n is not None:
                out.append(n)
        return out

    def _move_daughters(self, dtrs: list[Node], from_addr: NodeAddress, to_addr: NodeAddress) -> None:
        """Re-home subtrees under *to_addr* (Java ``moveDaughters``)."""
        from_s = from_addr.address
        to_s = to_addr.address
        for dtr in dtrs:
            self._move_daughters(self.get_daughters(dtr), from_addr, to_addr)
            old_a = dtr.address.address
            if not old_a.startswith(from_s):
                raise RuntimeError(f"moveDaughters: {old_a!r} does not start with {from_s!r}")
            new_addr = NodeAddress(to_s + old_a[len(from_s) :])
            new_node = Node(new_addr, list(dtr.labels))
            self[new_addr] = new_node
            del self[dtr.address]

    def merge(self, modality: Modality) -> None:
        """Merge the node at *modality* into the pointed node (Java ``Tree.merge``)."""
        other = self.get_node(modality)
        if other is None:
            raise RuntimeError("merge: no node at modality")
        self._move_daughters(self.get_daughters(other), other.address, self.pointer)
        self.pointed_node.merge_from(other)
        del self[other.address]

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

    put = put_label

    def delete_label(self, label: Label) -> None:
        """Remove a label from the pointed node (Java ``Tree.delete``)."""
        self.pointed_node.remove_label(label)

    def is_complete(self) -> bool:
        """True when no node still carries an outstanding :class:`Requirement` (Java ``Tree.isComplete`` sketch)."""
        from dylan.tree.label.labels import Requirement as _Req

        for node in self.values():
            for lab in node.labels:
                if isinstance(lab, _Req):
                    return False
        return True

    # ── semantics stubs ──

    def get_maximal_semantics(self, context: Any = None) -> "TTRRecordType":
        """Maximal TTR semantics (stub: empty record if root has no formula)."""
        from dylan.formula.ttr_record_type import TTRRecordType

        return TTRRecordType()

    def get_maximal_semantics_with_context(self, context: Any) -> "TTRRecordType":
        return self.get_maximal_semantics(context)
