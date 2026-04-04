"""Tree / node address invariants."""

from __future__ import annotations

from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree


def test_tree_has_axiom_requirement() -> None:
    t = Tree()
    assert t.root_addr == NodeAddress()
    assert any("?" in str(lab) for lab in t.pointed_node.labels)


def test_clone_is_distinct() -> None:
    a = Tree()
    b = a.clone()
    assert a is not b
    assert a.pointer == b.pointer
