"""Tests for Java-style ``MetaType`` binding in IF clauses."""

from __future__ import annotations

from dylan.action.meta_stub import MetaType, reset_all_meta_bindings
from dylan.tree.label.labels import ModalLabel, Requirement, TypeLabel, label_factory_create
from dylan.tree.modality import Modality
from dylan.tree.node import Node
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree
from dylan.type.dstype import ConstructedType, DSType


def test_meta_type_binds_requirement_ty() -> None:
    """``?ty(X)`` on an IF label matches ``?Ty(e>t)`` on the pointed node."""
    reset_all_meta_bindings()
    tree = Tree()
    tree.pointed_node.labels = [Requirement(TypeLabel(ConstructedType(DSType.e, DSType.t)))]
    if_lab = label_factory_create("?ty(X)")
    assert if_lab.check(tree.pointed_node) is True
    reset_all_meta_bindings()


def test_modal_ty_metavariables_match_constructed_type() -> None:
    """``<\\/1>ty(Y>X)`` binds ``Y`` and ``X`` against daughter ``Ty`` labels."""
    reset_all_meta_bindings()
    tree = Tree()
    tree[NodeAddress("01")] = Node(NodeAddress("01"))
    tree[NodeAddress("01")].add_label(TypeLabel(ConstructedType(DSType.e, DSType.t)))
    tree.pointer = NodeAddress()
    mod = Modality.parse("<\\/1>")
    lab = ModalLabel(mod, [TypeLabel(ConstructedType(MetaType.get("Y"), MetaType.get("X")))])
    assert lab.check_with_tuple_as_context(tree, None) is True
    reset_all_meta_bindings()
