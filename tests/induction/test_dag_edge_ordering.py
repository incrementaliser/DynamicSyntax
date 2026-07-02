"""Tests for DAG edge traversal order (Java ``EdgeComparatorByEndPointCompleteness``)."""

from __future__ import annotations

from dylan.dag.dag_induction_state import DAGInductionState
from dylan.induction.em_learner.tree_hypothesis import TreeHypothesis
from dylan.tree.label.labels import label_factory_create
from dylan.tree.node import Node
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree


def test_out_edges_prefer_more_complete_destination() -> None:
    """Out-edges are ordered by lower incompleteness, then more nodes (Java ``goFirst``)."""
    state = DAGInductionState()
    parent = state.cur
    less_complete = state.get_new_tuple(Tree())
    more_complete_tree = Tree()
    more_complete_tree[NodeAddress("00")] = Node(NodeAddress("00"))
    more_complete_tree[NodeAddress("00")].add_label(label_factory_create("Ty(e)"))
    more_complete = state.get_new_tuple(more_complete_tree)
    state.add_child(less_complete, TreeHypothesis([], less_complete.get_tree()), None)
    state.add_child(more_complete, TreeHypothesis([], more_complete.get_tree()), None)
    ordered = state.get_out_edges(parent)
    assert len(ordered) == 2
    m0 = ordered[0].dst.get_tree().get_incompleteness_measure()  # type: ignore[union-attr]
    m1 = ordered[1].dst.get_tree().get_incompleteness_measure()  # type: ignore[union-attr]
    assert m0 < m1
