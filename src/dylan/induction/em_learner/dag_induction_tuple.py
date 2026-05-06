"""DAG tuple with induction target trees."""

from __future__ import annotations

from dylan.dag.dag_tuple import DAGTuple
from dylan.tree.tree import Tree


class DAGInductionTuple(DAGTuple):
    """DAG tuple carrying current target and non-head target trees."""

    def __init__(self, tree: Tree | None = None, tuple_id: int = 0) -> None:
        """Create an induction tuple."""
        super().__init__(tree, tuple_id)
        self.cur_target = Tree()
        self.cur_non_head_target = Tree()

    def set_target(self, tree: Tree) -> None:
        """Set current target tree."""
        self.cur_target = tree

    def get_target_tree(self) -> Tree:
        """Return current target tree."""
        return self.cur_target

    def set_non_head_target(self, tree: Tree) -> None:
        """Set current non-head target tree."""
        self.cur_non_head_target = tree

    def get_non_head_target(self) -> Tree:
        """Return current non-head target tree."""
        return self.cur_non_head_target


DAGInductionTuple.setTarget = DAGInductionTuple.set_target  # type: ignore[attr-defined]
DAGInductionTuple.getTargetTree = DAGInductionTuple.get_target_tree  # type: ignore[attr-defined]
DAGInductionTuple.setNonHeadTarget = DAGInductionTuple.set_non_head_target  # type: ignore[attr-defined]
DAGInductionTuple.getNonHeadTarget = DAGInductionTuple.get_non_head_target  # type: ignore[attr-defined]
