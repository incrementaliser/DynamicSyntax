"""Induction DAG state."""

from __future__ import annotations

from typing import Any, Iterable

from dylan.dag.uttered_word import UtteredWord
from dylan.dag.word_level_context_dag import WordLevelContextDAG
from dylan.induction.em_learner.dag_induction_tuple import DAGInductionTuple
from dylan.tree.tree import Tree


class DAGInductionState(WordLevelContextDAG):
    """Word-level DAG specialised for induction hypotheses."""

    def __init__(self, words: Iterable[UtteredWord] | None = None) -> None:
        """Create an induction DAG seeded with optional words."""
        super().__init__()
        self.word_stack = list(words or [])
        self.root = self.get_new_tuple(Tree())
        self.cur = self.root
        self.first_tuple_after_last_word = self.cur

    def get_new_tuple(self, t: Tree) -> DAGInductionTuple:
        """Create a new induction tuple."""
        new_id = len(self.id_pool_nodes) + 1
        self.id_pool_nodes.append(new_id)
        tup = DAGInductionTuple(t, new_id)
        self._register(tup)
        return tup

    def add_child(self, child: DAGInductionTuple, edge: Any, word: UtteredWord | None = None) -> DAGInductionTuple:
        """Add *child* below current tuple using *edge*."""
        if not hasattr(edge, "traverse"):
            edge = self.get_new_edge([edge], word)
        return super().add_child(child, edge)

    def get_child_count(self) -> int:
        """Return number of children below current tuple."""
        return self.out_degree(self.cur)


DAGInductionState.getNewTuple = DAGInductionState.get_new_tuple  # type: ignore[attr-defined]
DAGInductionState.addChild = DAGInductionState.add_child  # type: ignore[attr-defined]
DAGInductionState.getChildCount = DAGInductionState.get_child_count  # type: ignore[attr-defined]
