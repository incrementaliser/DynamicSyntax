"""Groundable word edge (Java `GroundableEdge`)."""

from __future__ import annotations

from dylan.dag.dag_edge import DAGEdge


class GroundableEdge(DAGEdge):
    """Computational actions + lexical action + word."""

    def traverse(self, dag: object) -> None:
        from dylan.dag.word_level_context_dag import WordLevelContextDAG

        assert isinstance(dag, WordLevelContextDAG)
        assert self.src == dag.cur
        dag.cur = self.dst  # type: ignore[assignment]
        self.set_in_context(True)

    def backtrack(self, dag: object) -> None:
        from dylan.dag.word_level_context_dag import WordLevelContextDAG

        assert isinstance(dag, WordLevelContextDAG)
        assert self.dst == dag.cur
        self.set_seen(True)
        self.set_in_context(False)
        dag.cur = self.src  # type: ignore[assignment]
