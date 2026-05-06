"""Groundable word edge (Java `GroundableEdge`)."""

from __future__ import annotations

from dylan.dag.dag_edge import DAGEdge


class GroundableEdge(DAGEdge):
    """Computational actions + lexical action + word."""

    def traverse(self, dag: object) -> None:
        """Move the DAG cursor from source to destination."""
        from dylan.dag.word_level_context_dag import WordLevelContextDAG

        assert isinstance(dag, WordLevelContextDAG)
        assert self.src == dag.cur
        dag.cur = self.dst  # type: ignore[assignment]
        self.set_in_context(True)

    def backtrack(self, dag: object) -> None:
        """Backtrack the DAG cursor from destination to source."""
        from dylan.dag.word_level_context_dag import WordLevelContextDAG

        assert isinstance(dag, WordLevelContextDAG)
        assert self.dst == dag.cur
        self.set_seen(True)
        self.set_in_context(False)
        dag.cur = self.src  # type: ignore[assignment]


class CompletionEdge(GroundableEdge):
    """Edge created by parser completion actions."""


class BacktrackingEdge(GroundableEdge):
    """Edge marker used to initiate or represent local repair."""

    repair_init_prefix = "init-repair"


class VirtualRepairingEdge(GroundableEdge):
    """Virtual edge connecting repaired context to a recomputed parse tuple."""


class ActionReplayEdge(GroundableEdge):
    """Edge that replays backtracked actions at a right-edge indicator."""


GroundableEdge.groundFor = GroundableEdge.ground_for  # type: ignore[attr-defined]
GroundableEdge.isGroundedFor = GroundableEdge.is_grounded_for  # type: ignore[attr-defined]
GroundableEdge.setRepairable = GroundableEdge.set_repairable  # type: ignore[attr-defined]
GroundableEdge.isRepairable = GroundableEdge.is_repairable  # type: ignore[attr-defined]
