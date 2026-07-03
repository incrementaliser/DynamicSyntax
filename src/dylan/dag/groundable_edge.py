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

    def __init__(
        self,
        actions: list | None = None,
        word: object | None = None,
        edge_id: int = 0,
        overarching_repairing_edge: "VirtualRepairingEdge | None" = None,
    ) -> None:
        """Optionally attach the virtual repair edge used for traversal ordering (Java field)."""
        super().__init__(actions, word, edge_id)  # type: ignore[arg-type]
        self.overarching_repairing_edge = overarching_repairing_edge


class VirtualRepairingEdge(GroundableEdge):
    """Virtual edge connecting repaired context to a recomputed parse tuple."""

    def __init__(
        self,
        actions: list | None = None,
        word: object | None = None,
        edge_id: int = 0,
        length: int = 0,
    ) -> None:
        """Record backtrack distance for edge ordering (Java ``VirtualRepairingEdge.length``)."""
        super().__init__(actions, word, edge_id)  # type: ignore[arg-type]
        self.length = length


class ActionReplayEdge(GroundableEdge):
    """Edge that replays backtracked actions at a right-edge indicator."""


GroundableEdge.groundFor = GroundableEdge.ground_for  # type: ignore[attr-defined]
GroundableEdge.isGroundedFor = GroundableEdge.is_grounded_for  # type: ignore[attr-defined]
GroundableEdge.setRepairable = GroundableEdge.set_repairable  # type: ignore[attr-defined]
GroundableEdge.isRepairable = GroundableEdge.is_repairable  # type: ignore[attr-defined]
