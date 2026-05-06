"""DAG edge base (Java `DAGEdge`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dylan.action.action import Action
    from dylan.dag.uttered_word import UtteredWord


class DAGEdge:
    """Directed edge carrying actions and an uttered word."""

    SEEN = 1
    IN_CONTEXT = 3
    REPAIRABLE = 5

    def __init__(
        self,
        actions: list[Action] | None = None,
        word: UtteredWord | None = None,
        edge_id: int = 0,
    ) -> None:
        self.actions = actions if actions is not None else []
        self.word = word
        self.edge_id = edge_id
        self._props: set[int] = set()
        self.src: object | None = None
        self.dst: object | None = None
        self._grounded_for: set[str] = set()

    def has_been_seen(self) -> bool:
        """Return whether this edge has already been explored."""
        return self.SEEN in self._props

    def set_seen(self, b: bool) -> None:
        """Mark this edge as explored or unexplored."""
        if b:
            self._props.add(self.SEEN)
        else:
            self._props.discard(self.SEEN)

    def in_context(self) -> bool:
        """Return whether this edge is currently in the active context path."""
        return self.IN_CONTEXT in self._props

    def set_in_context(self, b: bool) -> None:
        """Mark whether this edge is on the active context path."""
        if b:
            self._props.add(self.IN_CONTEXT)
        else:
            self._props.discard(self.IN_CONTEXT)

    def get_action(self) -> Action:
        """Return the first action on the edge (Java ``getAction``)."""
        return self.actions[0]

    def get_actions(self) -> list[Action]:
        """Return all actions carried by this edge."""
        return list(self.actions)

    def get_word(self) -> UtteredWord | None:
        """Return the word carried by this edge."""
        return self.word

    def is_repairable(self) -> bool:
        """Return whether this edge may be targeted by repair."""
        return self.REPAIRABLE in self._props

    def set_repairable(self, b: bool) -> None:
        """Set the repairable flag."""
        if b:
            self._props.add(self.REPAIRABLE)
        else:
            self._props.discard(self.REPAIRABLE)

    def ground_for(self, speaker: str) -> None:
        """Mark this edge as grounded for *speaker*."""
        self._grounded_for.add(speaker)

    def is_grounded_for(self, speaker: str) -> bool:
        """Return whether this edge is grounded for *speaker*."""
        return speaker in self._grounded_for

    def grounded_speakers(self) -> set[str]:
        """Return speakers for whom this edge is grounded."""
        return set(self._grounded_for)

    def __hash__(self) -> int:
        """Hash by edge id."""
        return hash(self.edge_id)

    def __lt__(self, other: DAGEdge) -> bool:
        """Sort edges in creation order."""
        return self.edge_id < other.edge_id

    def __repr__(self) -> str:
        """Return a compact debug representation."""
        return f"{type(self).__name__}(id={self.edge_id}, word={self.word!r}, actions={len(self.actions)})"


DAGEdge.hasBeenSeen = DAGEdge.has_been_seen  # type: ignore[attr-defined]
DAGEdge.setSeen = DAGEdge.set_seen  # type: ignore[attr-defined]
DAGEdge.inContext = DAGEdge.in_context  # type: ignore[attr-defined]
DAGEdge.setInContext = DAGEdge.set_in_context  # type: ignore[attr-defined]
DAGEdge.getAction = DAGEdge.get_action  # type: ignore[attr-defined]
DAGEdge.getActions = DAGEdge.get_actions  # type: ignore[attr-defined]
DAGEdge.getWord = DAGEdge.get_word  # type: ignore[attr-defined]
DAGEdge.isRepairable = DAGEdge.is_repairable  # type: ignore[attr-defined]
DAGEdge.setRepairable = DAGEdge.set_repairable  # type: ignore[attr-defined]
DAGEdge.groundFor = DAGEdge.ground_for  # type: ignore[attr-defined]
DAGEdge.isGroundedFor = DAGEdge.is_grounded_for  # type: ignore[attr-defined]
