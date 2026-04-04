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

    def has_been_seen(self) -> bool:
        return self.SEEN in self._props

    def set_seen(self, b: bool) -> None:
        if b:
            self._props.add(self.SEEN)
        else:
            self._props.discard(self.SEEN)

    def in_context(self) -> bool:
        return self.IN_CONTEXT in self._props

    def set_in_context(self, b: bool) -> None:
        if b:
            self._props.add(self.IN_CONTEXT)
        else:
            self._props.discard(self.IN_CONTEXT)

    def get_action(self) -> Action:
        return self.actions[0]

    def __hash__(self) -> int:
        return hash(self.edge_id)

    def __lt__(self, other: DAGEdge) -> bool:
        return self.edge_id < other.edge_id
