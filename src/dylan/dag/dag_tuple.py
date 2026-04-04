"""DAG node type (Java `DAGTuple`)."""

from __future__ import annotations

from dylan.dag.parser_tuple import ParserTuple
from dylan.tree.tree import Tree


class DAGTuple(ParserTuple):
    """Tuple identified by id inside the word-level DAG."""

    def __init__(self, tree: Tree | None = None, tuple_id: int = 0) -> None:
        super().__init__(tree)
        self.tuple_id = tuple_id
        self._depth = 0

    def __hash__(self) -> int:
        return hash(self.tuple_id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DAGTuple) and self.tuple_id == other.tuple_id

    def get_depth(self) -> int:
        return self._depth

    def set_depth(self, d: int) -> None:
        self._depth = d
