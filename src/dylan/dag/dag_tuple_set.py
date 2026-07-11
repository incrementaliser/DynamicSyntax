"""Node in a :class:`~dylan.induction.em_learner.word_hypothesis.WordHypothesis` graph (Java ``DAGTupleSet``)."""

from __future__ import annotations

from dylan.dag.parser_tuple import ParserTuple
from dylan.tree.tree import Tree


class DAGTupleSet(list[ParserTuple]):
    """A set of parser tuples identified by a unique id (Java ``qmul.ds.dag.DAGTupleSet``)."""

    def __init__(self, id_or_start: int | ParserTuple = 0) -> None:
        """Construct with a numeric id, or seed from a single :class:`ParserTuple`."""
        super().__init__()
        if isinstance(id_or_start, ParserTuple):
            self.id = 0
            self.append(id_or_start)
        else:
            self.id = int(id_or_start)

    @staticmethod
    def get_new_tuple_set(
        id_pool: list[int],
        tree: Tree | None = None,
    ) -> "DAGTupleSet":
        """Allocate a fresh id from *id_pool* and optionally seed with *tree* (Java ``getNewTupleSet``)."""
        new_id = len(id_pool) + 1
        result = DAGTupleSet(len(id_pool) + 1)
        id_pool.append(new_id)
        if tree is not None:
            result.append(ParserTuple(tree))
        return result

    def add_tree(self, tree: Tree) -> None:
        """Append a new :class:`ParserTuple` wrapping *tree* (Java ``add(Tree)``)."""
        self.append(ParserTuple(tree))

    def get_id(self) -> int:
        """Return this node id (Java ``getId``)."""
        return self.id

    def __eq__(self, other: object) -> bool:
        """Java ``equals``: identity by ``id`` only."""
        if self is other:
            return True
        if not isinstance(other, DAGTupleSet):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        """Java ``hashCode``: hash of ``id``."""
        return hash(self.id)

    def __str__(self) -> str:
        """Java ``toString``: list rendering of contained tuples."""
        return super().__str__()


DAGTupleSet.getNewTupleSet = DAGTupleSet.get_new_tuple_set  # type: ignore[attr-defined]
DAGTupleSet.getId = DAGTupleSet.get_id  # type: ignore[attr-defined]
DAGTupleSet.addTree = DAGTupleSet.add_tree  # type: ignore[attr-defined]
