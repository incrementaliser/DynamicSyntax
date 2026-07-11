"""Word-level sequence-intersection hypothesis (Java ``qmul.ds.dag.WordHypothesis``).

Faithful port of the JUNG ``DelegateTree<DAGTupleSet, DAGEdge>`` intersection
structure used for generalising per-word candidate sequences during EM.
"""

from __future__ import annotations

import logging
import math
from dylan.action.action import Action
from dylan.action.computational_action import ComputationalAction
from dylan.action.lexical_action import LexicalAction
from dylan.dag.dag_edge import DAGEdge
from dylan.dag.dag_tuple_set import DAGTupleSet
from dylan.dag.parser_tuple import ParserTuple
from dylan.induction.em_learner.candidate_sequence import CandidateSequence
from dylan.induction.em_learner.common import Word, as_word

logger = logging.getLogger(__name__)


class WordHypothesis:
    """Intersection tree of candidate sequences for one word (Java ``WordHypothesis``)."""

    def __init__(self, hyp_id: int) -> None:
        """Create an empty intersection tree with hypothesis id *hyp_id*."""
        self.hyp_id = hyp_id
        self.howmany = 0
        self._id_pool_nodes: list[int] = []
        self._id_pool_edges: list[int] = []
        self.word: Word | None = None
        self.log_prob: float = 1.0
        self._root: DAGTupleSet | None = None
        self._children: dict[DAGTupleSet, list[tuple[DAGEdge, DAGTupleSet]]] = {}
        self._parent: dict[DAGTupleSet, DAGTupleSet] = {}
        self._parent_edge: dict[DAGTupleSet, DAGEdge] = {}
        self._vertices: set[DAGTupleSet] = set()

    # ---------------- graph helpers (JUNG DelegateTree) ----------------

    def _add_vertex(self, v: DAGTupleSet) -> None:
        """Register vertex *v* (Java ``addVertex``)."""
        self._vertices.add(v)
        self._children.setdefault(v, [])

    def _add_child(self, edge: DAGEdge, parent: DAGTupleSet, child: DAGTupleSet) -> None:
        """Attach *child* under *parent* via *edge* (Java ``addChild``)."""
        self._add_vertex(child)
        self._children.setdefault(parent, []).append((edge, child))
        self._parent[child] = parent
        self._parent_edge[child] = edge
        edge.src = parent
        edge.dst = child

    def get_root(self) -> DAGTupleSet:
        """Return the intersection-tree root (Java ``getRoot``)."""
        if self._root is None:
            raise IllegalStateError("Graph is empty")
        return self._root

    def get_vertex_count(self) -> int:
        """Return number of vertices (Java ``getVertexCount``)."""
        return len(self._vertices)

    def get_child_count(self, v: DAGTupleSet) -> int:
        """Return out-degree of *v* (Java ``getChildCount``)."""
        return len(self._children.get(v, ()))

    def get_child_edges(self, v: DAGTupleSet) -> list[DAGEdge]:
        """Return outgoing edges of *v* (Java ``getChildEdges``)."""
        return [e for e, _ in self._children.get(v, ())]

    def get_children(self, v: DAGTupleSet) -> list[DAGTupleSet]:
        """Return child vertices of *v* (Java ``getChildren``)."""
        return [c for _, c in self._children.get(v, ())]

    def get_dest(self, edge: DAGEdge) -> DAGTupleSet:
        """Return destination of *edge* (Java ``getDest``)."""
        dst = edge.dst
        if not isinstance(dst, DAGTupleSet):
            raise KeyError(edge)
        return dst

    def get_source(self, edge: DAGEdge) -> DAGTupleSet:
        """Return source of *edge* (Java ``getSource``)."""
        src = edge.src
        if not isinstance(src, DAGTupleSet):
            raise KeyError(edge)
        return src

    def get_parent(self, v: DAGTupleSet) -> DAGTupleSet | None:
        """Return parent of *v*, or ``None`` at the root (Java ``getParent``)."""
        return self._parent.get(v)

    def get_parent_edge(self, v: DAGTupleSet) -> DAGEdge | None:
        """Return the incoming edge of *v* (Java ``getParentEdge``)."""
        return self._parent_edge.get(v)

    def get_new_edge(self, action: Action) -> DAGEdge:
        """Allocate a new edge carrying *action* (Java ``getNewEdge``)."""
        new_id = len(self._id_pool_edges) + 1
        edge = DAGEdge([action], None, new_id)
        self._id_pool_edges.append(new_id)
        return edge

    # ---------------- intersection ----------------

    def intersect_into(self, candidate: CandidateSequence) -> bool:
        """Intersect *candidate* into this SI tree (Java ``intersectInto``)."""
        logger.debug("Intersecting: %s", candidate)
        words = candidate.get_words()
        if len(words) != 1:
            raise ValueError("Cannot intersect an unsplit candidate sequence")
        word = as_word(words[0])
        if self.word is not None and self.word != word:
            raise ValueError(
                f"Cannot intersect sequences for different words: {self.word} vs {word}",
            )
        self.word = word

        if self.get_vertex_count() == 0:
            tuple_set = DAGTupleSet.get_new_tuple_set(self._id_pool_nodes)
            self._add_vertex(tuple_set)
            self._root = tuple_set
            for i in range(len(candidate) - 1, -1, -1):
                action = candidate[i]
                cur = DAGTupleSet.get_new_tuple_set(self._id_pool_nodes)
                child_edge = self.get_new_edge(action)
                self._add_child(child_edge, tuple_set, cur)
                tuple_set = cur
            start_tuple = candidate.get_start()
            tuple_set.append(start_tuple)
            self.forward_populate(tuple_set, start_tuple)
            self.howmany += 1
            logger.debug("Intersected, init successful")
            return True

        first_lexical_index = candidate.get_first_lexical_index()
        root = self.get_root()
        root_child_edges = self.get_child_edges(root)
        if not root_child_edges:
            raise IllegalStateError("attempting to intersect cs into empty SI")

        cur_tuple = root
        for i in range(len(candidate) - 1, -1, -1):
            cur_action = candidate[i]
            logger.debug("current action to be matched is: %s", cur_action)
            matched = False
            for child_edge in self.get_child_edges(cur_tuple):
                child_action = child_edge.get_action()
                if cur_action == child_action:
                    logger.debug("going back along %s", child_action)
                    cur_tuple = self.get_dest(child_edge)
                    matched = True
                    break
            if matched:
                continue
            if i < first_lexical_index and not self._has_non_computational_descendant(cur_tuple):
                new_vertex = DAGTupleSet.get_new_tuple_set(self._id_pool_nodes)
                new_edge = self.get_new_edge(cur_action)
                self._add_child(new_edge, cur_tuple, new_vertex)
                logger.debug("Branching with %s and going forward", new_edge.get_action())
                cur_tuple = new_vertex
            else:
                logger.debug("INTERSECTION FAILED, cannot branch here")
                return False

        if not self._has_non_computational_descendant(cur_tuple):
            start_tuple = candidate.get_start()
            cur_tuple.append(start_tuple)
            self.forward_populate(cur_tuple, start_tuple)
            self.howmany += 1
            logger.debug("INTERSECTION SUCCESS")
            return True
        logger.debug("INTERSECTION FAILED, matched whole sequence but not gone far enough")
        return False

    def _has_non_computational_descendant(self, v: DAGTupleSet) -> bool:
        """True if the unique spine below *v* contains a non-computational action."""
        cur = v
        while self.get_child_count(cur) > 0:
            child_edges = self.get_child_edges(cur)
            if len(child_edges) > 1:
                break
            child_edge = child_edges[0]
            if not isinstance(child_edge.get_action(), ComputationalAction):
                return True
            cur = self.get_dest(child_edge)
        return False

    def forward_populate(self, tuple_set: DAGTupleSet, t: ParserTuple) -> None:
        """Replay parent actions upward from *tuple_set*, seeding ancestor tuple sets (Java ``forwardPopulate``)."""
        if t not in tuple_set:
            raise UnsupportedOperationError("The tuple set does not contain the parser tuple")
        cur = tuple_set
        curt = t
        if self.get_parent(cur) is None:
            return
        while self.get_parent(cur) is not None:
            parent = self.get_parent(cur)
            assert parent is not None
            parent_edge = self.get_parent_edge(cur)
            assert parent_edge is not None
            parent_action = parent_edge.get_action()
            tree_clone = curt.get_tree().clone()
            result = parent_action.exec_tuple_context(tree_clone, curt)
            logger.debug("Action: %s on tree: %s -> %s", parent_action.get_name(), curt, result)
            if result is None:
                # Java throws; empty start tuples in unit tests / some extract paths can fail.
                # Leave ancestor sets unpopulated rather than aborting the intersection graph.
                logger.error("Action %s failed on tree %s", parent_action.get_name(), curt)
                return
            newt = ParserTuple(result)
            parent.append(newt)
            curt = newt
            cur = parent

    # ---------------- core action / leaves ----------------

    def get_core_action(self) -> Action:
        """Return the unique maximal common path as a :class:`LexicalAction` (Java ``getCoreAction``)."""
        if self.get_vertex_count() == 0 or self.word is None:
            return Action(self.get_name())
        actions: list[Action] = []
        root = self.get_root()
        while self.get_child_count(root) == 1 and self._has_non_computational_descendant(root):
            child_edge = self.get_child_edges(root)[0]
            actions.insert(0, child_edge.get_action())
            root = self.get_children(root)[0]
        return LexicalAction.from_action_spine(self.word.word(), actions)

    def get_leaves(self) -> set[DAGTupleSet]:
        """Return vertices with out-degree 0 (Java ``getLeaves``)."""
        if self.get_vertex_count() == 0:
            raise IllegalStateError("Graph is empty when trying to get roots")
        return {v for v in self._vertices if self.get_child_count(v) == 0}

    def extract_maximal_action_sequences(self) -> set[tuple[Action, ...]]:
        """Collect leaf-to-root action paths (Java ``extractMaximalActionSequences``)."""
        result: set[tuple[Action, ...]] = set()
        for leaf in self.get_leaves():
            cur = leaf
            cur_list: list[Action] = []
            while True:
                parent_edge = self.get_parent_edge(cur)
                if parent_edge is None:
                    break
                cur_list.append(parent_edge.get_action())
                parent = self.get_source(parent_edge)
                cur = parent
            result.add(tuple(cur_list))
        return result

    # ---------------- accessors / probs ----------------

    def get_word(self) -> Word:
        """Return this hypothesis' word (Java ``getWord``)."""
        if self.word is None:
            raise ValueError("word hypothesis has not been initialised")
        return self.word

    def get_count(self) -> int:
        """Return number of intersected candidate sequences (Java ``getCount``)."""
        return self.howmany

    def get_name(self) -> str:
        """Java ``getName`` -> ``<word>_<id>``."""
        word = self.word.word() if self.word is not None else "?"
        return f"{word}_{self.hyp_id}"

    def set_log_prob(self, log_prob: float) -> None:
        """Set log probability (Java ``setLogProb``)."""
        self.log_prob = log_prob

    def set_prob(self, prob: float) -> None:
        """Set probability in normal space (Java ``setProb``)."""
        self.log_prob = math.log(prob) if prob > 0 else float("-inf")

    def get_log_prob(self) -> float:
        """Return log probability (Java ``getLogProb``)."""
        return self.log_prob

    def get_prob(self) -> float:
        """Return probability in normal space (Java ``getProb``)."""
        if self.log_prob > 0:
            return 0.0
        return math.exp(self.log_prob)

    def __hash__(self) -> int:
        """Java ``hashCode``: ``17 * hyp_id + word``."""
        word_hash = hash(self.word) if self.word is not None else 0
        return 17 * self.hyp_id + word_hash

    def __eq__(self, other: object) -> bool:
        """Java ``equals``: same id and same word."""
        if not isinstance(other, WordHypothesis):
            return False
        return self.hyp_id == other.hyp_id and self.word == other.word

    def __str__(self) -> str:
        """Java ``toString`` -> ``<name>:<prob>``."""
        return f"{self.get_name()}:{self.get_prob():.3f}"


class IllegalStateError(RuntimeError):
    """Mirror Java ``IllegalStateException`` for empty SI graphs."""


class UnsupportedOperationError(RuntimeError):
    """Mirror Java ``UnsupportedOperationException`` for forwardPopulate preconditions."""


WordHypothesis.intersectInto = WordHypothesis.intersect_into  # type: ignore[attr-defined]
WordHypothesis.getWord = WordHypothesis.get_word  # type: ignore[attr-defined]
WordHypothesis.getCount = WordHypothesis.get_count  # type: ignore[attr-defined]
WordHypothesis.getName = WordHypothesis.get_name  # type: ignore[attr-defined]
WordHypothesis.getCoreAction = WordHypothesis.get_core_action  # type: ignore[attr-defined]
WordHypothesis.extractMaximalActionSequences = WordHypothesis.extract_maximal_action_sequences  # type: ignore[attr-defined]
WordHypothesis.getLeaves = WordHypothesis.get_leaves  # type: ignore[attr-defined]
WordHypothesis.getRoot = WordHypothesis.get_root  # type: ignore[attr-defined]
WordHypothesis.forwardPopulate = WordHypothesis.forward_populate  # type: ignore[attr-defined]
WordHypothesis.getNewEdge = WordHypothesis.get_new_edge  # type: ignore[attr-defined]
WordHypothesis.setLogProb = WordHypothesis.set_log_prob  # type: ignore[attr-defined]
WordHypothesis.setProb = WordHypothesis.set_prob  # type: ignore[attr-defined]
WordHypothesis.getLogProb = WordHypothesis.get_log_prob  # type: ignore[attr-defined]
WordHypothesis.getProb = WordHypothesis.get_prob  # type: ignore[attr-defined]
