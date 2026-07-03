"""Induction-time DAG used by :class:`dylan.induction.em_learner.ttr_hypothesiser.TTRHypothesiser` (Java ``DAGInductionState``)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from dylan.dag.dag_induction_tuple import DAGInductionTuple
from dylan.dag.groundable_edge import CompletionEdge, GroundableEdge, VirtualRepairingEdge
from dylan.dag.uttered_word import UtteredWord
from dylan.dag.word_level_context_dag import WordLevelContextDAG, _edge_action_name
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.tree.tree import Tree

if TYPE_CHECKING:
    from dylan.action.action import Action

logger = logging.getLogger(__name__)


def _tree_set_order(
    edges: list[GroundableEdge],
    cmp: Callable[[GroundableEdge, GroundableEdge], int],
) -> list[GroundableEdge]:
    """Insert *edges* into a binary search tree using *cmp* and return the in-order walk.

    Java stores out-edges in ``TreeSet<E>(edgeComparator)``; the comparator never
    returns 0, so ties (``+1``) send the new element to the right subtree, which
    preserves insertion order among equal elements.  Emulating the BST insertion
    reproduces that exact order.
    """
    root: list[Any] | None = None
    for e in edges:
        if root is None:
            root = [e, None, None]
            continue
        node = root
        while True:
            c = cmp(e, node[0])
            if c < 0:
                if node[1] is None:
                    node[1] = [e, None, None]
                    break
                node = node[1]
            elif c > 0:
                if node[2] is None:
                    node[2] = [e, None, None]
                    break
                node = node[2]
            else:
                break  # Java TreeSet.add: element already present, keep existing
    out: list[GroundableEdge] = []
    stack: list[list[Any]] = []
    node = root
    while node is not None or stack:
        while node is not None:
            stack.append(node)
            node = node[1]
        node = stack.pop()
        out.append(node[0])
        node = node[2]
    return out


class DAGInductionState(WordLevelContextDAG):
    """Word-level DAG specialised to carry :class:`DAGInductionTuple` nodes."""

    def __init__(
        self,
        start: "Tree | list[UtteredWord] | None" = None,
        words: list[UtteredWord] | None = None,
        gold_target: TTRRecordType | None = None,
    ) -> None:
        """Build with an initial tree and optional uttered-word list (Java overloads)."""
        self._gold_target: TTRRecordType | None = gold_target.clone() if gold_target is not None else None
        super().__init__()
        if isinstance(start, list):
            # Java ``DAG(Tree, List<UtteredWord>)`` pushes words with ``i = size-1 .. 0`` so ``peek()`` is the
            # first token of the sentence; mirror that with a list stack whose top is ``[-1]``.
            self.word_stack = list(reversed(start))
            start = Tree()
        elif words is not None:
            self.word_stack = list(reversed(words))
        # Replace the initial root with an induction tuple of the supplied tree.
        if start is None:
            start = Tree()
        self.root = self._make_induction_tuple(start)
        self._register(self.root)
        self.cur = self.root
        self.first_tuple_after_last_word = self.cur

    # ---------------- factory helpers ----------------

    def _make_induction_tuple(self, t: Tree) -> DAGInductionTuple:
        """Allocate a new :class:`DAGInductionTuple` and bookkeep its id (Java ``getNewTuple``)."""
        new_id = len(self.id_pool_nodes) + 1
        self.id_pool_nodes.append(new_id)
        tup = DAGInductionTuple(t, new_id)
        if self._gold_target is not None:
            tup.set_gold_target_type(self._gold_target)
        return tup

    def get_new_tuple(self, t: Tree) -> DAGInductionTuple:
        """Create a new :class:`DAGInductionTuple` for tree *t* (Java override)."""
        tup = self._make_induction_tuple(t)
        self._register(tup)
        return tup

    # ---------------- induction-specific overrides ----------------

    def exec_action(self, action: "Action", word: UtteredWord | None) -> DAGInductionTuple | None:
        """Apply *action* to a clone of :attr:`cur`'s tree and add a child tuple if successful (Java ``execAction``)."""
        cur = self.cur
        assert isinstance(cur, DAGInductionTuple)
        cur_tree = cur.get_tree()
        res = action.exec_tuple_context(cur_tree.clone(), cur)
        if res is None:
            return None
        if self.loop_detected(res):
            logger.warning("Detected infinite branch. Not extending DAG.")
            for t in self.last_n:
                logger.warning("%s", t)
            return None
        edge = self.get_new_edge([action.instantiate()], word)
        target = self.get_new_tuple(res)
        self.add_edge(edge, cur, target)
        target.set_depth(cur.get_depth() + 1)
        return target

    def add_child(self, *args: Any) -> Any:  # type: ignore[override]
        """Java overloads: ``addChild(DAGInductionTuple, Action, UtteredWord)`` and ``addChild(Tree, Action, UtteredWord)``."""
        if len(args) == 3:
            t_or_tuple, action, word = args
            if isinstance(t_or_tuple, DAGInductionTuple):
                return self._add_child_tuple(t_or_tuple, action, word)
            if isinstance(t_or_tuple, Tree):
                return self._add_child_tree(t_or_tuple, action, word)
        return super().add_child(*args)

    def _add_child_tuple(
        self,
        target_tuple: DAGInductionTuple,
        action: "Action",
        word: UtteredWord | None,
    ) -> DAGInductionTuple | None:
        """Java ``addChild(DAGInductionTuple, Action, UtteredWord)``."""
        if target_tuple is None:
            return None
        from dylan.induction.em_learner.tree_hypothesis import TreeHypothesis

        if not isinstance(action, TreeHypothesis) and self.loop_detected(target_tuple.get_tree()):
            logger.info("Detected infinite branch. Not extending DAG.")
            return None
        edge = self.get_new_edge([action], word)
        target = self.get_new_tuple(target_tuple.get_tree())
        cur = self.cur
        assert isinstance(cur, DAGInductionTuple)
        target.set_target(cur.get_target_tree())
        target.set_non_head_target(cur.get_non_head_target())
        self.add_edge(edge, self.cur, target)
        return target

    def _add_child_tree(
        self,
        tree: Tree,
        action: "Action",
        word: UtteredWord | None,
    ) -> bool:
        """Java ``addChild(Tree, Action, UtteredWord)``."""
        if tree is None:
            return False
        from dylan.induction.em_learner.tree_hypothesis import TreeHypothesis

        if not isinstance(action, TreeHypothesis) and self.loop_detected(tree):
            logger.info("Detected infinite branch. Not extending DAG.")
            logger.debug("Action was: %s", action)
            logger.debug("tree was: %s", tree)
            return False
        edge = self.get_new_edge([action.instantiate()], word)
        target = self.get_new_tuple(tree)
        cur = self.cur
        assert isinstance(cur, DAGInductionTuple)
        target.set_target(cur.get_target_tree().clone())
        target.set_non_head_target(cur.get_non_head_target().clone())
        action_name = action.get_name() if hasattr(action, "get_name") else ""
        if action_name.startswith("hyp-build-cn-e"):
            target.set_target(target.get_target_tree().merge_tree(tree))
        target.set_depth(cur.get_depth() + 1)
        self.add_edge(edge, self.cur, target)
        return True

    def add_axiom(self, *args: Any) -> Any:  # type: ignore[override]
        """Java ``addAxiom(List<Action>, UtteredWord)``."""
        if not args:
            return super().add_axiom()
        actions, word = args
        axiom = self.get_new_tuple(Tree())
        edge = self.get_new_edge(list(actions), word)
        super().add_child(axiom, edge)
        return axiom

    def reset_to_first_tuple_after_last_word(self) -> None:  # type: ignore[override]
        """Match Java semantics: also clear :attr:`word_stack` and ``lastN``."""
        if self.first_tuple_after_last_word is not None:
            self.set_current_tuple(self.first_tuple_after_last_word)
            self.set_exhausted(False)
            self.word_stack.clear()
            self.reset_last_n()

    def _compare_edges_by_endpoint_completeness(  # type: ignore[override]
        self,
        o1: GroundableEdge,
        o2: GroundableEdge,
    ) -> int:
        """Faithful Java ``DAG.EdgeComparatorByEndPointCompleteness`` (never returns 0)."""
        if isinstance(o1, VirtualRepairingEdge) and isinstance(o2, VirtualRepairingEdge):
            if o1.length == o2.length:
                return 1
            return o1.length - o2.length
        if isinstance(o1, CompletionEdge) and isinstance(o2, CompletionEdge):
            pass
        elif isinstance(o1, CompletionEdge):
            return 1
        elif isinstance(o2, CompletionEdge):
            return -1
        n1, n2 = _edge_action_name(o1), _edge_action_name(o2)
        if n1 == "completion" and "hyp-build-cn-e" in n2:
            return -1
        if n2 == "completion" and "hyp-build-cn-e" in n1:
            return 1
        if n1.startswith("anticipation0") and "hyp-build-cn-e" in n2:
            return -1
        if n2.startswith("anticipation0") and "hyp-build-cn-e" in n1:
            return 1
        if n1.startswith("hyp-sem") and (n2 == "thinning" or n2.startswith("hyp-cn-decomp")):
            return -1
        if n2.startswith("hyp-sem") and (n1 == "thinning" or n1.startswith("hyp-cn-decomp")):
            return 1
        if n1 == "thinning" and n2.startswith("hyp-cn-decomp"):
            return -1
        if n2 == "thinning" and n1.startswith("hyp-cn-decomp"):
            return 1
        if n1 == "thinning" and n2.startswith("hyp-build-et"):
            return -1
        if n2 == "thinning" and n1.startswith("hyp-build-et"):
            return 1
        if n1 == "thinning" and n2.startswith("hyp-build-cn-e"):
            return -1
        if n2 == "thinning" and n1.startswith("hyp-build-cn-e"):
            return 1
        if "hyp-build-cn-e-1" in n1 and "hyp-build-cn-e-0" in n2:
            return -1
        if "hyp-build-cn-e-0" in n1 and "hyp-build-cn-e-1" in n2:
            return 1
        end1 = o1.dst
        end2 = o2.dst
        t1 = end1.get_tree().get_incompleteness_measure()
        t2 = end2.get_tree().get_incompleteness_measure()
        if t1 > t2:
            return 1
        if t2 > t1:
            return -1
        return 1

    def get_out_edges(self, n: DAGInductionTuple | None = None) -> list[GroundableEdge]:
        """Out-edges in Java ``TreeSet`` order under the endpoint-completeness comparator."""
        u = n if n is not None else self.cur
        raw = list(self._out.get(u, ()))
        return _tree_set_order(raw, self._compare_edges_by_endpoint_completeness)

    def more_unseen_edges(self) -> bool:
        """Whether the current tuple has an unseen out-edge (Java ``DAG.moreUnseenEdges``)."""
        for edge in self.get_out_edges(self.cur):
            if not edge.has_been_seen():
                logger.info("found unseen edge: %s", edge)
                return True
        logger.info("didn't find unseen edge")
        return False

    def attempt_backtrack(self) -> bool:
        """Backtrack until a tuple with unseen out-edges is reached (Java ``DAG.attemptBacktrack``)."""
        while not self.more_unseen_edges():
            if self.get_parent(self.cur) is None:
                logger.info("cannot backtrack from %s", self.cur)
                return False
            backover = self.get_parent_edge(self.cur)
            assert backover is not None
            if backover.word is not None:
                self.word_stack.append(backover.word)
                logger.debug("adding word to stack, now: %s", self.word_stack)
            gone = self.go_up_once()
            assert gone is not None
            gone.set_seen(True)
            gone.set_in_context(False)
        logger.debug("Backtrack succeeded")
        return True

    def go_first(self) -> GroundableEdge | None:
        """Traverse the first unseen edge (Java ``DAG.goFirst``, not parser ``WordLevelContextDAG``)."""
        if self.out_degree(self.cur) == 0:
            return None
        for edge in self.get_out_edges(self.cur):
            if edge.has_been_seen():
                continue
            logger.info("Going forward first along: %s", edge)
            child = edge.dst
            assert isinstance(child, DAGInductionTuple)
            edge.set_in_context(True)
            self.cur = child
            self.update_last_n()
            logger.info("Depth is now: %s", self.cur.get_depth())
            if edge.word is not None:
                if not self.word_stack or self.word_stack[-1] != edge.word:
                    logger.debug(
                        "Skipping edge %s; word stack top is %s",
                        edge,
                        self.word_stack[-1].word if self.word_stack else None,
                    )
                    continue
                self.word_stack.pop()
            return edge
        return None


DAGInductionState.execAction = DAGInductionState.exec_action  # type: ignore[attr-defined]
DAGInductionState.addAxiom = DAGInductionState.add_axiom  # type: ignore[attr-defined]
DAGInductionState.resetToFirstTupleAfterLastWord = DAGInductionState.reset_to_first_tuple_after_last_word  # type: ignore[attr-defined]
