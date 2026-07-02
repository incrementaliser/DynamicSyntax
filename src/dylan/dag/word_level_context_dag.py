"""Word-level context DAG (partial Java `WordLevelContextDAG`)."""

from __future__ import annotations

import logging
from functools import cmp_to_key
from typing import TYPE_CHECKING, Any

from dylan.dag.dag_tuple import DAGTuple
from dylan.dag.groundable_edge import (
    ActionReplayEdge,
    BacktrackingEdge,
    CompletionEdge,
    GroundableEdge,
    VirtualRepairingEdge,
)
from dylan.dag.uttered_word import UtteredWord
from dylan.tree.tree import Tree

if TYPE_CHECKING:
    from dylan.action.action import Action

logger = logging.getLogger(__name__)

REPAIR_INIT_PREFIX = "init-repair"
_ALWAYS_GOOD_EDGE_NAMES: frozenset[str] = frozenset({"thinning", "elimination", "link-evaluation"})


def _edge_action_name(edge: GroundableEdge) -> str:
    """Return the primary action name on a groundable edge, if any."""
    if not edge.actions:
        return ""
    action = edge.actions[0]
    return action.get_name() if hasattr(action, "get_name") else str(action)


class WordLevelContextDAG:
    """DAG whose edges correspond to words (Eshghi et al. 2015 style)."""

    def __init__(self) -> None:
        self.context: Any = None
        self.word_stack: list[UtteredWord] = []
        self.exhausted = False
        self.repair_processing = False
        self.last_n_capacity: int = 5
        self.last_n: list[Tree] = []
        self.id_pool_nodes: list[int] = []
        self.id_pool_edges: list[int] = []
        self._out: dict[DAGTuple, list[GroundableEdge]] = {}
        self._in: dict[DAGTuple, tuple[GroundableEdge, DAGTuple] | None] = {}
        self._nodes: set[DAGTuple] = set()
        self.first_tuple_after_last_word: DAGTuple | None = None
        ax = Tree()
        self.root = self.get_new_tuple(ax)
        self.cur = self.root
        self.first_tuple_after_last_word = self.cur

    def set_context(self, ctx: Any) -> None:
        """Attach the owning parser context."""
        self.context = ctx

    def _register(self, n: DAGTuple) -> None:
        """Register a tuple in adjacency maps."""
        self._nodes.add(n)
        self._out.setdefault(n, [])
        if n not in self._in:
            self._in[n] = None

    def contains_vertex(self, n: DAGTuple) -> bool:
        """Return whether tuple *n* belongs to this DAG."""
        return n in self._nodes

    def out_degree(self, n: DAGTuple) -> int:
        """Return outgoing edge count from *n*."""
        return len(self._out.get(n, ()))

    def _compare_edges_by_endpoint_completeness(
        self,
        o1: GroundableEdge,
        o2: GroundableEdge,
    ) -> int:
        """Order edges like Java ``EdgeComparatorByEndPointCompleteness`` (lower incompleteness first)."""
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
        end1 = o1.dst
        end2 = o2.dst
        if not isinstance(end1, DAGTuple) or not isinstance(end2, DAGTuple):
            return (o1.edge_id - o2.edge_id) or 1
        tree1 = end1.get_tree()
        tree2 = end2.get_tree()
        t1 = tree1.get_incompleteness_measure()
        t2 = tree2.get_incompleteness_measure()
        if t1 > t2:
            return 1
        if t2 > t1:
            return -1
        n1, n2 = _edge_action_name(o1), _edge_action_name(o2)
        if n1.startswith("hyp-sem") and (n2 == "thinning" or n2.startswith("hyp-cn-decomp")):
            return -1
        if n2.startswith("hyp-sem") and (n1 == "thinning" or n1.startswith("hyp-cn-decomp")):
            return 1
        if "hyp-build-cn-e-1" in n1 and "hyp-build-cn-e-0" in n2:
            return -1
        if "hyp-build-cn-e-0" in n1 and "hyp-build-cn-e-1" in n2:
            return 1
        return 1

    def get_out_edges(self, n: DAGTuple | None = None) -> list[GroundableEdge]:
        """Return outgoing edges from *n* sorted by endpoint completeness (Java ``DAG.getOutEdges``)."""
        u = n if n is not None else self.cur
        raw = list(self._out.get(u, ()))
        return sorted(raw, key=cmp_to_key(self._compare_edges_by_endpoint_completeness))

    def get_out_edges_for_traversal(self, n: DAGTuple | None = None) -> list[GroundableEdge]:
        """Return out-edges for ``go_first``, resolving repair back-edges (Java ``getOutEdgesForTraversal``)."""
        u = n if n is not None else self.cur
        result: list[GroundableEdge] = []
        for edge in self.get_out_edges(u):
            if isinstance(edge, BacktrackingEdge) and edge.overarching_repairing_edge is not None:
                result.append(edge.overarching_repairing_edge)
            else:
                result.append(edge)
        return result

    def get_parent_edge(self, n: DAGTuple | None = None) -> GroundableEdge | None:
        """Return incoming edge for *n* or current tuple."""
        u = n if n is not None else self.cur
        inc = self._in.get(u)
        if inc is None:
            return None
        return inc[0]

    def get_prev_action(self) -> "Action | None":
        """Return the inbound action entering :attr:`cur` (Java ``DAG.getPrevAction``)."""
        pe = self.get_parent_edge(self.cur)
        if pe is None or not pe.actions:
            return None
        return pe.actions[0]

    def get_parent(self, n: DAGTuple | None = None) -> DAGTuple | None:
        """Return parent tuple for *n* or current tuple."""
        u = n if n is not None else self.cur
        inc = self._in.get(u)
        if inc is None:
            return None
        return inc[1]

    def get_current_tuple(self) -> DAGTuple:
        """Return the current tuple."""
        return self.cur

    def set_current_tuple(self, t: DAGTuple) -> None:
        """Set the current tuple."""
        self.cur = t

    def word_stack_ref(self) -> list[UtteredWord]:
        """Return mutable word stack."""
        return self.word_stack

    def set_exhausted(self, b: bool) -> None:
        """Set parser-state exhausted flag."""
        self.exhausted = b

    def is_exhausted(self) -> bool:
        """Return whether parser state is exhausted."""
        return self.exhausted

    def set_repair_processing(self, b: bool) -> None:
        """Enable or disable repair processing."""
        self.repair_processing = b

    def repair_processing_enabled(self) -> bool:
        """Whether local-repair machinery is active (Java `DAG.repairProcessingEnabled`)."""
        return self.repair_processing

    def loop_detected(self, res: Tree) -> bool:
        """Return whether *res* repeats a recent tree on the path (Java ``DAG.loopDetected``)."""
        return any(res == t for t in self.last_n)

    def update_last_n(self) -> None:
        """Rebuild :attr:`last_n` from the current tuple and its ancestors (Java ``updateLastN``)."""
        self.last_n.clear()
        current: DAGTuple | None = self.cur
        n = 0
        while current is not None and n < self.last_n_capacity:
            self.last_n.append(current.get_tree())
            current = self.get_parent(current)
            n += 1

    def reset_last_n(self) -> None:
        """Reset loop-detection history from root toward :attr:`cur` (Java ``resetLastN``)."""
        self.last_n.clear()
        current: DAGTuple | None = self.cur
        i = 0
        while current is not None and not self.is_root(current) and i < self.last_n_capacity:
            self.last_n.insert(0, current.get_tree())
            i += 1
            current = self.get_parent(current)

    def is_root(self, node: DAGTuple | None = None) -> bool:
        """Return whether *node* (or :attr:`cur`) is the DAG root (Java ``DAG.isRoot``)."""
        u = node if node is not None else self.cur
        return u is self.root

    def get_new_tuple(self, t: Tree) -> DAGTuple:
        """Create and register a tuple for tree *t*."""
        new_id = len(self.id_pool_nodes) + 1
        self.id_pool_nodes.append(new_id)
        dt = DAGTuple(t, new_id)
        self._register(dt)
        return dt

    def get_new_edge(self, actions: list[Any], word: UtteredWord | None) -> GroundableEdge:
        """Create a word edge."""
        eid = len(self.id_pool_edges) + 1
        self.id_pool_edges.append(eid)
        return GroundableEdge(actions, word, eid)

    def _next_edge_id(self) -> int:
        """Allocate a fresh edge id."""
        eid = len(self.id_pool_edges) + 1
        self.id_pool_edges.append(eid)
        return eid

    def get_new_completion_edge(self, actions: list[Any], word: UtteredWord | None = None) -> CompletionEdge:
        """Create a completion edge."""
        return CompletionEdge(actions, word, self._next_edge_id())

    def get_new_action_replay_edge(self, actions: list[Any], word: UtteredWord | None = None) -> ActionReplayEdge:
        """Create an action-replay edge."""
        return ActionReplayEdge(actions, word, self._next_edge_id())

    def get_new_virtual_repairing_edge(
        self,
        actions: list[Any],
        word: UtteredWord | None,
    ) -> VirtualRepairingEdge:
        """Create a virtual repairing edge."""
        edge = VirtualRepairingEdge(actions, word, self._next_edge_id())
        edge.set_repairable(False)
        return edge

    def get_new_backtracking_edge(
        self,
        actions: list[Any] | None = None,
        word: UtteredWord | None = None,
    ) -> BacktrackingEdge:
        """Create a backtracking edge."""
        return BacktrackingEdge(actions or [], word, self._next_edge_id())

    def add_edge(self, edge: GroundableEdge, src: DAGTuple, dst: DAGTuple) -> None:
        """Add directed edge from *src* to *dst*."""
        edge.src = src
        edge.dst = dst
        self._out[src].append(edge)
        self._in[dst] = (edge, src)

    def add_child(self, child: DAGTuple, edge: GroundableEdge) -> DAGTuple:
        """Add *child* below the current tuple using *edge*."""
        return self.add_child_from(self.cur, child, edge)

    def add_child_from(self, parent: DAGTuple, child: DAGTuple, edge: GroundableEdge) -> DAGTuple:
        """Add *child* below *parent* using *edge*."""
        if not self.contains_vertex(parent):
            raise ValueError("parent must exist")
        child.set_depth(parent.get_depth() + 1)
        self.add_edge(edge, parent, child)
        return child

    def remove_children(self, current: DAGTuple | None = None) -> None:
        """Remove all descendants of *current* or current tuple."""
        c = current if current is not None else self.cur
        for e in list(self._out.get(c, ())):
            dest = e.dst
            assert isinstance(dest, DAGTuple)
            self._remove_subtree(dest)
        self._out[c] = []

    def _remove_subtree(self, n: DAGTuple) -> None:
        """Remove a tuple and all descendants."""
        for e in list(self._out.get(n, ())):
            self._remove_subtree(e.dst)  # type: ignore[arg-type]
        self._nodes.discard(n)
        self._out.pop(n, None)
        self._in.pop(n, None)

    def initiate_local_repair(self) -> None:
        """Push the repair-init marker onto the word stack."""
        if not self.repair_processing:
            return
        top = self.word_stack[-1]
        self.word_stack.append(UtteredWord(REPAIR_INIT_PREFIX, top.speaker))

    def repair_initiated(self) -> bool:
        """Return whether the top stack entry is the repair-init marker."""
        return bool(self.word_stack) and self.word_stack[-1].word == REPAIR_INIT_PREFIX

    def more_unseen_edges(self) -> bool:
        """Return whether current tuple has unexplored outgoing edges."""
        return any(not e.has_been_seen() for e in self.get_out_edges())

    def _word_edge_matches_stack(self, edge: GroundableEdge) -> bool:
        """Return whether *edge* can be traversed without a word-stack mismatch."""
        if isinstance(edge, CompletionEdge) or edge.word is None:
            return True
        return bool(self.word_stack) and self.word_stack[-1] == edge.word

    def has_viable_unseen_edges(self) -> bool:
        """Return whether some unseen out-edge can be taken now (word matches stack top if labelled)."""
        return any(
            not e.has_been_seen() and self._word_edge_matches_stack(e)
            for e in self.get_out_edges()
        )

    def go_up_once(self) -> GroundableEdge | None:
        """Move one edge toward the root."""
        pe = self.get_parent_edge(self.cur)
        if pe is None:
            return None
        parent = self.get_parent(self.cur)
        assert parent is not None
        self.cur = parent
        self.update_last_n()
        return pe

    def attempt_backtrack(self) -> bool:
        """Backtrack to the nearest tuple with stack-compatible unseen outgoing edges."""
        while not self.has_viable_unseen_edges():
            if self.get_parent(self.cur) is None:
                logger.info("cannot backtrack from %s", self.cur)
                return False
            back = self.get_parent_edge(self.cur)
            assert back is not None
            if back.word is not None:
                self.word_stack.append(back.word)
            gone = self.go_up_once()
            assert gone is not None
            gone.set_seen(True)
            gone.set_in_context(False)
        logger.debug("Backtrack succeeded")
        return True

    def go_first(self) -> GroundableEdge | None:
        """Traverse the first unseen outgoing edge from the current tuple (Java ``goFirst``)."""
        if self.out_degree(self.cur) == 0:
            return None
        for e in self.get_out_edges_for_traversal(self.cur):
            if e.has_been_seen():
                continue
            logger.info("Going forward first along: %s", e)
            if isinstance(e, VirtualRepairingEdge):
                pass
            elif not isinstance(e, CompletionEdge) and e.word is not None and self.word_stack:
                if self.word_stack[-1] != e.word:
                    continue
                self.word_stack.pop()
            elif not isinstance(e, CompletionEdge) and e.word is not None and not self.word_stack:
                continue
            e.traverse(self)
            self.update_last_n()
            for oe in self.get_out_edges(self.cur):
                oe.set_seen(False)
            logger.info("Depth is now: %s", self.cur.get_depth())
            return e
        return None

    def init(self) -> None:
        """Reset this DAG to a fresh axiom tree."""
        logger.debug("initialising DAG")
        self.cur = self.root
        self.remove_children(self.cur)
        self.cur.set_tree(Tree())
        self.cur.set_maximal_semantics(None)
        self.word_stack.clear()
        self.exhausted = False
        self.id_pool_edges.clear()
        self.this_is_first_tuple_after_last_word()

    def this_is_first_tuple_after_last_word(self) -> None:
        """Mark the current tuple as the post-word anchor."""
        self.first_tuple_after_last_word = self.cur

    def reset_to_first_tuple_after_last_word(self) -> None:
        """Reset current tuple to the last post-word anchor and clear children."""
        anchor = self.first_tuple_after_last_word or self.root
        path: list[tuple[DAGTuple, GroundableEdge]] = []
        node = anchor
        while node != self.root:
            inc = self._in.get(node)
            if inc is None:
                break
            edge, parent = inc
            path.append((parent, edge))
            node = parent
        path.reverse()

        for parent, _edge in path:
            for out_edge in self.get_out_edges(parent):
                out_edge.set_seen(False)
                out_edge.set_in_context(False)

        self.set_current_tuple(path[0][0] if path else anchor)
        for _parent, edge in path:
            edge.traverse(self)
        self.word_stack.clear()
        self.remove_children(self.cur)
        self.exhausted = False

    def get_depth(self) -> int:
        """Return current tuple depth."""
        return self.cur.get_depth()

    def add_axiom(self) -> None:
        """Reset to a new sentence axiom."""
        self.init()

    def roll_back(self, n: int) -> bool:
        """Roll back up to *n* word edges along the active path."""
        moved = 0
        while moved < n and self.get_parent(self.cur) is not None:
            edge = self.get_parent_edge(self.cur)
            if edge is None:
                break
            if edge.word is not None:
                moved += 1
            edge.backtrack(self)
        self.this_is_first_tuple_after_last_word()
        return moved == n

    def ground_to_root(self) -> None:
        """Mark every edge on the active path as grounded/in context."""
        node = self.cur
        while True:
            inc = self._in.get(node)
            if inc is None:
                break
            edge, parent = inc
            edge.set_in_context(True)
            node = parent

    def get_all_tuples(self) -> list[DAGTuple]:
        """Return all tuples in creation order."""
        return sorted(self._nodes, key=lambda t: t.tuple_id)

    def get_n_best_final_tuples(self, n: int) -> list[DAGTuple]:
        """Return up to *n* complete leaf tuples in creation order, not parser step-through order."""
        leaves = [
            node for node in self.get_all_tuples()
            if self.out_degree(node) == 0 and node.is_complete()
        ]
        return leaves[:n]

    def at_root(self) -> bool:
        """Return true when :attr:`cur` is the root tuple (Java ``DAG.atRoot``)."""
        return self.cur is self.root

    def get_root(self) -> DAGTuple:
        """Return the root tuple (Java ``DAG.getRoot``)."""
        return self.root

    def mark_edge_as_seen(self, edge: GroundableEdge) -> None:
        """Set the seen flag on *edge* (Java ``DAG.markEdgeAsSeen``)."""
        edge.set_seen(True)

    @staticmethod
    def _same_groundable_edge(a: GroundableEdge, b: GroundableEdge) -> bool:
        """Return whether *a* and *b* are the same DAG edge (identity or shared id + parent)."""
        if a is b:
            return True
        return (
            getattr(a, "edge_id", None) == getattr(b, "edge_id", None)
            and getattr(a, "src", None) is getattr(b, "src", None)
        )

    def mark_edge_as_seen_and_below_it_unseen(self, seen_edge: GroundableEdge) -> None:
        """Mark *seen_edge* as seen; outgoing edges after it from the same parent become unseen (Java ``markEdgeAsSeenAndBelowItUnseen``)."""
        parent = getattr(seen_edge, "src", None)
        if parent is None:
            logger.debug("mark_edge_as_seen_and_below_it_unseen: missing src on %s", seen_edge)
            return
        done = False
        for out_edge in self.get_out_edges(parent):
            if done:
                out_edge.set_seen(False)
                continue
            if self._same_groundable_edge(out_edge, seen_edge):
                out_edge.set_seen(True)
                done = True

    def get_sequence_to_root(self, end: DAGTuple | None = None) -> list[GroundableEdge]:
        """Return edges along the path from *end* (or :attr:`cur`) up to the root (Java ``DAG.getSequenceToRoot``)."""
        node = end if end is not None else self.cur
        result: list[GroundableEdge] = []
        while node is not None and node is not self.root:
            inc = self._in.get(node)
            if inc is None:
                break
            edge, parent = inc
            result.append(edge)
            node = parent
        result.reverse()
        return result


WordLevelContextDAG.setContext = WordLevelContextDAG.set_context  # type: ignore[attr-defined]
WordLevelContextDAG.containsVertex = WordLevelContextDAG.contains_vertex  # type: ignore[attr-defined]
WordLevelContextDAG.outDegree = WordLevelContextDAG.out_degree  # type: ignore[attr-defined]
WordLevelContextDAG.getOutEdges = WordLevelContextDAG.get_out_edges  # type: ignore[attr-defined]
WordLevelContextDAG.getOutEdgesForTraversal = WordLevelContextDAG.get_out_edges_for_traversal  # type: ignore[attr-defined]
WordLevelContextDAG.getParentEdge = WordLevelContextDAG.get_parent_edge  # type: ignore[attr-defined]
WordLevelContextDAG.getParent = WordLevelContextDAG.get_parent  # type: ignore[attr-defined]
WordLevelContextDAG.getCurrentTuple = WordLevelContextDAG.get_current_tuple  # type: ignore[attr-defined]
WordLevelContextDAG.setCurrentTuple = WordLevelContextDAG.set_current_tuple  # type: ignore[attr-defined]
WordLevelContextDAG.wordStack = WordLevelContextDAG.word_stack_ref  # type: ignore[attr-defined]
WordLevelContextDAG.setExhausted = WordLevelContextDAG.set_exhausted  # type: ignore[attr-defined]
WordLevelContextDAG.isExhausted = WordLevelContextDAG.is_exhausted  # type: ignore[attr-defined]
WordLevelContextDAG.setRepairProcessing = WordLevelContextDAG.set_repair_processing  # type: ignore[attr-defined]
WordLevelContextDAG.loopDetected = WordLevelContextDAG.loop_detected  # type: ignore[attr-defined]
WordLevelContextDAG.resetLastN = WordLevelContextDAG.reset_last_n  # type: ignore[attr-defined]
WordLevelContextDAG.updateLastN = WordLevelContextDAG.update_last_n  # type: ignore[attr-defined]
WordLevelContextDAG.repairProcessingEnabled = WordLevelContextDAG.repair_processing_enabled  # type: ignore[attr-defined]
WordLevelContextDAG.getNewTuple = WordLevelContextDAG.get_new_tuple  # type: ignore[attr-defined]
WordLevelContextDAG.getNewEdge = WordLevelContextDAG.get_new_edge  # type: ignore[attr-defined]
WordLevelContextDAG.getNewCompletionEdge = WordLevelContextDAG.get_new_completion_edge  # type: ignore[attr-defined]
WordLevelContextDAG.getNewActionReplayEdge = WordLevelContextDAG.get_new_action_replay_edge  # type: ignore[attr-defined]
WordLevelContextDAG.getNewVirtualRepairingEdge = WordLevelContextDAG.get_new_virtual_repairing_edge  # type: ignore[attr-defined]
WordLevelContextDAG.getNewBacktrackingEdge = WordLevelContextDAG.get_new_backtracking_edge  # type: ignore[attr-defined]
WordLevelContextDAG.addChild = WordLevelContextDAG.add_child  # type: ignore[attr-defined]
WordLevelContextDAG.addChildFrom = WordLevelContextDAG.add_child_from  # type: ignore[attr-defined]
WordLevelContextDAG.removeChildren = WordLevelContextDAG.remove_children  # type: ignore[attr-defined]
WordLevelContextDAG.initiateLocalRepair = WordLevelContextDAG.initiate_local_repair  # type: ignore[attr-defined]
WordLevelContextDAG.repairInitiated = WordLevelContextDAG.repair_initiated  # type: ignore[attr-defined]
WordLevelContextDAG.moreUnseenEdges = WordLevelContextDAG.more_unseen_edges  # type: ignore[attr-defined]
WordLevelContextDAG.goUpOnce = WordLevelContextDAG.go_up_once  # type: ignore[attr-defined]
WordLevelContextDAG.attemptBacktrack = WordLevelContextDAG.attempt_backtrack  # type: ignore[attr-defined]
WordLevelContextDAG.goFirst = WordLevelContextDAG.go_first  # type: ignore[attr-defined]
WordLevelContextDAG.thisIsFirstTupleAfterLastWord = WordLevelContextDAG.this_is_first_tuple_after_last_word  # type: ignore[attr-defined]
WordLevelContextDAG.resetToFirstTupleAfterLastWord = WordLevelContextDAG.reset_to_first_tuple_after_last_word  # type: ignore[attr-defined]
WordLevelContextDAG.getDepth = WordLevelContextDAG.get_depth  # type: ignore[attr-defined]
WordLevelContextDAG.addAxiom = WordLevelContextDAG.add_axiom  # type: ignore[attr-defined]
WordLevelContextDAG.rollBack = WordLevelContextDAG.roll_back  # type: ignore[attr-defined]
WordLevelContextDAG.groundToRoot = WordLevelContextDAG.ground_to_root  # type: ignore[attr-defined]
WordLevelContextDAG.atRoot = WordLevelContextDAG.at_root  # type: ignore[attr-defined]
WordLevelContextDAG.getRoot = WordLevelContextDAG.get_root  # type: ignore[attr-defined]
WordLevelContextDAG.markEdgeAsSeen = WordLevelContextDAG.mark_edge_as_seen  # type: ignore[attr-defined]
WordLevelContextDAG.markEdgeAsSeenAndBelowItUnseen = WordLevelContextDAG.mark_edge_as_seen_and_below_it_unseen  # type: ignore[attr-defined]
WordLevelContextDAG.getPrevAction = WordLevelContextDAG.get_prev_action  # type: ignore[attr-defined]
WordLevelContextDAG.getSequenceToRoot = WordLevelContextDAG.get_sequence_to_root  # type: ignore[attr-defined]
