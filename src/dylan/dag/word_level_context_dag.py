"""Word-level context DAG (partial Java `WordLevelContextDAG`)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dylan.dag.dag_tuple import DAGTuple
from dylan.dag.groundable_edge import GroundableEdge
from dylan.dag.uttered_word import UtteredWord
from dylan.tree.tree import Tree

if TYPE_CHECKING:
    from dylan.action.action import Action

logger = logging.getLogger(__name__)

REPAIR_INIT_PREFIX = "init-repair"


class WordLevelContextDAG:
    """DAG whose edges correspond to words (Eshghi et al. 2015 style)."""

    def __init__(self) -> None:
        self.context: Any = None
        self.word_stack: list[UtteredWord] = []
        self.exhausted = False
        self.repair_processing = False
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
        self.context = ctx

    def _register(self, n: DAGTuple) -> None:
        self._nodes.add(n)
        self._out.setdefault(n, [])
        if n not in self._in:
            self._in[n] = None

    def contains_vertex(self, n: DAGTuple) -> bool:
        return n in self._nodes

    def out_degree(self, n: DAGTuple) -> int:
        return len(self._out.get(n, ()))

    def get_out_edges(self, n: DAGTuple | None = None) -> list[GroundableEdge]:
        u = n if n is not None else self.cur
        return sorted(self._out.get(u, ()), key=lambda e: e.edge_id)

    def get_parent_edge(self, n: DAGTuple | None = None) -> GroundableEdge | None:
        u = n if n is not None else self.cur
        inc = self._in.get(u)
        if inc is None:
            return None
        return inc[0]

    def get_parent(self, n: DAGTuple | None = None) -> DAGTuple | None:
        u = n if n is not None else self.cur
        inc = self._in.get(u)
        if inc is None:
            return None
        return inc[1]

    def get_current_tuple(self) -> DAGTuple:
        return self.cur

    def set_current_tuple(self, t: DAGTuple) -> None:
        self.cur = t

    def word_stack_ref(self) -> list[UtteredWord]:
        return self.word_stack

    def set_exhausted(self, b: bool) -> None:
        self.exhausted = b

    def is_exhausted(self) -> bool:
        return self.exhausted

    def set_repair_processing(self, b: bool) -> None:
        self.repair_processing = b

    def repair_processing_enabled(self) -> bool:
        """Whether local-repair machinery is active (Java `DAG.repairProcessingEnabled`)."""
        return self.repair_processing

    def get_new_tuple(self, t: Tree) -> DAGTuple:
        new_id = len(self.id_pool_nodes) + 1
        self.id_pool_nodes.append(new_id)
        dt = DAGTuple(t, new_id)
        self._register(dt)
        return dt

    def get_new_edge(self, actions: list[Any], word: UtteredWord | None) -> GroundableEdge:
        eid = len(self.id_pool_edges) + 1
        self.id_pool_edges.append(eid)
        return GroundableEdge(actions, word, eid)

    def add_edge(self, edge: GroundableEdge, src: DAGTuple, dst: DAGTuple) -> None:
        edge.src = src
        edge.dst = dst
        self._out[src].append(edge)
        self._in[dst] = (edge, src)

    def add_child(self, child: DAGTuple, edge: GroundableEdge) -> DAGTuple:
        return self.add_child_from(self.cur, child, edge)

    def add_child_from(self, parent: DAGTuple, child: DAGTuple, edge: GroundableEdge) -> DAGTuple:
        if not self.contains_vertex(parent):
            raise ValueError("parent must exist")
        child.set_depth(parent.get_depth() + 1)
        self.add_edge(edge, parent, child)
        return child

    def remove_children(self, current: DAGTuple | None = None) -> None:
        c = current if current is not None else self.cur
        for e in list(self._out.get(c, ())):
            dest = e.dst
            assert isinstance(dest, DAGTuple)
            self._remove_subtree(dest)
        self._out[c] = []

    def _remove_subtree(self, n: DAGTuple) -> None:
        for e in list(self._out.get(n, ())):
            self._remove_subtree(e.dst)  # type: ignore[arg-type]
        self._nodes.discard(n)
        self._out.pop(n, None)
        self._in.pop(n, None)

    def initiate_local_repair(self) -> None:
        if not self.repair_processing:
            return
        top = self.word_stack[-1]
        self.word_stack.append(UtteredWord(REPAIR_INIT_PREFIX, top.speaker))

    def repair_initiated(self) -> bool:
        return bool(self.word_stack) and self.word_stack[-1].word == REPAIR_INIT_PREFIX

    def more_unseen_edges(self) -> bool:
        return any(not e.has_been_seen() for e in self.get_out_edges())

    def go_up_once(self) -> GroundableEdge | None:
        pe = self.get_parent_edge(self.cur)
        if pe is None:
            return None
        parent = self.get_parent(self.cur)
        assert parent is not None
        self.cur = parent
        return pe

    def attempt_backtrack(self) -> bool:
        while not self.more_unseen_edges():
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
        if self.out_degree(self.cur) == 0:
            return None
        for e in self.get_out_edges(self.cur):
            if e.has_been_seen():
                continue
            logger.info("Going forward first along: %s", e)
            if e.word is not None and self.word_stack:
                if self.word_stack[-1] != e.word:
                    msg = f"stack {self.word_stack[-1]!r} vs edge {e.word!r}"
                    logger.error(msg)
                    raise RuntimeError(msg)
                self.word_stack.pop()
            elif e.word is not None and not self.word_stack:
                raise RuntimeError("empty stack for word edge")
            e.traverse(self)
            for oe in self.get_out_edges(self.cur):
                oe.set_seen(False)
            logger.info("Depth is now: %s", self.cur.get_depth())
            return e
        return None

    def init(self) -> None:
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
        self.first_tuple_after_last_word = self.cur

    def reset_to_first_tuple_after_last_word(self) -> None:
        anchor = self.first_tuple_after_last_word or self.root
        self.set_current_tuple(anchor)
        self.word_stack.clear()
        self.remove_children(self.cur)
        self.exhausted = False

    def get_depth(self) -> int:
        return self.cur.get_depth()
