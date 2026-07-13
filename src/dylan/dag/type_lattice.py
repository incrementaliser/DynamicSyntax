"""TTR record-type lattice used during induction (Java ``TypeLattice``).

A directed tree of :class:`TypeTuple` vertices linked by
:class:`TypeLatticeIncrement` edges.  Provides Java-compatible
``goFirst`` / ``attemptBacktrack`` / ``getIncrements`` / ``mergeLatticeAt``
traversal so :class:`dylan.induction.em_learner.ttr_hypothesiser.TTRHypothesiser`
can enumerate sub-typing increments in priority order.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dylan.dag.type_lattice_increment import TypeLatticeIncrement
from dylan.dag.type_tuple import TypeTuple

if TYPE_CHECKING:
    from dylan.formula.ttr_field import TTRField
    from dylan.formula.ttr_label import TTRLabel
    from dylan.formula.ttr_record_type import TTRRecordType
    from dylan.type.dstype import DSType

logger = logging.getLogger(__name__)


class TypeLattice:
    """Tree-shaped record-type lattice (replaces JUNG ``DelegateTree``)."""

    priority_templates: list["TTRRecordType"] = []

    def __init__(
        self,
        from_or_to: "TTRRecordType | None" = None,
        to: "TTRRecordType | None" = None,
    ) -> None:
        """Java ``TypeLattice()`` / ``TypeLattice(to)`` / ``TypeLattice(from, to)`` constructors merged."""
        from dylan.formula.ttr_record_type import TTRRecordType
        from dylan.type.dstype import DSType

        self._id_pool_nodes: list[int] = []
        self._id_pool_edges: list[int] = []
        self.depth: int = 0
        self.entity_types: list = [DSType.e, DSType.es]
        self.seen_types: set = set()
        self.priority_fields: list = []
        self.last_inc: list[TypeLatticeIncrement] | None = None
        self.cur_increment_on: "TTRLabel | None" = None
        # Tree storage: parent[v] -> incoming edge; children[v] -> [(edge, dest), ...]
        self._parent_edge: dict[TypeTuple, TypeLatticeIncrement] = {}
        self._parent: dict[TypeTuple, TypeTuple] = {}
        self._children: dict[TypeTuple, list[tuple[TypeLatticeIncrement, TypeTuple]]] = {}

        # root with empty type
        self.root: TypeTuple = TypeTuple.get_new_tuple(self._id_pool_nodes)
        self._add_vertex(self.root)
        self.cur: TypeTuple = self.root
        self.init_templates()

        self.rec_type: TTRRecordType
        if from_or_to is not None and to is None:
            self.rec_type = from_or_to
            self.priority_fields = self.get_priority_fields()
            self.initialise(from_or_to)
        elif from_or_to is not None and to is not None:
            self.rec_type = to
            self.priority_fields = self.get_priority_fields()
            self.cur.set_type(from_or_to)
            self.cur.increment_so_far = TTRRecordType()
            self._populate(self.cur)
        else:
            self.rec_type = TTRRecordType()

    # ---------------- graph operations (replacing JUNG) ----------------

    def _add_vertex(self, v: TypeTuple) -> None:
        """Register a vertex (Java ``addVertex``)."""
        self._children.setdefault(v, [])

    def add_edge(self, edge: TypeLatticeIncrement, src: TypeTuple, dest: TypeTuple) -> None:
        """Attach *dest* below *src* via *edge* (Java ``addEdge``)."""
        self._add_vertex(dest)
        self._children.setdefault(src, []).append((edge, dest))
        self._parent[dest] = src
        self._parent_edge[dest] = edge

    def get_root(self) -> TypeTuple:
        """Return the lattice root tuple (Java ``getRoot``)."""
        return self.root

    def get_dest(self, edge: TypeLatticeIncrement) -> TypeTuple:
        """Return the child tuple connected by *edge* (Java ``getDest``)."""
        for parent_v, kids in self._children.items():
            for e, d in kids:
                if e is edge or e == edge:
                    return d
        raise KeyError(edge)

    def get_parent(self, v: "TypeTuple | None" = None) -> TypeTuple | None:
        """Return the parent of *v* (or :attr:`cur`); ``None`` for the root (Java ``getParent``)."""
        node = v if v is not None else self.cur
        return self._parent.get(node)

    def get_parent_edge(self, v: "TypeTuple | None" = None) -> TypeLatticeIncrement | None:
        """Return the incoming edge of *v* (or :attr:`cur`) (Java ``getParentEdge``)."""
        node = v if v is not None else self.cur
        return self._parent_edge.get(node)

    def get_children(self, v: "TypeTuple | None" = None) -> list[TypeTuple]:
        """Return outgoing children of *v* (or :attr:`cur`) (Java ``getChildren``)."""
        node = v if v is not None else self.cur
        return [d for _, d in self._children.get(node, [])]

    def get_out_edges(self, v: "TypeTuple | None" = None) -> list[TypeLatticeIncrement]:
        """Return outgoing edges of *v* (or :attr:`cur`) (Java ``getOutEdges``)."""
        node = v if v is not None else self.cur
        return [e for e, _ in self._children.get(node, [])]

    def get_child_count(self, v: "TypeTuple | None" = None) -> int:
        """Return the number of children of *v* (or :attr:`cur`) (Java ``getChildCount``)."""
        node = v if v is not None else self.cur
        return len(self._children.get(node, []))

    def is_root(self, v: TypeTuple) -> bool:
        """Return true when *v* is the root tuple (Java ``isRoot``)."""
        return v is self.root or v == self.root

    def remove_child(self, child: TypeTuple) -> None:
        """Detach *child* (and its descendants) from the parent (Java ``removeChild``)."""
        parent = self._parent.pop(child, None)
        edge = self._parent_edge.pop(child, None)
        if parent is not None:
            self._children[parent] = [
                (e, d) for e, d in self._children.get(parent, []) if not (d is child or e is edge)
            ]
        for _, dest in list(self._children.get(child, [])):
            self.remove_child(dest)
        self._children.pop(child, None)

    def remove_children(self) -> None:
        """Drop all children of :attr:`cur` (Java ``removeChildren``)."""
        for child in list(self.get_children(self.cur)):
            self.remove_child(child)

    # ---------------- template population ----------------

    def init_templates(self) -> None:
        """Populate the static priority templates (Java ``initTemplates``)."""
        from dylan.formula.ttr_record_type import TTRRecordType
        from dylan.induction.corpus_profile import get_active_profile

        if TypeLattice.priority_templates:
            self.init_priority_fields()
            return
        for spec in get_active_profile().priority_template_specs:
            tpl = TTRRecordType.parse(spec)
            if tpl is not None:
                TypeLattice.priority_templates.append(tpl)
        self.init_priority_fields()

    def init_priority_fields(self) -> None:
        """Initialise priority fields used to bias subtype population (Java ``initPriorityFields``)."""
        from dylan.formula.ttr_field import TTRField
        from dylan.induction.corpus_profile import get_active_profile

        if self.priority_fields:
            return
        for spec in get_active_profile().priority_field_specs:
            tf = TTRField.parse(spec)
            if tf is not None:
                self.priority_fields.append(tf)

    def get_priority_fields(self) -> list:
        """Filter ``rec_type`` fields against the priority templates (Java ``getPriorityFields``)."""
        result: list = []
        if self.rec_type is None:
            return result
        for f in self.rec_type.get_fields():
            for template in self.priority_fields or []:
                if template.subsumes(f):
                    result.append(f)
        return result

    def get_entity_fields(self, t: "TTRRecordType") -> list:
        """Return entity-typed fields (e/es or untyped restrictors) (Java ``getEntityFields``)."""
        out: list = []
        for f in t.get_fields():
            ds = f.get_ds_type()
            if (ds is not None and ds in self.entity_types) or ds is None:
                out.append(f)
        return out

    # ---------------- traversal ----------------

    def init(self) -> None:
        """Reset traversal: clear ``seen`` flags from the root downwards (Java ``init``)."""
        self.seen_types = set()
        self.cur = self.get_root()
        self._init(self.cur)

    def _init(self, tt: TypeTuple) -> None:
        """Recursive ``init`` helper."""
        for edge in self.get_out_edges(tt):
            edge.set_seen(False)
            child = self.get_dest(edge)
            self._init(child)

    def attempt_backtrack(self, label: "TTRLabel | None" = None) -> bool:
        """Walk back to a tuple with unseen children, marking edges as we go (Java ``attemptBacktrack``)."""
        while not self.more_unseen_edges(label):
            if self.is_root(self.cur):
                return False
            back_over = self.go_up_once()
            if back_over is not None:
                back_over.set_seen(True)
        return True

    def next_increment(self, label: "TTRLabel | None" = None) -> "TTRRecordType | None":
        """Return the next *increment_so_far* by going forward then back-tracking (Java ``nextIncrement``)."""
        edge: TypeLatticeIncrement | None = None
        while True:
            edge = self.go_first(label)
            if edge is not None:
                break
            if not self.attempt_backtrack(label):
                break
        if edge is not None:
            return self.cur.get_increment_so_far()
        return None

    def initialise(self, rt: "TTRRecordType") -> None:
        """Build the lattice for record type *rt* with priority-template seeding (Java ``initialise``)."""
        from dylan.formula.ttr_record_type import TTRRecordType

        head_field = rt.get_head_field()
        if head_field is None:
            return
        head_label = head_field.get_label()
        first_inc = rt.get_minimal_increment_with(rt.get_field(head_label), head_label)
        sub_lattice = TypeLattice(first_inc, rt)
        for template in TypeLattice.priority_templates:
            sub_lattice.init()
            inc = sub_lattice.next_increment(head_label)
            cur_subtype = sub_lattice.get_cur_subtype()
            while inc is not None:
                if template.subsumes(cur_subtype):
                    below_root = TypeTuple.get_new_tuple(cur_subtype, self._id_pool_nodes)
                    edge = TypeLatticeIncrement.get_new_edge(cur_subtype, head_label, self._id_pool_edges)
                    below_root.increment_so_far = cur_subtype
                    self.get_root().increment_so_far = TTRRecordType()
                    self.add_edge(edge, self.get_root(), below_root)
                    self.merge_lattice_at(below_root, sub_lattice.cur, sub_lattice)
                    return
                inc = sub_lattice.next_increment(head_label)
                cur_subtype = sub_lattice.get_cur_subtype()
        # Fallback when no template subsumes anything.
        below_root = TypeTuple.get_new_tuple(sub_lattice.get_root(), self._id_pool_nodes)
        first_inc.deem_head(head_label)
        edge = TypeLatticeIncrement.get_new_edge(first_inc, head_label, self._id_pool_edges)
        below_root.increment_so_far = first_inc
        self.get_root().increment_so_far = TTRRecordType()
        self.add_edge(edge, self.get_root(), below_root)
        self.merge_lattice_at(below_root, sub_lattice.cur, sub_lattice)

    def get_cur_subtype(self) -> "TTRRecordType":
        """Return the type of the active tuple (Java ``getCurSubtype``)."""
        return self.cur.get_type()

    def _populate(self, cur_tuple: TypeTuple) -> None:
        """Java ``populate`` — recursively expand sub-types out of ``rec_type``."""
        from dylan.formula.ttr_record_type import TTRRecordType
        from dylan.type.dstype import DSType

        if cur_tuple.get_type().equals_ignore_heads(self.rec_type):
            return
        if cur_tuple.get_type().is_empty():
            for f in self.rec_type.get_fields():
                if f.get_ds_type() is not None and self.rec_type.has_dependent(f):
                    increment = self.rec_type.get_super_type_with_parents(f)
                    increment.deem_head(f.get_label())
                    subtype = increment.asymmetric_merge(cur_tuple.get_type())
                    if not isinstance(subtype, TTRRecordType):
                        continue
                    self._populate(self._add_child(cur_tuple, subtype, increment, f.get_label()))
            return

        for entity_field in self.get_entity_fields(cur_tuple.get_type()):
            if not entity_field.is_head() and (
                entity_field.get_ds_type() is None or entity_field.get_ds_type() == DSType.cn
            ):
                target_restr = self.rec_type.get(entity_field.get_label())
                cur_restr = cur_tuple.get_type().get(entity_field.get_label())
                if not isinstance(target_restr, TTRRecordType) or not isinstance(
                    cur_restr, TTRRecordType,
                ):
                    continue
                lower = TypeLattice(cur_restr, target_restr)
                self.merge_lattice_at_label(cur_tuple, lower, entity_field.get_label())
                continue

            handled = False
            for prior in self.priority_fields:
                if not cur_tuple.get_type().has_field(prior) and prior.depends_on(entity_field):
                    increment = self.rec_type.get_minimal_increment_with(prior, entity_field.get_label())
                    increment.deem_head(entity_field.get_label())
                    subtype = increment.asymmetric_merge(cur_tuple.get_type())
                    if not isinstance(subtype, TTRRecordType):
                        continue
                    self._populate(
                        self._add_child(cur_tuple, subtype, increment, entity_field.get_label()),
                    )
                    handled = True
                    break
            if handled:
                continue

            from dylan.formula.ttr_label import HEAD as _HEAD

            for f in self.rec_type.get_fields():
                if f.get_label() == _HEAD:
                    continue
                if not cur_tuple.get_type().has_field(f) and f.depends_on(entity_field):
                    increment = self.rec_type.get_minimal_increment_with(f, entity_field.get_label())
                    increment.deem_head(entity_field.get_label())
                    subtype = cur_tuple.get_type().asymmetric_merge(increment)
                    if not isinstance(subtype, TTRRecordType):
                        continue
                    self._populate(
                        self._add_child(cur_tuple, subtype, increment, entity_field.get_label()),
                    )

    def _add_child(
        self,
        cur_tuple: TypeTuple,
        rec: "TTRRecordType",
        increment: "TTRRecordType",
        label: "TTRLabel",
    ) -> TypeTuple:
        """Java ``addChild`` — create a child tuple via *increment* and connect with a new edge."""
        from dylan.formula.ttr_record_type import TTRRecordType

        edge = TypeLatticeIncrement.get_new_edge(increment, label, self._id_pool_edges)
        target = TypeTuple.get_new_tuple(rec, self._id_pool_nodes)
        merged = cur_tuple.increment_so_far.asymmetric_merge(increment)
        target.increment_so_far = merged if isinstance(merged, TTRRecordType) else TTRRecordType()
        self.add_edge(edge, cur_tuple, target)
        return target

    def go_first(self, label: "TTRLabel | None" = None) -> TypeLatticeIncrement | None:
        """Step forward via the first unseen edge (optionally on *label*) (Java ``goFirst``)."""
        if self.get_child_count(self.cur) == 0:
            return None
        for e in self.get_out_edges(self.cur):
            if e.has_been_seen():
                continue
            if label is not None and e.increment_on != label:
                continue
            child = self.get_dest(e)
            if child.get_type() in self.seen_types:
                e.set_seen(True)
                continue
            self.cur = child
            self.seen_types.add(child.get_type())
            self.depth += 1
            return e
        return None

    def go_up_once(self) -> TypeLatticeIncrement | None:
        """Step back to the parent and return the edge we came down (Java ``goUpOnce``)."""
        if self.is_root(self.cur):
            return None
        result = self.get_parent_edge()
        parent = self.get_parent()
        if parent is not None:
            self.cur = parent
        self.depth -= 1
        return result

    def get_depth(self) -> int:
        """Depth of :attr:`cur` from the root (Java ``getDepth``)."""
        d = 0
        v: TypeTuple | None = self.cur
        while v is not None and not self.is_root(v):
            v = self.get_parent(v)
            d += 1
        return d

    def get_current_tuple(self) -> TypeTuple:
        """Return :attr:`cur` (Java ``getCurrentTuple``)."""
        return self.cur

    def set_current_tuple(self, tt: TypeTuple) -> None:
        """Replace :attr:`cur` (Java ``setCurrentTuple``)."""
        self.cur = tt

    def more_unseen_edges(self, label: "TTRLabel | None" = None) -> bool:
        """Return true if any outgoing edge of :attr:`cur` is unseen (optionally on *label*) (Java ``moreUnseenEdges``)."""
        for edge in self.get_out_edges():
            if edge.has_been_seen():
                continue
            if label is None or edge.increment_on == label:
                return True
        return False

    def get_increment_so_far(self) -> "TTRRecordType":
        """Return :attr:`cur`'s ``increment_so_far`` (Java ``getIncrementSoFar``)."""
        return self.cur.increment_so_far

    def get_increments(self, label: "TTRLabel") -> set:
        """Java ``getIncrements(TTRLabel)`` — collect increment paths starting at :attr:`cur` constrained to *label*."""
        if self.is_root(self.cur):
            return self._get_increments_from(self.cur, label)
        current = self.cur
        self.seen_types = set()
        negative_incs: list[TypeLatticeIncrement] = []
        while not self.is_root(current):
            parent_edge = self.get_parent_edge(current)
            if (
                current.get_type().has_label(label)
                and parent_edge is not None
                and not parent_edge.get_increment().is_empty()
            ):
                tail = self._get_increments_from(current, label)
                return self._add_at_begin(negative_incs, tail)
            if parent_edge is not None:
                neg = TypeLatticeIncrement(parent_edge)
                neg.positive = False
                negative_incs.append(neg)
            parent = self.get_parent(current)
            if parent is None:
                break
            current = parent
        return set()

    def _get_increments_from(self, cur: TypeTuple, label: "TTRLabel") -> set:
        """Recursive helper for :meth:`get_increments`."""
        if not self.get_children(cur):
            return set()
        result: set = set()
        for edge1 in self.get_out_edges(cur):
            edge = TypeLatticeIncrement(edge1)
            if edge.increment_on != label:
                continue
            dest = self.get_dest(edge1)
            if dest.get_type() in self.seen_types:
                continue
            self.seen_types.add(dest.get_type())
            child_inc = (
                self._get_head_increments(dest)
                if edge.get_increment().is_empty()
                else self._get_increments_from(dest, label)
            )
            single = (edge,)
            if not edge.get_increment().is_empty():
                result.add(single)
            for child_list in child_inc:
                new_list = (edge, *tuple(child_list))
                result.add(new_list)
        return result

    def _get_head_increments(self, current: TypeTuple) -> set:
        """Convenience wrapper: increments rooted at *current*'s head label."""
        head = current.get_type().get_head_field()
        if head is None:
            return set()
        return self._get_increments_from(current, head.get_label())

    def get_head_increments(self) -> set:
        """Java ``getHeadIncrements`` on :attr:`cur`."""
        head = self.cur.get_type().get_head_field()
        if head is None:
            return set()
        return self.get_increments(head.get_label())

    @staticmethod
    def _add_at_begin(prefix: list, suffix_set: set) -> set:
        """Prepend *prefix* to every list in *suffix_set* (Java ``addAtBegin``)."""
        result: set = set()
        for item in suffix_set:
            result.add((*tuple(prefix), *tuple(item)))
        return result

    def go(self, ops: "TypeLatticeIncrement | list[TypeLatticeIncrement]") -> bool:
        """Java ``go(List<TypeLatticeIncrement>)`` and ``go(TypeLatticeIncrement)`` overloads."""
        if isinstance(ops, TypeLatticeIncrement):
            self._go_single(ops)
            return True
        for inc in ops:
            self._go_single(inc)
        return True

    def _go_single(self, inc: TypeLatticeIncrement) -> None:
        """Move :attr:`cur` along *inc* (positive forward, negative backward)."""
        if inc.positive and inc in self.get_out_edges():
            self.cur = self.get_dest(inc)
            return
        parent_edge = self.get_parent_edge()
        if not inc.positive and parent_edge == inc:
            parent = self.get_parent()
            if parent is not None:
                self.cur = parent
            return
        raise RuntimeError(f"cannot traverse {inc} at {self.cur}")

    def backtrack(self, increments: "TypeLatticeIncrement | list[TypeLatticeIncrement]") -> None:
        """Reverse a path produced by :meth:`get_increments` (Java ``backtrack``)."""
        if isinstance(increments, TypeLatticeIncrement):
            inc = increments
            if not inc.positive and inc in self.get_out_edges():
                self.cur = self.get_dest(inc)
                return
            parent_edge = self.get_parent_edge()
            if inc.positive and parent_edge == inc:
                parent = self.get_parent()
                if parent is not None:
                    self.cur = parent
                return
            raise RuntimeError(f"cannot backtrack {inc} at {self.cur}")
        for inc in reversed(list(increments)):
            self.backtrack(inc)

    def merge_lattice_at_label(
        self,
        node: TypeTuple,
        lattice: "TypeLattice",
        label: "TTRLabel",
    ) -> None:
        """Java ``mergeLatticeAt(TypeTuple, TypeLattice, TTRLabel)``."""
        from dylan.formula.ttr_record_type import TTRRecordType

        transition = TypeLatticeIncrement.get_new_edge(TTRRecordType(), label, self._id_pool_edges)
        root = TypeTuple.get_new_tuple(lattice.root, self._id_pool_nodes)
        self.add_edge(transition, node, root)
        self.merge_lattice_at(root, lattice.get_root(), lattice)

    def merge_lattice_at(self, this_root: TypeTuple, *args: object) -> None:
        """Recursively merge *lattice* under *this_root* (Java ``mergeLatticeAt`` overloads)."""
        from dylan.formula.ttr_record_type import TTRRecordType

        if len(args) == 1 and isinstance(args[0], TypeLattice):
            lattice = args[0]
            other_root = lattice.get_root()
        elif len(args) == 2 and isinstance(args[0], TypeTuple) and isinstance(args[1], TypeLattice):
            other_root, lattice = args  # type: ignore[assignment]
        else:
            raise TypeError(f"merge_lattice_at: bad args {args!r}")
        for edge in lattice.get_out_edges(other_root):
            edge_copy = TypeLatticeIncrement.get_new_edge(edge, self._id_pool_edges)
            child_copy = TypeTuple.get_new_tuple(lattice.get_dest(edge), self._id_pool_nodes)
            inc_copy_inc = edge_copy.increment
            merged = this_root.increment_so_far.asymmetric_merge(inc_copy_inc) if inc_copy_inc is not None else this_root.increment_so_far
            child_copy.increment_so_far = merged if isinstance(merged, TTRRecordType) else TTRRecordType()
            self.add_edge(edge_copy, this_root, child_copy)
            self.merge_lattice_at(child_copy, lattice.get_dest(edge), lattice)

    def get_first_increment(self) -> TypeLatticeIncrement:
        """Return the single root-out edge (Java ``getFirstIncrement``)."""
        self.cur = self.get_root()
        edges = self.get_out_edges()
        if len(self.get_children()) != 1:
            raise RuntimeError("more than one child to root")
        return edges[0]

    def get_first_increments(self) -> set:
        """Move to the unique root child, then return :meth:`get_head_increments`."""
        self.cur = self.get_root()
        edges = self.get_out_edges()
        if len(self.get_children()) != 1:
            raise RuntimeError("more than one child to root")
        self.cur = self.get_dest(edges[0])
        return self.get_head_increments()

    def back_track_last(self) -> None:
        """Reverse the last recorded increment list (Java ``backTrackLast``)."""
        if self.last_inc is not None:
            self.backtrack(self.last_inc)


# Java-style aliases
TypeLattice.initTemplates = TypeLattice.init_templates  # type: ignore[attr-defined]
TypeLattice.initPriorityFields = TypeLattice.init_priority_fields  # type: ignore[attr-defined]
TypeLattice.attemptBacktrack = TypeLattice.attempt_backtrack  # type: ignore[attr-defined]
TypeLattice.nextIncrement = TypeLattice.next_increment  # type: ignore[attr-defined]
TypeLattice.getCurSubtype = TypeLattice.get_cur_subtype  # type: ignore[attr-defined]
TypeLattice.getEntityFields = TypeLattice.get_entity_fields  # type: ignore[attr-defined]
TypeLattice.getPriorityFields = TypeLattice.get_priority_fields  # type: ignore[attr-defined]
TypeLattice.getCurrentTuple = TypeLattice.get_current_tuple  # type: ignore[attr-defined]
TypeLattice.setCurrentTuple = TypeLattice.set_current_tuple  # type: ignore[attr-defined]
TypeLattice.getDepth = TypeLattice.get_depth  # type: ignore[attr-defined]
TypeLattice.getOutEdges = TypeLattice.get_out_edges  # type: ignore[attr-defined]
TypeLattice.getChildren = TypeLattice.get_children  # type: ignore[attr-defined]
TypeLattice.getChildCount = TypeLattice.get_child_count  # type: ignore[attr-defined]
TypeLattice.getRoot = TypeLattice.get_root  # type: ignore[attr-defined]
TypeLattice.getDest = TypeLattice.get_dest  # type: ignore[attr-defined]
TypeLattice.getParent = TypeLattice.get_parent  # type: ignore[attr-defined]
TypeLattice.getParentEdge = TypeLattice.get_parent_edge  # type: ignore[attr-defined]
TypeLattice.isRoot = TypeLattice.is_root  # type: ignore[attr-defined]
TypeLattice.removeChild = TypeLattice.remove_child  # type: ignore[attr-defined]
TypeLattice.removeChildren = TypeLattice.remove_children  # type: ignore[attr-defined]
TypeLattice.addEdge = TypeLattice.add_edge  # type: ignore[attr-defined]
TypeLattice.moreUnseenEdges = TypeLattice.more_unseen_edges  # type: ignore[attr-defined]
TypeLattice.goFirst = TypeLattice.go_first  # type: ignore[attr-defined]
TypeLattice.goUpOnce = TypeLattice.go_up_once  # type: ignore[attr-defined]
TypeLattice.getIncrements = TypeLattice.get_increments  # type: ignore[attr-defined]
TypeLattice.getHeadIncrements = TypeLattice.get_head_increments  # type: ignore[attr-defined]
TypeLattice.getIncrementSoFar = TypeLattice.get_increment_so_far  # type: ignore[attr-defined]
TypeLattice.mergeLatticeAt = TypeLattice.merge_lattice_at  # type: ignore[attr-defined]
TypeLattice.getFirstIncrement = TypeLattice.get_first_increment  # type: ignore[attr-defined]
TypeLattice.getFirstIncrements = TypeLattice.get_first_increments  # type: ignore[attr-defined]
TypeLattice.backTrackLast = TypeLattice.back_track_last  # type: ignore[attr-defined]
