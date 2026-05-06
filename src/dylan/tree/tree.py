"""DS tree (partial Java ``Tree``)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dylan.formula.variable import Variable

if TYPE_CHECKING:
    from dylan.formula.ttr_formula import TTRFormula
    from dylan.formula.ttr_record_type import TTRRecordType
from dylan.tree.basic_operator import BasicOperator
from dylan.tree.label.labels import FeatureLabel, FormulaLabel, Label, Requirement, TypeLabel
from dylan.tree.modality import Modality
from dylan.tree.node import Node
from dylan.tree.node_address import PATH_LOCAL_UNFIXED, PATH_UNFIXED, NodeAddress
from dylan.tree.underspecified_type_map import get_static_type_map
from dylan.type.dstype import DSType

logger = logging.getLogger(__name__)

ENTITY_VARIABLE_ROOT = "x"
EVENT_VARIABLE_ROOT = "e"
PROPOSITION_VARIABLE_ROOT = "p"
RECORD_TYPE_VARIABLE_ROOT = "r"
PREDICATE_VARIABLE_ROOT = "pred"

QUESTION_REC: "TTRRecordType | None" = None


def _question_record() -> "TTRRecordType":
    from dylan.formula.ttr_record_type import TTRRecordType

    global QUESTION_REC
    if QUESTION_REC is None:
        q = TTRRecordType.parse("[p==question(head):t]")
        assert q is not None
        QUESTION_REC = q
    return QUESTION_REC


class Tree(dict[NodeAddress, Node]):
    """Maps node addresses to nodes (Java extends TreeMap)."""

    def __init__(self, other: Tree | None = None) -> None:
        super().__init__()
        if other is None:
            self.root_addr = NodeAddress()
            self.pointer = self.root_addr
            self._entity_pool: list[Variable] = []
            self._event_pool: list[Variable] = []
            self._proposition_pool: list[Variable] = []
            self._record_type_pool: list[Variable] = []
            self._predicate_pool: list[Variable] = []
            root = Node(self.root_addr)
            root.add_label(Requirement(TypeLabel.t))
            self[self.root_addr] = root
        else:
            self.root_addr = other.root_addr
            self.pointer = other.pointer
            self._entity_pool = list(other._entity_pool)
            self._event_pool = list(other._event_pool)
            self._proposition_pool = list(other._proposition_pool)
            self._record_type_pool = list(other._record_type_pool)
            self._predicate_pool = list(other._predicate_pool)
            for k, v in other.items():
                self[k] = Node(v.address, list(v.labels))

    @property
    def pointed_node(self) -> Node:
        return self[self.pointer]

    def clone(self) -> Tree:
        """Deep copy nodes, pointer, and variable pools (Java ``Tree.clone`` sketch)."""
        return Tree(self)

    def set_pointer(self, addr: NodeAddress) -> None:
        """Set the pointer address (Java ``setPointer``)."""
        self.pointer = addr

    def get_fresh_entity_variable(self) -> Variable:
        """Allocate ``x0``, ``x1``, … (0-based; Java used 1-based)."""
        v = Variable(ENTITY_VARIABLE_ROOT + str(len(self._entity_pool)))
        self._entity_pool.append(v)
        return v

    def get_fresh_event_variable(self) -> Variable:
        """Allocate ``e0``, ``e1``, …."""
        v = Variable(EVENT_VARIABLE_ROOT + str(len(self._event_pool)))
        self._event_pool.append(v)
        return v

    def get_fresh_proposition_variable(self) -> Variable:
        """Allocate ``p0``, ``p1``, …."""
        v = Variable(PROPOSITION_VARIABLE_ROOT + str(len(self._proposition_pool)))
        self._proposition_pool.append(v)
        return v

    def get_fresh_record_type_variable(self) -> Variable:
        """Allocate record-type metavariables ``r0``, ``r1``, … (Java ``getFreshRecTypeVariable``)."""
        v = Variable(RECORD_TYPE_VARIABLE_ROOT + str(len(self._record_type_pool)))
        self._record_type_pool.append(v)
        return v

    def get_fresh_predicate_variable(self) -> Variable:
        """Allocate ``pred0``, ``pred1``, … (Java ``getFreshPredicateVariable``)."""
        v = Variable(PREDICATE_VARIABLE_ROOT + str(len(self._predicate_pool)))
        self._predicate_pool.append(v)
        return v

    def get_root_node(self) -> Node:
        """Return the tree root node (Java ``getRootNode``)."""
        return self[self.root_addr]

    def node_at(self, addr: NodeAddress) -> Node | None:
        """Return the node at *addr* or ``None`` (Java ``get``)."""
        return self.get(addr)

    def get_node(self, modality: Modality) -> Node | None:
        """Node reached from the pointer via *modality* (Java ``getNode(Modality)``)."""
        dest = self.pointer.go_modality(modality)
        if dest is None:
            return None
        return self.get(dest)

    def get_daughters(self, node: Node, order: str | None = None) -> list[Node]:
        """Daughter nodes in fixed order, optionally filtered by *order* chars (Java ``getDaughters``)."""
        addr = node.address
        seq: list[NodeAddress] = []
        if order is None:
            seq = [
                addr.down0(),
                addr.down1(),
                addr.down_link(),
                addr.down_star(),
                addr.down_local_unfixed(),
            ]
        else:
            for ch in order:
                seq.append(addr.down_char(ch))
        out: list[Node] = []
        for a in seq:
            n = self.get(a)
            if n is not None:
                out.append(n)
        return out

    def get_unfixed_nodes(self) -> list[Node]:
        """Nodes on star- or local-unfixed addresses (Java ``getUnfixedNodes``)."""
        out: list[Node] = []
        for n in self.values():
            a = n.address.address
            if a.endswith(PATH_UNFIXED) or a.endswith(PATH_LOCAL_UNFIXED):
                out.append(n)
        return out

    def _move_daughters(self, dtrs: list[Node], from_addr: NodeAddress, to_addr: NodeAddress) -> None:
        """Re-home subtrees under *to_addr* (Java ``moveDaughters``)."""
        from_s = from_addr.address
        to_s = to_addr.address
        for dtr in dtrs:
            self._move_daughters(self.get_daughters(dtr), from_addr, to_addr)
            old_a = dtr.address.address
            if not old_a.startswith(from_s):
                raise RuntimeError(f"moveDaughters: {old_a!r} does not start with {from_s!r}")
            new_addr = NodeAddress(to_s + old_a[len(from_s) :])
            new_node = Node(new_addr, list(dtr.labels))
            self[new_addr] = new_node
            del self[dtr.address]

    def merge(self, modality: Modality) -> None:
        """Merge the node at *modality* into the pointed node (Java ``Tree.merge``)."""
        other = self.get_node(modality)
        if other is None:
            raise RuntimeError("merge: no node at modality")
        self._move_daughters(self.get_daughters(other), other.address, self.pointer)
        self.pointed_node.merge_from(other)
        del self[other.address]

    def merge_node(self, node: Node) -> None:
        """Merge *node* into the pointed node (Java ``Tree.merge(Node)``)."""
        self._move_daughters(self.get_daughters(node), node.address, self.pointer)
        self.pointed_node.merge_from(node)
        del self[node.address]

    def merge_unfixed(self) -> list[Tree]:
        """Return one or two trees after optional unfixed merge (Java ``mergeUnfixed``)."""
        original = self.clone()
        result = self.clone()
        merged = False
        is_late_unfixed = False
        et_merge = False
        for unfixed in result.get_unfixed_nodes():
            merge_point_f_chosen: FormulaLabel | None = None
            merge_point_chosen: Node | None = None
            for merge_point in result.values():
                if not merge_point.is_locally_fixed():
                    continue
                if len(result.get_daughters(merge_point, "01")) > 0:
                    continue
                if not unfixed.is_unifiable(merge_point):
                    continue
                result.set_pointer(merge_point.address)
                mfl = result.pointed_node.get_formula_label()
                if mfl is not None and unfixed.get_formula_label() is not None:
                    merge_point_f_chosen = result.pointed_node.get_formula_label()
                    merge_point_chosen = result.pointed_node
                addr_s = unfixed.address.address
                is_late_unfixed = addr_s.endswith(PATH_UNFIXED) and addr_s != "0" + PATH_UNFIXED
                et_merge = merge_point.address.address == "01"
                result.merge_node(unfixed)
                merged = True
                break
            if merge_point_chosen is not None and merge_point_f_chosen is not None:
                merge_point_chosen.remove_formula_label()
        if merged and not is_late_unfixed and not et_merge:
            return [original, result]
        if merged and (is_late_unfixed or et_merge):
            return [result]
        return [original]

    def _add_underspecified_formulae(self, context: Any) -> None:
        """Attach underspecified ``Fo`` from static ``typeMap`` (Java ``addUnderspecifiedFormulae(Context)``)."""
        type_map = get_static_type_map()
        for n in self.values():
            if len(self.get_daughters(n, "01")) > 0:
                continue
            ds_type = n.get_required_type() or n.get_type()
            fo = n.get_formula()
            if ds_type is None or fo is not None:
                continue
            if ds_type not in type_map:
                if ds_type != DSType.t:
                    logger.warning(
                        "could not add underspecified formula; type not in typeMap: %s at %s",
                        ds_type,
                        n.address,
                    )
                continue
            tmpl = type_map[ds_type]
            fresh = tmpl.freshen_vars(self)
            n.add_label(FormulaLabel(fresh))
        logger.debug("After adding underspec formulae: %s", self)

    def _add_underspecified_formulae_no_context(self) -> None:
        """Java no-arg ``addUnderspecifiedFormulae`` — use static map plus ``e>t`` / ``?Ex.fo`` special case."""
        from dylan.formula.formula import Formula
        from dylan.tree.label.labels import label_factory_create

        type_map = get_static_type_map()
        form_req_label = label_factory_create("?Ex.fo(x)")
        e_gt = DSType.parse("e>t")
        for n in self.values():
            if len(self.get_daughters(n, "01")) > 0:
                continue
            ds_type = n.get_required_type() or n.get_type()
            fo = n.get_formula()
            if ds_type is None or fo is not None:
                continue
            if ds_type not in type_map:
                if ds_type != DSType.t:
                    logger.warning(
                        "could not add underspecified formula; type not in typeMap: %s at %s",
                        ds_type,
                        n.address,
                    )
                continue
            if e_gt is not None and ds_type == e_gt and n.contains(form_req_label):
                sp = Formula.create(
                    "R1^(R1 ++ [e1:es|head==e1:es|p==subj(e1,R1.head):t])",
                )
                if sp is not None:
                    n.add_label(FormulaLabel(sp.freshen_vars(self)))
                    continue
            tmpl = type_map[ds_type]
            n.add_label(FormulaLabel(tmpl.freshen_vars(self)))

    def _max_sem_at(self, root: Node, context: Any) -> "TTRFormula":
        """Maximal TTR at *root* (Java ``getMaximalSemantics(Node, Context)``)."""
        from dylan.formula.ttr_formula import TTRFormula
        from dylan.formula.ttr_lambda import TTRLambdaAbstract
        from dylan.formula.ttr_record_type import TTRRecordType

        d01 = self.get_daughters(root, "01")
        if len(d01) == 1:
            logger.error("node with only one fixed daughter: %s", root)
            return TTRRecordType()

        unfixed_n = self.node_at(root.address.down_star())
        local_unfixed_n = self.node_at(root.address.down_local_unfixed())
        unfixed_functor = False
        unfixed_reduced: TTRFormula | None = None  # type: ignore[name-defined]
        if unfixed_n is not None:
            unfixed_reduced = self._max_sem_at(unfixed_n, context)
            if isinstance(unfixed_reduced, TTRLambdaAbstract):
                img = TTRRecordType.parse("[x:e|head==x:e]")
                assert img is not None
                unfixed_reduced = unfixed_reduced.beta_reduce(img)
                unfixed_functor = True

        local_unfixed_reduced: TTRFormula | None = None  # type: ignore[name-defined]
        if local_unfixed_n is not None:
            local_unfixed_reduced = self._max_sem_at(local_unfixed_n, context)
            if isinstance(local_unfixed_reduced, TTRLambdaAbstract):
                img2 = TTRRecordType.parse("[x:e|head==x:e]")
                assert img2 is not None
                local_unfixed_reduced = local_unfixed_reduced.beta_reduce(img2)
                unfixed_functor = True

        root_fo = root.get_formula()
        if root_fo is None:
            root_reduced = TTRRecordType()
        elif isinstance(root_fo, TTRFormula):
            root_reduced = root_fo
        else:
            root_reduced = TTRRecordType()

        if len(self.get_daughters(root, "01")) == 2:
            d0 = self.node_at(root.address.down0())
            d1 = self.node_at(root.address.down1())
            assert d0 is not None and d1 is not None
            arg_max = self._max_sem_at(d0, context)
            funct_max = self._max_sem_at(d1, context)
            if not isinstance(funct_max, TTRLambdaAbstract):
                raise TypeError(f"expected TTR lambda at functor, got {type(funct_max).__name__}")
            logger.debug("beta-reducing functor %s arg %s", funct_max, arg_max)
            root_reduced = funct_max.beta_reduce(arg_max)
            logger.debug("beta result %s", root_reduced)
            if unfixed_reduced is not None:
                root_reduced = root_reduced.conjoin(unfixed_reduced.remove_head())
            if local_unfixed_reduced is not None:
                root_reduced = root_reduced.conjoin(local_unfixed_reduced.remove_head())
        else:
            if unfixed_reduced is not None and local_unfixed_reduced is not None:
                root_reduced = root_reduced.conjoin(
                    unfixed_reduced.remove_head().conjoin(local_unfixed_reduced.remove_head()),
                )
            elif unfixed_reduced is not None:
                root_reduced = (
                    unfixed_reduced
                    if unfixed_functor
                    else root_reduced.conjoin(unfixed_reduced)
                )
            elif local_unfixed_reduced is not None:
                root_reduced = (
                    local_unfixed_reduced
                    if unfixed_functor
                    else root_reduced.conjoin(local_unfixed_reduced)
                )

        if len(self.get_daughters(root, "L")) > 0:
            link_n = self.node_at(root.address.down_link())
            if link_n is not None:
                max_sem_l = self._max_sem_at(link_n, context)
                root_reduced = root_reduced.conjoin(max_sem_l)

        q_lab = FeatureLabel("Q")
        if root.contains(q_lab):
            qr = _question_record().freshen_vars(self)
            root_reduced = qr.conjoin(root_reduced)

        return root_reduced

    def get_maximal_semantics(self, context: Any = None) -> "TTRFormula":
        """Compute maximal TTR semantics (Java ``getMaximalSemantics(Context)``)."""
        from dylan.formula.disjunctive_type import DisjunctiveType
        from dylan.formula.ttr_formula import TTRFormula

        logger.debug("Merging unfixed if possible; before: %s", self)
        work = self.clone()
        merged_list = work.merge_unfixed()
        logger.debug("after merge: %s", merged_list)
        if len(merged_list) > 2:
            raise NotImplementedError("more than two trees after mergeUnfixed")

        first = merged_list[0]
        if context is not None:
            first._add_underspecified_formulae(context)
        else:
            first._add_underspecified_formulae_no_context()

        if len(merged_list) == 1:
            sem = first._max_sem_at(first.get_root_node(), context)
            ev = sem.evaluate()
            return ev if isinstance(ev, TTRFormula) else sem

        second = merged_list[1]
        if context is not None:
            second._add_underspecified_formulae(context)
        else:
            second._add_underspecified_formulae_no_context()

        a = first._max_sem_at(first.get_root_node(), context)
        b = second._max_sem_at(second.get_root_node(), context)
        ea, eb = a.evaluate(), b.evaluate()
        return DisjunctiveType(
            ea if isinstance(ea, TTRFormula) else a,
            eb if isinstance(eb, TTRFormula) else b,
        )

    def get_maximal_semantics_with_context(self, context: Any) -> "TTRFormula":
        return self.get_maximal_semantics(context)

    # ── tree-modifying operations (Java Tree.make / go / put / delete) ──

    def make(self, op: BasicOperator) -> None:
        """Create a new daughter node below the pointed node (Java ``Tree.make``)."""
        if not op.is_down():
            raise RuntimeError(f"Can't make non-daughter node {op}")
        addr = self.pointer.go_op(op)
        if addr is not None and addr not in self:
            self[addr] = Node(addr)

    def go(self, mod: Modality) -> None:
        """Move the pointer along *mod* (Java ``Tree.go(Modality)``)."""
        addr = self.pointer.go_modality(mod)
        if addr is None or addr not in self:
            raise RuntimeError(f"Can't go to non-existent node from {self.pointer} via {mod}")
        self.pointer = addr

    def go_op(self, op: BasicOperator) -> None:
        """Move the pointer one step (Java ``Tree.go(BasicOperator)``)."""
        addr = self.pointer.go_op(op)
        if addr is not None:
            self.pointer = addr

    def put_label(self, label: Label) -> None:
        """Add a label at the pointed node (Java ``Tree.put``)."""
        self.pointed_node.add_label(label)

    put = put_label

    def delete_label(self, label: Label) -> None:
        """Remove a label from the pointed node (Java ``Tree.delete``)."""
        self.pointed_node.remove_label(label)

    def is_complete(self) -> bool:
        """True when pointer is at root and no node carries a :class:`Requirement` (Java ``Tree.isComplete``)."""
        from dylan.tree.label.labels import Requirement as _Req

        if not self.pointer.is_root():
            return False
        for node in self.values():
            for lab in node.labels:
                if isinstance(lab, _Req):
                    return False
        return True


Tree.setPointer = Tree.set_pointer  # type: ignore[attr-defined]
Tree.getFreshEntityVariable = Tree.get_fresh_entity_variable  # type: ignore[attr-defined]
Tree.getFreshEventVariable = Tree.get_fresh_event_variable  # type: ignore[attr-defined]
Tree.getFreshPropositionVariable = Tree.get_fresh_proposition_variable  # type: ignore[attr-defined]
Tree.getFreshRecTypeVariable = Tree.get_fresh_record_type_variable  # type: ignore[attr-defined]
Tree.getFreshRecordTypeVariable = Tree.get_fresh_record_type_variable  # type: ignore[attr-defined]
Tree.getFreshPredicateVariable = Tree.get_fresh_predicate_variable  # type: ignore[attr-defined]
Tree.getRootNode = Tree.get_root_node  # type: ignore[attr-defined]
Tree.getNode = Tree.get_node  # type: ignore[attr-defined]
Tree.getDaughters = Tree.get_daughters  # type: ignore[attr-defined]
Tree.getUnfixedNodes = Tree.get_unfixed_nodes  # type: ignore[attr-defined]
Tree.mergeNode = Tree.merge_node  # type: ignore[attr-defined]
Tree.mergeUnfixed = Tree.merge_unfixed  # type: ignore[attr-defined]
Tree.getMaximalSemantics = Tree.get_maximal_semantics  # type: ignore[attr-defined]
Tree.getMaximalSemanticsWithContext = Tree.get_maximal_semantics_with_context  # type: ignore[attr-defined]
Tree.goOp = Tree.go_op  # type: ignore[attr-defined]
Tree.putLabel = Tree.put_label  # type: ignore[attr-defined]
Tree.deleteLabel = Tree.delete_label  # type: ignore[attr-defined]
Tree.isComplete = Tree.is_complete  # type: ignore[attr-defined]
