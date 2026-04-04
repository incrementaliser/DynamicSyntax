"""``unreduce`` — walk up the tree restoring requirements (Java ``Unreduce``)."""

from __future__ import annotations

import logging
from typing import Any

from dylan.action.atomic.effect import Effect
from dylan.context.context import Context
from dylan.dag.dag_tuple import DAGTuple
from dylan.tree.label.labels import BottomLabel, FeatureLabel, FormulaLabel, Requirement, TypeLabel, label_factory_create
from dylan.tree.modality import Modality
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree
from dylan.type.dstype import DSType

logger = logging.getLogger(__name__)


class Unreduce(Effect):
    """Move up from the pointer with ``</\\>`` and adjust types/formulas (Java ``Unreduce``)."""

    FUNCTOR = "unreduce"

    def exec(self, tree: Tree, context: Context[DAGTuple, Any] | None) -> Tree | None:
        cur = tree.pointer
        up = Modality.parse("</\\>")
        bottom = BottomLabel()
        eval_label = label_factory_create("?+eval")
        while True:
            nxt = cur.go_modality(up)
            if nxt is None:
                break
            cur = nxt
            cur_node = tree[cur]
            ty = cur_node.get_type()
            if ty is None:
                return tree
            tyl = cur_node.get_type_label()
            fol = cur_node.get_formula_label()
            if len(tree.get_daughters(cur_node, "01")) == 2:
                if tyl is not None:
                    cur_node.remove_label(tyl)
                    cur_node.add_label(Requirement(TypeLabel(ty)))
                if fol is not None:
                    cur_node.remove_label(fol)
            if tree.get_daughters(cur_node, "L"):
                cur_node.add_label(eval_label)
            if any(isinstance(lab, BottomLabel) for lab in cur_node.labels):
                for lab in list(cur_node.labels):
                    if isinstance(lab, BottomLabel):
                        cur_node.remove_label(lab)
            if cur == NodeAddress():
                break
        return tree

    def exec_tuple_context(self, tree: Tree, context: Any) -> Tree | None:
        return self.exec(tree, None)

    def instantiate(self) -> Effect:
        return Unreduce()

    def __str__(self) -> str:
        return self.FUNCTOR
