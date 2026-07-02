"""DAG induction tuple carrying current and non-head target trees (Java ``DAGInductionTuple``)."""

from __future__ import annotations

from typing import Any

from dylan.dag.dag_tuple import DAGTuple
from dylan.dag.parser_tuple import ParserTuple
from dylan.formula.ttr_formula import TTRFormula
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.tree.tree import Tree


class DAGInductionTuple(DAGTuple):
    """Specialised :class:`DAGTuple` that also carries induction-time target trees."""

    def __init__(
        self,
        tree_or_tuple: "Tree | ParserTuple | DAGInductionTuple | None" = None,
        tuple_id: int = 0,
    ) -> None:
        """Java overloads ``DAGInductionTuple(long)``, ``(Tree, long)``, ``(ParserTuple, long)``, ``(Tree)``."""
        if tree_or_tuple is None:
            super().__init__(None, tuple_id)
        elif isinstance(tree_or_tuple, Tree):
            super().__init__(tree_or_tuple, tuple_id)
        elif isinstance(tree_or_tuple, DAGInductionTuple):
            super().__init__(tree_or_tuple.get_tree().clone(), tuple_id)
            self._gold_target_type: TTRRecordType | None = tree_or_tuple.get_gold_target_type()
        elif isinstance(tree_or_tuple, ParserTuple):
            super().__init__(tree_or_tuple.get_tree(), tuple_id)
        else:
            raise TypeError(f"DAGInductionTuple: bad tree_or_tuple {tree_or_tuple!r}")
        if not isinstance(tree_or_tuple, DAGInductionTuple):
            self._gold_target_type: TTRRecordType | None = None
        self._cur_target: Tree = Tree()
        self._cur_non_head_target: Tree = Tree()

    def get_target_tree(self) -> Tree:
        """Return the current target tree (Java ``getTargetTree``)."""
        return self._cur_target

    def set_target(self, t: Tree) -> None:
        """Replace the current target tree (Java ``setTarget``)."""
        self._cur_target = t

    def get_non_head_target(self) -> Tree:
        """Return the non-head target tree (Java ``getNonHeadTarget``)."""
        return self._cur_non_head_target

    def set_non_head_target(self, t: Tree) -> None:
        """Replace the non-head target tree (Java ``setNonHeadTarget``)."""
        self._cur_non_head_target = t

    def get_gold_target_type(self) -> TTRRecordType | None:
        """Return the corpus gold :class:`TTRRecordType` for metavar domain binding (Java full ``targetType``)."""
        return self._gold_target_type

    def set_gold_target_type(self, t: TTRRecordType | None) -> None:
        """Attach the training example gold record for induction-time formula binding."""
        self._gold_target_type = t.clone() if t is not None and hasattr(t, "clone") else t

    def get_semantics(self, context: Any = None) -> TTRFormula:
        """Compute maximal semantics (Java ``ParserTuple.getSemantics`` / ``getSemantics(Context)``)."""
        if self.semantics is not None:
            return self.semantics
        if context is None:
            return self.tree.get_maximal_semantics(None)
        use_induction = (
            hasattr(context, "get_gold_target_type")
            and context.get_gold_target_type() is not None
        )
        return self.tree.get_maximal_semantics(context, induction_mode=use_induction)

    def get_fresh_entity_variable(self) -> Any:
        """Delegate fresh-variable allocation to the tuple tree (Java ``Context``)."""
        return self.tree.get_fresh_entity_variable()

    def get_fresh_event_variable(self) -> Any:
        """Delegate fresh-variable allocation to the tuple tree."""
        return self.tree.get_fresh_event_variable()

    def get_fresh_proposition_variable(self) -> Any:
        """Delegate fresh-variable allocation to the tuple tree."""
        return self.tree.get_fresh_proposition_variable()

    def get_fresh_record_type_variable(self) -> Any:
        """Delegate fresh-variable allocation to the tuple tree."""
        return self.tree.get_fresh_record_type_variable()

    def get_fresh_predicate_variable(self) -> Any:
        """Delegate fresh-variable allocation to the tuple tree."""
        return self.tree.get_fresh_predicate_variable()


DAGInductionTuple.getTargetTree = DAGInductionTuple.get_target_tree  # type: ignore[attr-defined]
DAGInductionTuple.setTarget = DAGInductionTuple.set_target  # type: ignore[attr-defined]
DAGInductionTuple.getNonHeadTarget = DAGInductionTuple.get_non_head_target  # type: ignore[attr-defined]
DAGInductionTuple.setNonHeadTarget = DAGInductionTuple.set_non_head_target  # type: ignore[attr-defined]
DAGInductionTuple.getGoldTargetType = DAGInductionTuple.get_gold_target_type  # type: ignore[attr-defined]
DAGInductionTuple.setGoldTargetType = DAGInductionTuple.set_gold_target_type  # type: ignore[attr-defined]
DAGInductionTuple.getSemantics = DAGInductionTuple.get_semantics  # type: ignore[attr-defined]
