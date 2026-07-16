"""TTR formula abstract base (Java `TTRFormula`).

Ports the Java `TTRFormula.getAbstractions` tree-builder: the multi-arity
overload that turns a list of basic argument types into a *DS abstraction
tree* annotated with TTR record types.  All comments and quirks
(``AA DANGEROUS test``, premature trees, the ``cn``-shortcut, the
``try/except`` around lower abstractions, the cache by ``(rt, type)`` key)
are preserved.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from dylan.formula.formula import Formula

if TYPE_CHECKING:
    from dylan.formula.ttr_lambda import TTRLambdaAbstract
    from dylan.formula.ttr_record_type import TTRRecordType
    from dylan.tree.node_address import NodeAddress
    from dylan.tree.tree import Tree
    from dylan.type.dstype import BasicType, DSType

logger = logging.getLogger(__name__)


def resolve_freshen_tree(context: Any) -> Any:
    """Return the :class:`~dylan.tree.tree.Tree` used for variable pools (Java ``Context`` / ``Tree``)."""
    from dylan.dag.dag_induction_tuple import DAGInductionTuple
    from dylan.dag.parser_tuple import ParserTuple

    if isinstance(context, (DAGInductionTuple, ParserTuple)):
        return context.get_tree()
    return context


# Module-level abstraction-order counter (Java ``TTRFormula.abstractionOrder``).
abstraction_order: int = 1


class TTRFormula(Formula):
    """Formulae used in TTR semantics; carries the abstraction-tree builder."""

    # Java ``abstractsCache``: (TTRRecordType, BasicType) -> list[(rt, lambda)]
    def __init__(self) -> None:
        """Initialise the per-instance abstraction cache (Java ``abstractsCache``)."""
        super().__init__()
        self.abstracts_cache: dict[tuple[Any, Any], list[tuple[TTRRecordType, TTRLambdaAbstract]]] = {}

    # ------------------------------------------------------------------ abstract API

    @abstractmethod
    def clone(self) -> TTRFormula:
        """Return a deep copy (Java ``TTRFormula.clone``)."""
        raise NotImplementedError

    @abstractmethod
    def asymmetric_merge(self, rt: TTRFormula) -> TTRFormula:
        """Right-asymmetrical merge (Java ``TTRFormula.asymmetricMerge``)."""
        raise NotImplementedError

    def conjoin(self, other: Formula | None) -> TTRFormula:
        """``other.asymmetric_merge(self)`` (Java ``TTRFormula.conjoin``)."""
        if other is None:
            return self
        if isinstance(other, TTRFormula):
            return other.asymmetric_merge(self)
        raise TypeError(
            f"Can only conjoin TTRFormula with TTRFormula, got {type(other).__name__}",
        )

    def remove_head(self) -> TTRFormula:
        """Strip head field when present; default raises (Java ``removeHead``)."""
        raise NotImplementedError(f"removeHead unsupported for {type(self).__name__}")

    def remove_head_if_manifest(self) -> TTRFormula:
        """Strip the head field only when manifest (Java ``removeHeadIfManifest``)."""
        raise NotImplementedError(f"removeHeadIfManifest unsupported for {type(self).__name__}")

    def subsumes(self, other: object) -> bool:
        """Java ``Formula.subsumes``: string/basic equality then ``subsumesMapped``.

        Subclasses override ``subsumes_basic`` / ``subsumes_mapped`` (not this method),
        so ``TTRFreshPut`` mutual subsumption can α-rename via mapped labels.
        """
        if not isinstance(other, Formula):
            return False
        if self == other or str(self) == str(other):
            return True
        if self.subsumes_basic(other):
            return True
        return self.subsumes_mapped(other, {})

    def evaluate(self) -> TTRFormula:
        """Resolve metavariables / lazy ops; default returns ``self`` (Java ``TTRFormula.evaluate``)."""
        return self

    def freshen_vars(self, context: Any, var_map: dict[Any, Any] | None = None) -> TTRFormula:
        """Alpha-rename using a tree/context, or relative to a gold record and *var_map* (Java overloads)."""
        if var_map is not None:
            from dylan.formula.ttr_record_type import TTRRecordType

            if not isinstance(context, TTRRecordType):
                raise TypeError("freshen_vars with var_map expects a TTRRecordType gold record")
            return self.freshen_vars_mapped(context, var_map)
        tree = resolve_freshen_tree(context)
        return self.freshen_vars_tree(tree)

    def freshen_vars_tree(self, tree: Any) -> TTRFormula:
        """Alpha-rename using *tree* variable pools (default: ``clone``)."""
        return self.clone()

    def freshen_vars_mapped(self, gold: "TTRRecordType", var_map: dict[Any, Any]) -> TTRFormula:
        """Rename relative to gold record *gold*; updates *var_map* (Java ``freshenVars(TTRRecordType, Map)``)."""
        return self.clone()

    def instantiate(self) -> TTRFormula:
        """Default: deep copy (Java ``TTRFormula.instantiate``)."""
        return self.clone()

    def sort_fields_by_specificity(self) -> TTRFormula:
        """Used when computing common supertypes; default raises (Java ``sortFieldsBySpecificity``)."""
        raise TypeError(f"Cannot sort_fields_by_specificity on {type(self).__name__}")

    def get_head_field(self) -> Any | None:
        """Return the head ``TTRField`` if any (Java ``getHeadField``); base class returns None."""
        return None

    # ------------------------------------------------------------------ abstraction tree builder

    def get_abstractions_basic(
        self,
        basic: BasicType,
        new_var_suffix: int = 1,
    ) -> list[tuple[TTRRecordType, TTRLambdaAbstract]]:
        """Pair-of-(rt, λ) abstractions over *basic* (Java RT-level ``getAbstractions(BasicType, int)``).

        TTRFormula subclasses with fields override this to produce the actual
        ``(remainder, lambda)`` pairs.  Default returns an empty list.
        """
        _ = (basic, new_var_suffix)
        return []

    def get_abstractions(
        self,
        first: Any,
        second: Any | None = None,
        third: Any | None = None,
    ) -> Any:
        """Multi-arity dispatcher mirroring Java ``getAbstractions`` overloads.

        Forms:

        * ``get_abstractions(funcType, prefix)`` (Java 2-arg) → list[Tree]
        * ``get_abstractions(funcType, root, rootType)`` (Java 3-arg) → list[Tree]
        * ``get_abstractions(typesList, root, rootType)`` (Java 3-arg) → list[Tree]
        * ``get_abstractions(basic, newVarSuffix)`` (Java RT-level 2-arg) → list[(rt, λ)]
        """
        from dylan.tree.node_address import NodeAddress as _NA
        from dylan.type.dstype import BasicType as _BT, DSType as _DST

        if second is None:
            return self.get_abstractions_basic(first, 1)

        if isinstance(first, _DST) and isinstance(second, _NA) and third is None:
            return self._abstractions_from_func_type(first, second, first.get_final_type())

        if isinstance(first, _DST) and isinstance(second, _NA):
            return self._abstractions_from_func_type(first, second, third)

        if isinstance(first, list):
            return self._abstractions_from_types(first, second, third)

        if isinstance(first, _BT) and isinstance(second, int):
            return self.get_abstractions_basic(first, second)

        raise TypeError(
            f"get_abstractions: unsupported overload {type(first).__name__}, "
            f"{type(second).__name__ if second is not None else None}, "
            f"{type(third).__name__ if third is not None else None}",
        )

    def _abstractions_from_func_type(
        self,
        func_type: DSType,
        root: NodeAddress,
        root_type: DSType,
    ) -> list[Tree]:
        """Java 3-arg form: pull subject-first basic types from *func_type*, drop the result, recurse."""
        types = list(func_type.get_types_subj_first())
        logger.debug("list of basic types on funcType %s is: %s", func_type, types)
        if types:
            types.pop(0)
        return self._abstractions_from_types(types, root, root_type)

    def _abstractions_from_types(
        self,
        types: list[BasicType],
        root: NodeAddress,
        root_type: DSType,
    ) -> list[Tree]:
        """Main Java loop — recursively build abstraction trees decorated with TTR labels.

        Under the ``childes`` profile, cn nesting matches pre-BabyDS Java
        (``argumentAbstracts.size() == 1`` only, no premature multi-cn trees).
        BabyDS keeps the tip ``!isEmpty()`` + premature path.
        """
        from dylan.induction.corpus_profile import get_active_profile
        from dylan.tree.basic_operator import BasicOperator
        from dylan.tree.label.labels import FormulaLabel, Requirement, TypeLabel
        from dylan.tree.tree import Tree
        from dylan.type.dstype import DSType as _DST

        pre_babyds_cn = get_active_profile().name == "childes"
        result: list[Tree] = []
        logger.debug("Abstractions for: %s", types)
        logger.debug("on: %s", self)

        if not types:
            local = Tree(root)
            logger.debug("Reached base! returning one node tree...")
            local.get_pointed_node().add_label(FormulaLabel(self))
            local.get_pointed_node().add_label(TypeLabel(root_type))
            local.get_pointed_node().remove_label(Requirement(TypeLabel(_DST.t)))
            logger.debug("Constructed: %s", local)
            result.append(local)
            return result

        basic = types[0]
        basic_abstracts = self.get_abstractions_basic(basic, 1)
        logger.debug("Got %d basic abstractions with %s", len(basic_abstracts), basic)

        for rt, lam in basic_abstracts:
            local = Tree(root)
            logger.debug("constructing local tree at: %s", root)
            local.get_pointed_node().add_label(FormulaLabel(self))
            local.get_pointed_node().add_label(TypeLabel(root_type))
            local.make(BasicOperator.DOWN_0)
            local.go_op(BasicOperator.DOWN_0)

            key = (rt, _DST.cn)
            cache = self.abstracts_cache
            if key in cache:
                argument_abstracts = cache[key]
            else:
                argument_abstracts = rt.get_abstractions_basic(_DST.cn, 1)
                cache[key] = argument_abstracts
            logger.debug("cn abstracts size: %d", len(argument_abstracts))

            premature_tree_copies: list[Tree] = []
            nest_cn = (
                len(argument_abstracts) == 1
                if pre_babyds_cn
                else bool(argument_abstracts)
            )
            if nest_cn:
                arg0_rt, arg0_lam = argument_abstracts[0]
                local.make(BasicOperator.DOWN_1)
                local.go_op(BasicOperator.DOWN_1)
                local.put(FormulaLabel(arg0_lam))
                abstracted_type = _DST.create(_DST.cn, basic)
                local.put(TypeLabel(abstracted_type))
                local.go_op(BasicOperator.UP_1)
                local.make(BasicOperator.DOWN_0)
                local.go_op(BasicOperator.DOWN_0)
                local.put(TypeLabel(_DST.cn))
                local.put(FormulaLabel(arg0_rt))
                local.go_op(BasicOperator.UP_0)

                if not pre_babyds_cn:
                    for i in range(1, len(argument_abstracts)):
                        copy_local = local.clone()
                        child_rt, child_lam = argument_abstracts[i]
                        copy_local.go_op(BasicOperator.DOWN_0)
                        copy_local.make(BasicOperator.DOWN_1)
                        copy_local.go_op(BasicOperator.DOWN_1)
                        copy_local.put(FormulaLabel(child_lam))
                        abstracted_type2 = _DST.create(_DST.cn, _DST.cn)
                        copy_local.put(TypeLabel(abstracted_type2))
                        copy_local.go_op(BasicOperator.UP_1)
                        copy_local.make(BasicOperator.DOWN_0)
                        copy_local.go_op(BasicOperator.DOWN_0)
                        copy_local.put(TypeLabel(_DST.cn))
                        copy_local.put(FormulaLabel(child_rt))

                        copy_local.go_op(BasicOperator.UP_0)
                        logger.debug("constructed: %s", copy_local)
                        result.append(copy_local)
                        copy_local.go_op(BasicOperator.UP_0)

                        copy_local.put(TypeLabel(basic))
                        copy_local.put(FormulaLabel(rt))
                        copy_local.go_op(BasicOperator.UP_0)
                        copy_local.make(BasicOperator.DOWN_1)
                        copy_local.go_op(BasicOperator.DOWN_1)
                        copy_local.put(FormulaLabel(lam))
                        abstracted_type_left = _DST.create(basic, root_type)
                        copy_local.put(TypeLabel(abstracted_type_left))
                        logger.debug("constructed left: %s", copy_local)

                        try:
                            lower_abstracts = lam.get_abstractions(
                                types[1:], copy_local.get_pointer(), abstracted_type_left,
                            )
                        except Exception:  # noqa: BLE001 - mirrors Java try/catch
                            logger.warning("AA lower abstractions empty (caught exception).")
                            lower_abstracts = []

                        logger.debug("lower abstracts: %s", lower_abstracts)
                        for lower in lower_abstracts:
                            merged = copy_local.merge_tree(lower)
                            if merged.get_pointer().down1() in merged.key_set():
                                merged.go_op(BasicOperator.DOWN_1)
                            premature_tree_copies.append(merged)
                        if not lower_abstracts:
                            premature_tree_copies.append(copy_local)

            logger.debug("len of prematureTreeCopies: %d", len(premature_tree_copies))
            local.put(TypeLabel(basic))
            local.put(FormulaLabel(rt))
            local.go_op(BasicOperator.UP_0)
            local.make(BasicOperator.DOWN_1)
            local.go_op(BasicOperator.DOWN_1)
            local.put(FormulaLabel(lam))
            abstracted_type_left = _DST.create(basic, root_type)
            local.put(TypeLabel(abstracted_type_left))
            logger.debug("constructed: %s", local)

            lower_abstracts = lam.get_abstractions(types[1:], local.get_pointer(), abstracted_type_left)
            logger.debug("lower abstracts size: %d", len(lower_abstracts))
            for lower in lower_abstracts:
                merged = local.merge_tree(lower)
                if merged.get_pointer().down1() in merged.key_set():
                    merged.go_op(BasicOperator.DOWN_1)
                for premature in premature_tree_copies:
                    merged2 = premature.merge_tree(merged)
                    if merged2.get_pointer().down1() in merged2.key_set():
                        merged2.go_op(BasicOperator.DOWN_1)
                    result.append(merged2)
                result.append(merged)

        return result

    # ------------------------------------------------------------------ helpers used by RT abstractions

    def get_filtered_abstractions(
        self,
        prefix: NodeAddress,
        type_: DSType,
        filtering: bool,
    ) -> list[Tree]:
        """RT-level filtered abstractions; non-record formulae have none by default."""
        _ = (prefix, type_, filtering)
        return []

    def get_maximal_filtered_abstractions(
        self,
        prefix: NodeAddress,
        type_: DSType,
        filtering: bool,
    ) -> list[Tree]:
        """Maximally-extended filtered abstractions; default delegates to filtered."""
        return self.get_filtered_abstractions(prefix, type_, filtering)


# Camel-case aliases for Java compatibility.
TTRFormula.getAbstractions = TTRFormula.get_abstractions  # type: ignore[attr-defined]
TTRFormula.getFilteredAbstractions = TTRFormula.get_filtered_abstractions  # type: ignore[attr-defined]
TTRFormula.getMaximalFilteredAbstractions = TTRFormula.get_maximal_filtered_abstractions  # type: ignore[attr-defined]
TTRFormula.asymmetricMerge = TTRFormula.asymmetric_merge  # type: ignore[attr-defined]
TTRFormula.removeHead = TTRFormula.remove_head  # type: ignore[attr-defined]
TTRFormula.removeHeadIfManifest = TTRFormula.remove_head_if_manifest  # type: ignore[attr-defined]
TTRFormula.freshenVars = TTRFormula.freshen_vars  # type: ignore[attr-defined]
TTRFormula.sortFieldsBySpecificity = TTRFormula.sort_fields_by_specificity  # type: ignore[attr-defined]
TTRFormula.getHeadField = TTRFormula.get_head_field  # type: ignore[attr-defined]
