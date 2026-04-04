"""Formula base and string factory (partial `qmul.ds.formula.Formula`)."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dylan.formula.ttr_record_type import TTRRecordType
    from dylan.formula.variable import Variable

logger = logging.getLogger(__name__)

REC_METAVARIABLE_PATTERN = re.compile(r"^REC\d*$", re.IGNORECASE)
FORMULA_METAVARIABLE_PATTERN = re.compile(r"^U[1-9]*$")
_FRESHPUT_META_FORMULA = re.compile(r"^[S-U]$")


class Formula(ABC):
    """Semantic formula appearing on DS nodes (Java `Formula`)."""

    def __init__(self) -> None:
        self._parent_rec_type: TTRRecordType | None = None

    @property
    def parent_rec_type(self) -> TTRRecordType | None:
        return self._parent_rec_type

    @parent_rec_type.setter
    def parent_rec_type(self, r: TTRRecordType | None) -> None:
        self._parent_rec_type = r

    @abstractmethod
    def clone(self) -> Formula:
        raise NotImplementedError

    def instantiate(self) -> Formula:
        """Replace metavariables with values; default is a deep copy (Java ``Formula.instantiate``)."""
        return self.clone()

    def evaluate(self) -> Formula:
        """Simplify lazy operators; default is identity (Java ``Formula.evaluate``)."""
        return self

    def conjoin(self, other: Formula) -> Formula:
        """Conjoin with *other*; subclasses implement (Java ``Formula.conjoin``)."""
        raise TypeError(f"conjoin not implemented for {type(self).__name__} with {type(other).__name__}")

    def substitute(self, var: Variable, arg: Formula) -> Formula:
        """Replace occurrences of *var* with *arg* (Java ``Formula.substitute`` default: no change)."""
        return self

    @staticmethod
    def create(string: str, in_ex_conj: bool = False) -> Formula | None:  # noqa: ARG004
        """Parse formula specs from lexicon / TTR (partial implementation)."""
        from dylan.formula.ttr_record_type import TTRRecordType
        from dylan.formula.variable import Variable

        s = string.strip()
        if Variable.is_variable_string(s):
            return Variable(s)
        if _FRESHPUT_META_FORMULA.match(s):
            from dylan.action.meta.meta_formula import MetaFormula

            return MetaFormula.get(s)
        if FORMULA_METAVARIABLE_PATTERN.match(s):
            from dylan.formula.formula_metavariable import FormulaMetavariable

            return FormulaMetavariable.get(s)
        if REC_METAVARIABLE_PATTERN.match(s):
            from dylan.formula.meta_ttr_record_type import MetaTTRRecordType

            return MetaTTRRecordType.get(s)
        if "^" in s:
            caret = s.find("^")
            var_s = s[:caret].strip()
            body_s = s[caret + 1 :].strip()
            if Variable.is_variable_string(var_s):
                body = Formula.create(body_s)
                if body is not None:
                    from dylan.formula.fol_lambda import FOLLambdaAbstract

                    return FOLLambdaAbstract(Variable(var_s), body)
        m = re.match(r"^([a-z][a-z][a-z_0-9]*)\((.+)\)$", s)
        if m:
            from dylan.formula.predicate_argument import Predicate, PredicateArgumentFormula

            pred = Predicate(m.group(1))
            args_raw = m.group(2)
            parts = [p.strip() for p in args_raw.split(",")]
            args: list[Formula] = []
            for p in parts:
                f = Formula.create(p)
                if f is None:
                    return None
                args.append(f)
            return PredicateArgumentFormula(pred, tuple(args))
        rt = TTRRecordType.parse(s)
        if rt is not None:
            return rt
        logger.debug("Formula.create could not parse: %s", string)
        return None
