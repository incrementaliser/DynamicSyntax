"""Formula base and string factory (partial `qmul.ds.formula.Formula`)."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from dylan.formula.variable import Variable

if TYPE_CHECKING:
    from dylan.formula.ttr_record_type import TTRRecordType

logger = logging.getLogger(__name__)

REC_METAVARIABLE_PATTERN = re.compile(r"^REC\d*$", re.IGNORECASE)
FORMULA_METAVARIABLE_PATTERN = re.compile(r"^U[1-9]*$")


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

    @staticmethod
    def create(string: str, in_ex_conj: bool = False) -> Formula | None:  # noqa: ARG004
        """Parse formula specs from lexicon / TTR (partial implementation)."""
        from dylan.formula.ttr_record_type import TTRRecordType

        s = string.strip()
        if Variable.is_variable_string(s):
            return Variable(s)
        if FORMULA_METAVARIABLE_PATTERN.match(s):
            from dylan.formula.formula_metavariable import FormulaMetavariable

            return FormulaMetavariable.get(s)
        if REC_METAVARIABLE_PATTERN.match(s):
            from dylan.formula.meta_ttr_record_type import MetaTTRRecordType

            return MetaTTRRecordType.get(s)
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
