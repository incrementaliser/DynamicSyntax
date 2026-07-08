"""Formula base and string factory (partial `qmul.ds.formula.Formula`)."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dylan.formula.ttr_record_type import TTRRecordType
    from dylan.formula.variable import Variable

logger = logging.getLogger(__name__)


def _strip_matching_outer_parens(s: str) -> str | None:
    """If *s* is fully wrapped in one balanced ``(…)`` pair, return the inner string."""
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return None
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i < len(s) - 1:
                return None
    if depth != 0:
        return None
    return s[1:-1].strip()


def _split_top_level_commas(s: str) -> list[str]:
    """Split on commas not inside parentheses (predicate arguments)."""
    parts: list[str] = []
    depth = 0
    start = 0
    for i, c in enumerate(s):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(s[start:i].strip())
            start = i + 1
    parts.append(s[start:].strip())
    return parts


REC_METAVARIABLE_PATTERN = re.compile(r"^REC\d*$", re.IGNORECASE)
FORMULA_METAVARIABLE_PATTERN = re.compile(r"^U[1-9]*$")
_FRESHPUT_META_FORMULA = re.compile(r"^[S-U]$")
_REC_BINDER_PATTERN = re.compile(r"^R\d*$", re.IGNORECASE)
# Java ``LabelFactory.METAVARIABLE_PATTERN``: rule metavariables ``V``–``Z`` and ``META``.
_LEXICAL_FORMULA_METAVARIABLE = re.compile(r"^(?:[V-Z][0-9]*|META)$")
_ATOMIC_FORMULA_PATTERN = re.compile(r"^[a-z]+[a-z_0-9]*$")


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

    def subsumes(self, other: object) -> bool:
        """Structural subsumption (Java ``Formula.subsumes``: basic then mapped)."""
        if not isinstance(other, Formula):
            return False
        if self == other or str(self) == str(other):
            return True
        if self.subsumes_basic(other):
            return True
        return self.subsumes_mapped(other, {})

    def subsumes_basic(self, other: Formula) -> bool:
        """Quick subsumption without variable mapping (Java ``Formula.subsumesBasic``)."""
        return self == other

    def subsumes_mapped(self, other: Formula, map_: dict["Variable", "Variable"]) -> bool:
        """Mapped subsumption; default fails after basic (Java ``Formula.subsumesMapped``)."""
        _ = (other, map_)
        return False

    def freshen_vars(self, tree: Any) -> Formula:
        """Default: no renaming (Java ``Formula.freshenVars`` fallback)."""
        return self.clone()

    def get_variables(self) -> set["Variable"]:
        """Return the variables mentioned by this formula (Java ``Formula.getVariables``); base is empty."""
        return set()

    def get_ttr_paths(self) -> list["Formula"]:
        """Return TTR paths inside this formula (Java ``Formula.getTTRPaths``); base is empty."""
        return []

    def has_manifest_content(self) -> bool:
        """True by default; variables and unmanifest-head records return false (Java ``hasManifestContent``)."""
        return True

    @staticmethod
    def create(string: str, in_ex_conj: bool = False) -> Formula | None:
        """Parse formula specs from lexicon / TTR (partial implementation)."""
        from dylan.formula.bound_formula_variable import BoundFormulaVariable
        from dylan.formula.ttr_record_type import TTRRecordType
        from dylan.formula.variable import Variable

        s = string.strip()
        inner_paren = _strip_matching_outer_parens(s)
        if inner_paren is not None:
            inner_f = Formula.create(inner_paren, in_ex_conj)
            if inner_f is not None:
                return inner_f
        if not in_ex_conj and Variable.is_variable_string(s):
            return Variable(s)
        if _LEXICAL_FORMULA_METAVARIABLE.match(s):
            from dylan.action.meta.meta_formula import MetaFormula

            return MetaFormula.get(s)
        if _FRESHPUT_META_FORMULA.match(s):
            from dylan.action.meta.meta_formula import MetaFormula

            return MetaFormula.get(s)
        if FORMULA_METAVARIABLE_PATTERN.match(s):
            from dylan.formula.formula_metavariable import FormulaMetavariable

            return FormulaMetavariable.get(s)
        if REC_METAVARIABLE_PATTERN.match(s):
            from dylan.formula.meta_ttr_record_type import MetaTTRRecordType

            return MetaTTRRecordType.get(s)
        if in_ex_conj and BoundFormulaVariable.is_bound_name(s):
            return BoundFormulaVariable(s)
        if Variable.is_variable_string(s):
            return Variable(s)
        from dylan.formula.ttr_path import parse_ttr_path

        ttrp = parse_ttr_path(s)
        if ttrp is not None:
            return ttrp
        if "^" in s:
            caret = s.find("^")
            var_s = s[:caret].strip()
            body_s = s[caret + 1 :].strip()
            if Variable.is_variable_string(var_s):
                body = Formula.create(body_s, in_ex_conj)
                if body is None:
                    return None
                v = Variable(var_s)
                if _REC_BINDER_PATTERN.fullmatch(var_s):
                    from dylan.formula.ttr_formula import TTRFormula
                    from dylan.formula.ttr_lambda import TTRLambdaAbstract

                    if isinstance(body, TTRFormula):
                        return TTRLambdaAbstract(v, body)
                from dylan.formula.fol_lambda import FOLLambdaAbstract

                return FOLLambdaAbstract(v, body)
        m = re.match(r"^([a-z][a-z][a-z_0-9]*)\((.+)\)$", s)
        if m:
            from dylan.formula.predicate_argument import Predicate, PredicateArgumentFormula

            pred = Predicate(m.group(1))
            args_raw = m.group(2)
            parts = _split_top_level_commas(args_raw)
            args: list[Formula] = []
            for p in parts:
                f = Formula.create(p, in_ex_conj)
                if f is None:
                    return None
                args.append(f)
            return PredicateArgumentFormula(pred, tuple(args))
        if "++" in s:
            from dylan.formula.predicate_argument import Predicate
            from dylan.formula.ttr_infix_expression import TTRInfixExpression, split_top_level_merge

            pair = split_top_level_merge(s)
            if pair is not None:
                left, right = pair
                a = Formula.create(left, in_ex_conj)
                b = Formula.create(right, in_ex_conj)
                if a is not None and b is not None:
                    return TTRInfixExpression(Predicate("++"), a, b)
        rt = TTRRecordType.parse(s)
        if rt is not None:
            return rt
        if _ATOMIC_FORMULA_PATTERN.fullmatch(s) and not Variable.is_variable_string(s):
            from dylan.formula.atomic_formula import AtomicFormula

            return AtomicFormula(s)
        logger.debug("Formula.create could not parse: %s", string)
        return None
