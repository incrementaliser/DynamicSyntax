"""TTR infix ``++`` / ``||`` (Java ``TTRInfixExpression``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dylan.formula.formula import Formula
from dylan.formula.predicate_argument import Predicate
from dylan.formula.ttr_formula import TTRFormula
from dylan.formula.ttr_path import TTRPath
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.formula.variable import Variable

_ASYM = Predicate("++")
ASYM_MERGE_FUNCTOR = _ASYM


@dataclass
class TTRInfixExpression(TTRFormula):
    """Binary TTR combine; ``++`` evaluates to :meth:`TTRRecordType.asymmetric_merge` when possible."""

    functor: Predicate
    arg1: Formula
    arg2: Formula

    ASYM_MERGE_FUNCTOR = _ASYM

    def __post_init__(self) -> None:
        """Initialise abstraction cache via :class:`TTRFormula`."""
        super().__init__()

    def get_arg1(self) -> Formula:
        """Return the left argument (Java ``TTRInfixExpression.getArg1``)."""
        return self.arg1

    def get_arg2(self) -> Formula:
        """Return the right argument (Java ``TTRInfixExpression.getArg2``)."""
        return self.arg2

    def clone(self) -> TTRFormula:
        return TTRInfixExpression(self.functor, self.arg1.clone(), self.arg2.clone())

    def instantiate(self) -> TTRFormula:
        return TTRInfixExpression(
            self.functor, self.arg1.instantiate(), self.arg2.instantiate(),
        )

    def evaluate(self) -> TTRFormula:
        """Reduce ``++`` on concrete record types (Java ``TTRInfixExpression.evaluate``)."""
        e1 = self.arg1.evaluate()
        e2 = self.arg2.evaluate()
        if (
            isinstance(e1, (TTRInfixExpression, Variable, TTRPath))
            or isinstance(e2, (TTRInfixExpression, Variable, TTRPath))
        ):
            return TTRInfixExpression(self.functor, e1, e2)
        if not isinstance(e1, TTRFormula) or not isinstance(e2, TTRFormula):
            return TTRInfixExpression(self.functor, e1, e2)
        if self.functor.name == _ASYM.name:
            return e1.asymmetric_merge(e2)
        return TTRInfixExpression(self.functor, e1, e2)

    def freshen_vars_tree(self, tree: Any) -> TTRFormula:
        """Re-freshen both arguments when they support it."""
        a1 = self.arg1.freshen_vars_tree(tree) if hasattr(self.arg1, "freshen_vars_tree") else self.arg1.clone()
        a2 = self.arg2.freshen_vars_tree(tree) if hasattr(self.arg2, "freshen_vars_tree") else self.arg2.clone()
        return TTRInfixExpression(self.functor, a1, a2)

    def freshen_vars_mapped(self, gold: Any, var_map: dict[Any, Any]) -> TTRFormula:
        """Re-freshen both arguments relative to gold record *gold*."""
        a1 = (
            self.arg1.freshen_vars_mapped(gold, var_map)
            if hasattr(self.arg1, "freshen_vars_mapped")
            else self.arg1.clone()
        )
        a2 = (
            self.arg2.freshen_vars_mapped(gold, var_map)
            if hasattr(self.arg2, "freshen_vars_mapped")
            else self.arg2.clone()
        )
        return TTRInfixExpression(self.functor, a1, a2)

    def substitute(self, var: Variable, arg: Formula) -> TTRFormula:
        return TTRInfixExpression(
            self.functor,
            self.arg1.substitute(var, arg),
            self.arg2.substitute(var, arg),
        )

    def asymmetric_merge(self, ttrf: TTRFormula) -> TTRFormula:
        """Build ``++`` infix then rely on :meth:`evaluate` (Java ``TTRInfixExpression.asymmetricMerge``)."""
        from dylan.formula.ttr_lambda import TTRLambdaAbstract

        if isinstance(ttrf, TTRLambdaAbstract):
            la = ttrf
            return la.replace_core(self.asymmetric_merge(la.get_core()))
        return TTRInfixExpression(_ASYM, self, ttrf).evaluate()

    def get_variables(self) -> set:
        """Union of variables from both arguments (Java ``getVariables``)."""
        out: set = set()
        if hasattr(self.arg1, "get_variables"):
            out |= self.arg1.get_variables()
        if hasattr(self.arg2, "get_variables"):
            out |= self.arg2.get_variables()
        return out

    def __str__(self) -> str:
        """Render parenthesised infix form."""
        return f"({self.arg1} {self.functor} {self.arg2})"


TTRInfixExpression.getArg1 = TTRInfixExpression.get_arg1  # type: ignore[attr-defined]
TTRInfixExpression.getArg2 = TTRInfixExpression.get_arg2  # type: ignore[attr-defined]


def split_top_level_merge(s: str) -> tuple[str, str] | None:
    """Split *s* on the first top-level `` ++ ``, or return ``None``."""
    depth_paren = 0
    depth_brack = 0
    i = 0
    sep = " ++ "
    while i < len(s):
        c = s[i]
        if c == "(":
            depth_paren += 1
        elif c == ")":
            depth_paren -= 1
        elif c == "[":
            depth_brack += 1
        elif c == "]":
            depth_brack -= 1
        elif depth_paren == 0 and depth_brack == 0 and s.startswith(sep, i):
            return s[:i].strip(), s[i + len(sep) :].strip()
        i += 1
    return None
