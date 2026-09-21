"""TTR infix ``++`` / ``||`` (Java ``TTRInfixExpression``)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from dylan.formula.formula import Formula
from dylan.formula.predicate_argument import Predicate
from dylan.formula.ttr_formula import TTRFormula
from dylan.formula.ttr_path import TTRPath
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.formula.variable import Variable

logger = logging.getLogger(__name__)

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

    def subsumes_mapped(self, other: Formula, map_: dict) -> bool:
        """Functor match plus mapped args (Java ``TTRInfixExpression.subsumesMapped``)."""
        if not isinstance(other, TTRInfixExpression):
            return False
        if self.functor != other.functor and self.functor.name != other.functor.name:
            return False
        return self.arg1.subsumes_mapped(other.arg1, map_) and self.arg2.subsumes_mapped(
            other.arg2, map_
        )

    def get_types(self) -> list[TTRRecordType]:
        """Collect record-type cores under this infix (Java ``TTRInfixExpression.getTypes``)."""
        result: list[TTRRecordType] = []
        a1, a2 = self.arg1, self.arg2
        if isinstance(a1, Variable) and isinstance(a2, TTRFormula) and hasattr(a2, "get_types"):
            result.extend(a2.get_types())
        elif isinstance(a2, Variable) and isinstance(a1, TTRFormula) and hasattr(a1, "get_types"):
            result.extend(a1.get_types())
        elif isinstance(a1, TTRFormula) and isinstance(a2, TTRFormula):
            if hasattr(a1, "get_types"):
                result.extend(a1.get_types())
            if hasattr(a2, "get_types"):
                result.extend(a2.get_types())
        return result

    def get_abstractions_basic(self, basic: Any, new_var_suffix: int = 1) -> list[Any]:
        """Peel the single ``++`` core then rewrap (Java ``TTRInfixExpression.getAbstractions``).

        Assumes a single record-type core (``Variable ++ RT``). Each core
        abstraction mutates a clone of this infix and wraps ``R{suffix} ++ infix``.
        """
        from dylan.formula.ttr_lambda import TTRLambdaAbstract

        types = self.get_types()
        if len(types) > 1:
            raise NotImplementedError(
                "get_abstractions_basic on infix with more than one record-type core",
            )
        if not types:
            return []
        core = types[0]
        core_abstractions = core.get_abstractions_basic(basic, new_var_suffix)
        result: list[Any] = []
        for argument, abs_lambda in core_abstractions:
            if not isinstance(abs_lambda, TTRLambdaAbstract):
                continue
            infix = self.clone()
            assert isinstance(infix, TTRInfixExpression)
            new_cores = infix.get_types()
            if not new_cores:
                continue
            new_core = new_cores[0]
            abs_core = abs_lambda.get_core()
            if isinstance(abs_core, TTRRecordType):
                new_core.replace_content(abs_core)
            elif hasattr(abs_core, "get_types"):
                abs_types = abs_core.get_types()
                if not abs_types:
                    continue
                new_core.replace_content(abs_types[0])
            else:
                logger.warning("infix abstraction: unexpected core type %s", type(abs_core))
                continue
            binder = Variable(f"R{new_var_suffix}")
            core_final = TTRInfixExpression(_ASYM, binder, infix)
            lambda_abs = TTRLambdaAbstract(binder, core_final)
            result.append((argument, lambda_abs))
        return result

    def __str__(self) -> str:
        """Render parenthesised infix form."""
        return f"({self.arg1} {self.functor} {self.arg2})"

    def java_hash_code(self) -> int:
        """Java ``TTRInfixExpression.hashCode`` (super string-hash then fold args/functor)."""
        from dylan.tree.label.labels import _java_int_add, java_string_hashcode

        def _mul31(x: int) -> int:
            r = (31 * x) & 0xFFFFFFFF
            if r >= 0x80000000:
                r -= 0x100000000
            return r

        result = super().java_hash_code()
        a1 = 0 if self.arg1 is None else self.arg1.java_hash_code()
        a2 = 0 if self.arg2 is None else self.arg2.java_hash_code()
        pred = 0 if self.functor is None else (
            self.functor.java_hash_code()
            if hasattr(self.functor, "java_hash_code")
            else java_string_hashcode(str(self.functor))
        )
        result = _java_int_add(_mul31(result), a1)
        result = _java_int_add(_mul31(result), a2)
        result = _java_int_add(_mul31(result), pred)
        return result


TTRInfixExpression.getArg1 = TTRInfixExpression.get_arg1  # type: ignore[attr-defined]
TTRInfixExpression.getArg2 = TTRInfixExpression.get_arg2  # type: ignore[attr-defined]
TTRInfixExpression.subsumesMapped = TTRInfixExpression.subsumes_mapped  # type: ignore[attr-defined]
TTRInfixExpression.getTypes = TTRInfixExpression.get_types  # type: ignore[attr-defined]


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
