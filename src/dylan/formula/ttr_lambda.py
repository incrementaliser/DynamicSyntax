"""TTR lambda abstracts ``R^F`` (Java ``TTRLambdaAbstract``)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from dylan.formula.fol_lambda import FOLLambdaAbstract

logger = logging.getLogger(__name__)
from dylan.formula.formula import Formula
from dylan.formula.ttr_formula import TTRFormula
from dylan.formula.variable import Variable


@dataclass
class TTRLambdaAbstract(TTRFormula):
    """Lambda over a record metavar binder ``R``, ``R1``, …; body is a :class:`TTRFormula`."""

    variable: Variable
    body: TTRFormula

    def __post_init__(self) -> None:
        super().__init__()

    def clone(self) -> TTRFormula:
        return TTRLambdaAbstract(self.variable, self.body.clone())

    def instantiate(self) -> TTRFormula:
        return TTRLambdaAbstract(self.variable, self.body.instantiate())

    def evaluate(self) -> TTRFormula:
        return TTRLambdaAbstract(self.variable, self.body.evaluate())

    def freshen_vars_tree(self, tree: Any) -> TTRFormula:
        """Freshen only the core body, keeping lambda binders fixed (Java ``TTRLambdaAbstract.freshenVars``)."""
        return self.replace_core(self.get_core().freshen_vars_tree(tree))

    def freshen_vars_mapped(self, gold: Any, var_map: dict[Any, Any]) -> TTRFormula:
        """Freshen the core body relative to gold record *gold* (Java ``TTRLambdaAbstract.freshenVars(TTRRecordType, Map)``)."""
        return self.replace_core(self.get_core().freshen_vars_mapped(gold, var_map))

    def beta_reduce(self, argument: Formula) -> TTRFormula:
        """Apply *argument* and evaluate (Java ``TTRLambdaAbstract.betaReduce``)."""
        if isinstance(argument, FOLLambdaAbstract):
            raise RuntimeError(f"Not allowing higher-order argument: {argument}")
        sub = self.body.substitute(self.variable, argument)
        ev = sub.evaluate()
        if not isinstance(ev, TTRFormula):
            raise RuntimeError(f"TTR beta-reduce expected TTRFormula, got {type(ev).__name__}")
        return ev

    def substitute(self, var: Variable, arg: Formula) -> TTRFormula:
        if self.variable == var:
            if not isinstance(arg, TTRFormula):
                raise TypeError(f"TTR lambda substitute expects TTRFormula, got {type(arg).__name__}")
            return arg
        nb = self.body.substitute(var, arg)
        return TTRLambdaAbstract(self.variable, nb)

    def get_core(self) -> TTRFormula:
        """Innermost non-lambda body (Java ``TTRLambdaAbstract.getCore``)."""
        if isinstance(self.body, TTRLambdaAbstract):
            return self.body.get_core()
        return self.body

    def replace_core(self, f: TTRFormula) -> TTRLambdaAbstract:
        """Replace the core body (Java ``TTRLambdaAbstract.replaceCore``)."""
        if isinstance(self.body, TTRLambdaAbstract):
            return TTRLambdaAbstract(self.variable, self.body.replace_core(f))
        return TTRLambdaAbstract(self.variable, f)

    def asymmetric_merge(self, rt: TTRFormula) -> TTRFormula:
        """Merge *rt* into the core (Java ``TTRLambdaAbstract.asymmetricMerge``)."""
        if rt is None:
            logger.warning("asymmetric_merge on TTRLambdaAbstract with None")
            return self
        merged = self.get_core().asymmetric_merge(rt)
        return self.replace_core(merged)

    def get_variable(self) -> Variable:
        """Return the bound metavariable (Java ``getVariable``)."""
        return self.variable

    def get_body(self) -> TTRFormula:
        """Return the immediate body — possibly another lambda (Java ``getBody``)."""
        return self.body

    def get_variables(self) -> set[Variable]:
        """Return body variables minus the binder (Java ``getVariables``)."""
        body_vars = self.body.get_variables() if hasattr(self.body, "get_variables") else set()
        return {v for v in body_vars if v != self.variable}

    def subsumes_mapped(self, other: Formula, map_: dict) -> bool:
        """α-rename binder and body under *map_* (Java ``TTRLambdaAbstract.subsumesMapped``)."""
        if not isinstance(other, TTRLambdaAbstract):
            return False
        if self.variable.subsumes_basic(other.variable) and self.body.subsumes_basic(other.body):
            return True
        return self.variable.subsumes_mapped(other.variable, map_) and self.body.subsumes_mapped(
            other.body, map_
        )

    def get_abstractions_basic(self, basic: Any, new_var_suffix: int = 1) -> list[Any]:
        """Delegate basic abstractions to the body (Java ``TTRLambdaAbstract.getAbstractions(BasicType, int)``)."""
        return self.body.get_abstractions_basic(basic, new_var_suffix) if hasattr(
            self.body, "get_abstractions_basic",
        ) else []

    def __str__(self) -> str:
        """Render ``R^body`` in Java style."""
        return f"{self.variable}^{self.body}"

    def java_hash_code(self) -> int:
        """Java ``TTRLambdaAbstract.hashCode``: ``31``-fold of body then variable."""
        from dylan.tree.label.labels import _java_int_add

        def _mul31(x: int) -> int:
            r = (31 * x) & 0xFFFFFFFF
            if r >= 0x80000000:
                r -= 0x100000000
            return r

        result = 1
        body_h = 0 if self.body is None else self.body.java_hash_code()
        var_h = 0 if self.variable is None else self.variable.java_hash_code()
        result = _java_int_add(_mul31(result), body_h)
        result = _java_int_add(_mul31(result), var_h)
        return result


TTRLambdaAbstract.getCore = TTRLambdaAbstract.get_core  # type: ignore[attr-defined]
TTRLambdaAbstract.replaceCore = TTRLambdaAbstract.replace_core  # type: ignore[attr-defined]
TTRLambdaAbstract.getVariable = TTRLambdaAbstract.get_variable  # type: ignore[attr-defined]
TTRLambdaAbstract.getBody = TTRLambdaAbstract.get_body  # type: ignore[attr-defined]
TTRLambdaAbstract.getVariables = TTRLambdaAbstract.get_variables  # type: ignore[attr-defined]
TTRLambdaAbstract.betaReduce = TTRLambdaAbstract.beta_reduce  # type: ignore[attr-defined]
TTRLambdaAbstract.subsumesMapped = TTRLambdaAbstract.subsumes_mapped  # type: ignore[attr-defined]
