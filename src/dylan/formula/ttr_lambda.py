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

    def freshen_vars(self, tree: Any) -> TTRFormula:
        """Re-freshen inner TTR body (Java ``TTRLambdaAbstract.freshenVars``)."""
        return TTRLambdaAbstract(self.variable, self.body.freshen_vars(tree))

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

    def __str__(self) -> str:
        return f"{self.variable}^{self.body}"
