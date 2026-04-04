"""First-order lambda abstracts ``x^F`` (Java ``FOLLambdaAbstract``)."""

from __future__ import annotations

from dataclasses import dataclass

from dylan.formula.formula import Formula
from dylan.formula.variable import Variable


@dataclass
class FOLLambdaAbstract(Formula):
    """Lambda ``var^body`` used with ``beta-reduce`` (Java ``FOLLambdaAbstract`` / ``LambdaAbstract``)."""

    variable: Variable
    body: Formula

    def __post_init__(self) -> None:
        super().__init__()

    def clone(self) -> Formula:
        return FOLLambdaAbstract(self.variable, self.body.clone())

    def instantiate(self) -> Formula:
        return FOLLambdaAbstract(self.variable, self.body.instantiate())

    def evaluate(self) -> Formula:
        return FOLLambdaAbstract(self.variable, self.body.evaluate())

    def beta_reduce(self, argument: Formula) -> Formula:
        """Apply *argument* to this lambda (Java ``LambdaAbstract.betaReduce``)."""
        if isinstance(argument, FOLLambdaAbstract):
            raise RuntimeError(f"Not allowing higher-order argument: {argument}")
        return self.body.substitute(self.variable, argument)

    def substitute(self, var: Variable, arg: Formula) -> Formula:
        if var == self.variable:
            return self
        return FOLLambdaAbstract(self.variable, self.body.substitute(var, arg))

    def conjoin(self, other: Formula) -> Formula:
        raise TypeError(f"Cannot conjoin FOLLambdaAbstract with {type(other).__name__}")

    def __str__(self) -> str:
        return f"{self.variable}^{self.body}"
