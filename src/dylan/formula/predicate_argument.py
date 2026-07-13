"""Predicate–argument formulae (Java `Predicate`, `PredicateArgumentFormula`)."""

from __future__ import annotations

from dataclasses import dataclass

from dylan.formula.formula import Formula


@dataclass(frozen=True, slots=True)
class Predicate:
    """Predicate name such as ``like`` or ``obj``."""

    name: str

    def __str__(self) -> str:
        """Surface form for infix / debug (not dataclass repr)."""
        return self.name

    def java_hash_code(self) -> int:
        """Java ``AtomicFormula.hashCode`` (``Predicate`` extends it): ``31 + name.hashCode()``."""
        from dylan.tree.label.labels import _java_int_add, java_string_hashcode

        return _java_int_add(31, java_string_hashcode(self.name))


def _subsumes_mapped_lists(l1: list[Formula], l2: list[Formula], map_: dict) -> bool:
    """Ordered pairwise mapped subsumption (Java ``Formula.subsumesMapped(List, List, map)``)."""
    if len(l1) != len(l2):
        return False
    if not l1:
        return True
    if l1[0].subsumes_mapped(l2[0], map_):
        return _subsumes_mapped_lists(l1[1:], l2[1:], map_)
    return False


@dataclass
class PredicateArgumentFormula(Formula):
    """Formula ``pred(arg1, …)``."""

    predicate: Predicate
    arguments: tuple[Formula, ...]

    def __post_init__(self) -> None:
        super().__init__()

    def clone(self) -> Formula:
        return PredicateArgumentFormula(
            self.predicate,
            tuple(a.clone() for a in self.arguments),
        )

    def instantiate(self) -> Formula:
        return PredicateArgumentFormula(
            self.predicate,
            tuple(a.instantiate() for a in self.arguments),
        )

    def evaluate(self) -> Formula:
        """Evaluate arguments, keeping relative paths unevaluated (Java ``PredicateArgumentFormula.evaluate``)."""
        from dylan.formula.ttr_path import TTRRelativePath

        new_args: list[Formula] = []
        for a in self.arguments:
            if isinstance(a, TTRRelativePath):
                new_args.append(a.clone())
            else:
                ev = a.evaluate()
                new_args.append(a.clone() if ev is None else ev)
        return PredicateArgumentFormula(self.predicate, tuple(new_args))

    def substitute(self, var: Formula, arg: Formula) -> Formula:
        """Substitute *var* with *arg* in every argument (Java ``substitute``)."""
        from dylan.formula.ttr_label import TTRLabel
        from dylan.formula.variable import Variable as _V

        if self == var:
            return arg
        sub_arg = _V(arg.name) if type(arg) is TTRLabel else arg  # noqa: E721
        return PredicateArgumentFormula(
            self.predicate,
            tuple(x.substitute(var, sub_arg) for x in self.arguments),
        )

    def conjoin(self, other: Formula) -> Formula:
        """Predicate-argument formulae are atomic and not conjoinable."""
        raise TypeError(f"Cannot conjoin PredicateArgumentFormula with {type(other).__name__}")

    def get_variables(self) -> set["Variable"]:
        """Collect variables across every argument (Java ``getVariables``)."""
        out: set = set()
        for a in self.arguments:
            out |= a.get_variables() if hasattr(a, "get_variables") else set()
        return out

    def get_ttr_paths(self) -> list[Formula]:
        """Collect TTR paths across every argument (Java ``getTTRPaths``)."""
        out: list[Formula] = []
        for a in self.arguments:
            out.extend(a.get_ttr_paths())
        return out

    def subsumes(self, other: object) -> bool:
        """Basic-then-mapped subsumption (Java ``Formula.subsumes``)."""
        if not isinstance(other, Formula):
            return False
        if self == other:
            return True
        if self.subsumes_basic(other):
            return True
        return self.subsumes_mapped(other, {})

    def subsumes_basic(self, other: Formula) -> bool:
        """Same predicate and pairwise-equal arguments (Java ``subsumesBasic``)."""
        if not isinstance(other, PredicateArgumentFormula):
            return False
        if len(self.arguments) != len(other.arguments):
            return False
        if self.predicate.name != other.predicate.name:
            return False
        return all(sa == oa for sa, oa in zip(self.arguments, other.arguments, strict=True))

    def subsumes_mapped(self, other: Formula, map_: dict) -> bool:
        """Predicate match plus ordered argument subsumption under *map_* (Java ``subsumesMapped``)."""
        if not isinstance(other, PredicateArgumentFormula):
            return False
        if self.predicate.name != other.predicate.name or len(self.arguments) != len(other.arguments):
            return False
        return _subsumes_mapped_lists(list(self.arguments), list(other.arguments), map_)

    def java_hash_code(self) -> int:
        """Java ``PredicateArgumentFormula.hashCode``: string-hash then fold args list and predicate."""
        from dylan.tree.label.labels import _java_int_add

        def _mul31(x: int) -> int:
            r = (31 * x) & 0xFFFFFFFF
            if r >= 0x80000000:
                r -= 0x100000000
            return r

        result = super().java_hash_code()
        args_h = 1
        for arg in self.arguments:
            ah = 0 if arg is None else arg.java_hash_code()
            args_h = _java_int_add(_mul31(args_h), ah)
        pred_h = 0 if self.predicate is None else self.predicate.java_hash_code()
        result = _java_int_add(_mul31(result), args_h)
        result = _java_int_add(_mul31(result), pred_h)
        return result

    def __str__(self) -> str:
        """Render ``pred(arg1, arg2)`` with Java's ``", "`` separator."""
        if not self.arguments:
            return self.predicate.name
        inner = ", ".join(str(a) for a in self.arguments)
        return f"{self.predicate.name}({inner})"
