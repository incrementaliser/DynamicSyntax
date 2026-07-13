"""TTR record field labels (Java `TTRLabel`)."""

from __future__ import annotations

import re

from dylan.formula.variable import Variable

# Java ``TTRLabel.LABEL_PATTERN = ([a-z]+?)(\d*)`` — lowercase letters plus digits only.
LABEL_PATTERN = re.compile(r"^[a-z]+\d*$")
# Java ``TTRLabel.META_LABEL_PATTERN = (L+?|P+?|PRED+?)(\d*)``.
META_LABEL_PATTERN = re.compile(r"^(?:L+|P+|PRED+)\d*$")


class TTRLabel(Variable):
    """Label used in TTR record fields, e.g. ``p31``, ``head`` (Java ``TTRLabel`` extends ``Variable``)."""

    def __init__(self, label: str | Variable) -> None:
        """Accept a raw string or an existing Variable/TTRLabel (Java constructors)."""
        name = label.name if isinstance(label, Variable) else str(label)
        super().__init__(name)

    @property
    def label(self) -> str:
        """Return the label text (legacy accessor mirroring the old dataclass field)."""
        return self.name

    def clone(self) -> "TTRLabel":
        """Return a copy (Java ``TTRLabel.clone``)."""
        return TTRLabel(self.name)

    def instantiate(self) -> "TTRLabel":
        """Labels instantiate to themselves (Java ``TTRLabel.instantiate``)."""
        return self

    def substitute(self, var: object, arg: object) -> "TTRLabel":
        """Return the substituted label when equal to *var* (Java ``TTRLabel.substitute``)."""
        if isinstance(var, Variable) and self == var and isinstance(arg, Variable):
            return TTRLabel(arg)
        return self

    def __str__(self) -> str:
        """Return the label text."""
        return self.name

    def __repr__(self) -> str:
        """Debug form mirroring the old dataclass repr."""
        return f"TTRLabel(label={self.name!r})"

    def subsumes_basic(self, other: object) -> bool:
        """Label basic subsumption is name equality (Java ``Variable.subsumesBasic``).

        Must not call :meth:`subsumes` here — ``Variable.subsumes`` already
        delegates to ``subsumes_basic``, and a back-call recurses forever.
        """
        return isinstance(other, Variable) and self.name == other.name

    def subsumes_mapped(self, other: object, map_: dict) -> bool:
        """Map record labels across freshened names (Java ``Variable.subsumesMapped``)."""
        from dylan.formula.variable import Variable

        if isinstance(other, TTRLabel):
            return Variable(self.label).subsumes_mapped(Variable(other.label), map_)
        return False


HEAD = TTRLabel("head")
REF_TIME = TTRLabel("reftime")


def ttr_label_from_variable(v: Variable) -> TTRLabel:
    """Build a label from a variable (Java ``new TTRLabel(Variable)``)."""
    return TTRLabel(v.name)
