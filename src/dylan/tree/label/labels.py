"""DS tree labels (partial ``qmul.ds.tree.label``).

Recognised label types are parsed into proper subclasses; everything
else falls through to :class:`GenericLabel` so the grammar can load
without crashing.  ``GenericLabel.check()`` always returns ``False``,
meaning any computational/lexical rule whose IF clause uses an
unimplemented label will simply fail — safe, but not yet functional.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from dylan.type.dstype import DSType

logger = logging.getLogger(__name__)


# ── abstract base ────────────────────────────────────────────────────

class Label(ABC):
    """Node label in a DS tree."""

    def reset_metas(self) -> None:
        """Reset any bound meta-variables (no-op in base)."""
        return

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

    def check(self, node: Any) -> bool:
        """Return ``True`` if *node* carries this label (Java ``Node.hasLabel`` / symmetric ``equals``)."""
        for lab in node.labels:
            if self == lab or lab == self:
                return True
        return False

    def check_with_tuple_as_context(self, tree: Any, context: Any) -> bool:
        """Check against the pointed node (default; subclasses may use *context*)."""
        return self.check(tree.pointed_node)


# ── concrete label types ─────────────────────────────────────────────

class TypeLabel(Label):
    """Type label ``Ty(e)`` etc."""

    FUNCTOR = "Ty"

    def __init__(self, t: DSType) -> None:
        super().__init__()
        self.type = t

    def __hash__(self) -> int:
        return hash(self.type)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TypeLabel) and self.type == other.type

    def __str__(self) -> str:
        return f"{self.FUNCTOR}({self.type})"


TypeLabel.t = TypeLabel(DSType.t)  # type: ignore[attr-defined]
TypeLabel.e = TypeLabel(DSType.e)  # type: ignore[attr-defined]
TypeLabel.cn = TypeLabel(DSType.cn)  # type: ignore[attr-defined]


class Requirement(Label):
    """Requirement ``?X``."""

    PREFIX = "?"

    def __init__(self, inner: Label) -> None:
        super().__init__()
        self.inner = inner

    def __hash__(self) -> int:
        return hash(self.inner)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Requirement) and self.inner == other.inner

    def __str__(self) -> str:
        return f"{self.PREFIX}{self.inner}"


class NegatedLabel(Label):
    """Negation ``\u00acX`` — satisfied when the inner label is NOT present."""

    PREFIX = "\u00ac"

    def __init__(self, inner: Label) -> None:
        super().__init__()
        self.inner = inner

    def __hash__(self) -> int:
        return hash(("NOT", self.inner))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NegatedLabel) and self.inner == other.inner

    def check(self, node: Any) -> bool:
        return not self.inner.check(node)

    def __str__(self) -> str:
        return f"{self.PREFIX}{self.inner}"


class GenericLabel(Label):
    """Fallback for unrecognised label specs — always fails ``check()``.

    Allows grammars to load without crashing; the action simply won't
    fire until a proper label class is ported.
    """

    def __init__(self, spec: str) -> None:
        super().__init__()
        self.spec = spec

    def __hash__(self) -> int:
        return hash(self.spec)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GenericLabel) and self.spec == other.spec

    def check(self, node: Any) -> bool:
        return False

    def check_with_tuple_as_context(self, tree: Any, context: Any) -> bool:
        return False

    def __str__(self) -> str:
        return self.spec


# ── factory ──────────────────────────────────────────────────────────

_NEG = "\u00ac"
_DISJUNCTION_SEP = "||"


def label_factory_create(string: str, ite: Any = None) -> Label:  # noqa: ARG001
    """Parse label specs used in IF clauses (partial Java ``LabelFactory.create``).

    Recognised patterns are parsed into proper label objects; everything
    else produces a :class:`GenericLabel` so the grammar file can be read.
    """
    s = string.strip()

    if s.startswith("(") and _DISJUNCTION_SEP in s:
        return GenericLabel(s)

    if s.startswith(_NEG):
        inner_s = s[len(_NEG):].strip()
        inner = label_factory_create(inner_s, ite)
        return NegatedLabel(inner)

    if s.startswith(Requirement.PREFIX):
        inner_s = s[len(Requirement.PREFIX):].strip()
        inner = label_factory_create(inner_s, ite)
        return Requirement(inner)

    low = s.lower()
    if low.startswith(TypeLabel.FUNCTOR.lower() + "("):
        return _parse_ty(s)

    return GenericLabel(s)


def _parse_ty(s: str) -> TypeLabel:
    """Parse ``Ty(...)`` into a :class:`TypeLabel`."""
    low = s.strip().lower()
    if not low.startswith(TypeLabel.FUNCTOR.lower()):
        return GenericLabel(s)  # type: ignore[return-value]
    try:
        inner = s[s.index("(") + 1 : s.rindex(")")]
        dt = DSType.parse(inner)
        if dt is None:
            return GenericLabel(s)  # type: ignore[return-value]
        return TypeLabel(dt)
    except (ValueError, IndexError):
        return GenericLabel(s)  # type: ignore[return-value]
