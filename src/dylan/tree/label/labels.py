"""DS tree labels (partial ``qmul.ds.tree.label``).

Recognised label types are parsed into proper subclasses; everything
else falls through to :class:`GenericLabel` so the grammar can load
without crashing.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from dylan.formula.formula import Formula
from dylan.formula.opaque_formula import OpaqueFormula
from dylan.tree.basic_operator import OP_PATTERN
from dylan.tree.modality import Modality
from dylan.tree.node_address import NodeAddress
from dylan.type.dstype import DSType

logger = logging.getLogger(__name__)

_UNARY_PRED_RE = re.compile(r"(?i)^(Tense|Class|person|Accept)\((.+)\)\s*$")
# Note: do not use a repeated *capturing* group for operators — in Python only the
# last repetition is stored; Java's regex differs.  We slice by the closing bracket.


# ── abstract base ────────────────────────────────────────────────────


class Label(ABC):
    """Node label in a DS tree."""

    def reset_metas(self) -> None:
        """Reset any bound meta-variables (no-op in base)."""
        return

    def instantiate(self) -> Label:
        """Fresh copy with metavariables resolved (Java ``Label.instantiate``)."""
        return self

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

    def instantiate(self) -> Label:
        return TypeLabel(self.type.instantiate())

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


class UnaryPredicateLabel(Label):
    """Unary DS predicates ``person(s3)``, ``class(obj)``, etc. (Java ``UnaryPredicateLabel``)."""

    def __init__(self, predicate: str, arg: str) -> None:
        super().__init__()
        self.predicate = predicate
        self.arg = arg

    @classmethod
    def parse(cls, s: str) -> UnaryPredicateLabel | None:
        """Parse *s* or return ``None`` if it is not a unary predicate."""
        m = _UNARY_PRED_RE.match(s.strip())
        if not m:
            return None
        raw = m.group(1)
        pred = raw[0].upper() + raw[1:].lower()
        return cls(pred, m.group(2).strip())

    def __hash__(self) -> int:
        return hash((self.predicate.lower(), self.arg))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, UnaryPredicateLabel)
            and self.predicate.lower() == other.predicate.lower()
            and self.arg == other.arg
        )

    def __str__(self) -> str:
        return f"{self.predicate}({self.arg})"


class FeatureLabel(Label):
    """Feature such as ``+Q``, ``+eval``, ``+BE`` (Java ``FeatureLabel``)."""

    PREFIX = "+"

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __hash__(self) -> int:
        return hash(("+", self.name))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FeatureLabel) and self.name == other.name

    def check(self, node: Any) -> bool:
        for lab in node.labels:
            if isinstance(lab, FeatureLabel) and lab.name == self.name:
                return True
        return False

    def __str__(self) -> str:
        return f"{self.PREFIX}{self.name}"


class SpeechActLabel(Label):
    """Speech-act annotation on a node (Java ``SpeechActLabel``; minimal stub)."""

    def __init__(self, name: str = "sa") -> None:
        super().__init__()
        self.name = name

    def __hash__(self) -> int:
        return hash(("SpeechAct", self.name))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SpeechActLabel) and self.name == other.name

    def __str__(self) -> str:
        return f"SA({self.name})"


class AssertionLabel(Label):
    """Assertion marker removed by ``unassert`` (Java ``AssertionLabel``; minimal stub)."""

    def __hash__(self) -> int:
        return hash("Assert")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AssertionLabel)

    def __str__(self) -> str:
        return "Assert"


class ScopeStatement(Label):
    """Scope dependency statement ``sat_scope_dep`` (Java ``ScopeStatement``; minimal stub)."""

    def __init__(self, wide: Formula, narrow: Formula) -> None:
        super().__init__()
        self.wide = wide
        self.narrow = narrow

    def get_widest(self) -> Formula:
        return self.wide

    def get_narrowest(self) -> Formula:
        return self.narrow

    def instantiate(self) -> Label:
        return ScopeStatement(self.wide.instantiate(), self.narrow.instantiate())

    def __hash__(self) -> int:
        return hash((ScopeStatement, self.wide, self.narrow))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ScopeStatement) and self.wide == other.wide and self.narrow == other.narrow
        )

    def __str__(self) -> str:
        return f"Scope({self.wide},{self.narrow})"


class BottomLabel(Label):
    """Bottom / done marker ``!`` (Java ``BottomLabel``)."""

    def __hash__(self) -> int:
        return hash("!")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BottomLabel)

    def __str__(self) -> str:
        return "!"


class FormulaLabel(Label):
    """Formula label ``Fo(…)`` wrapping a :class:`Formula` (Java ``FormulaLabel``)."""

    FUNCTOR = "Fo"

    def __init__(self, formula: Formula) -> None:
        super().__init__()
        self._formula = formula

    def get_formula(self) -> Formula:
        """Return the semantic formula (Java ``FormulaLabel.getFormula``)."""
        return self._formula

    def instantiate(self) -> Label:
        return FormulaLabel(self._formula.instantiate().evaluate())

    def __hash__(self) -> int:
        return hash((FormulaLabel, self._formula))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FormulaLabel) and self._formula == other._formula

    def __str__(self) -> str:
        return f"{self.FUNCTOR}({self._formula})"


def _formula_for_fo_inner(inner: str) -> Formula:
    """Parse *inner* or wrap as :class:`OpaqueFormula` (lexicon / IF ``Fo`` specs)."""
    s = inner.strip()
    parsed = Formula.create(s)
    if parsed is not None:
        return parsed
    return OpaqueFormula(s)


class ArbitraryLabel(Label):
    """Placeholder ``x`` inside ``Ex.x`` (Java ``ArbitraryLabel``)."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ArbitraryLabel) and self.name == other.name

    def check(self, node: Any) -> bool:
        """Approximate existential witness: any label on the node (metas omitted)."""
        return len(node.labels) > 0

    def __str__(self) -> str:
        return self.name


class ExistentialLabelConjunction(Label):
    """Existential bundle ``Ex.fo(x)`` (simplified Java ``ExistentialLabelConjunction``)."""

    FUNCTOR = "Ex."

    def __init__(self, parts: list[Label]) -> None:
        super().__init__()
        self.parts = parts

    def __hash__(self) -> int:
        return hash((self.FUNCTOR, tuple(self.parts)))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ExistentialLabelConjunction) and self.parts == other.parts

    def check(self, node: Any) -> bool:
        return all(p.check(node) for p in self.parts)

    def check_with_tuple_as_context(self, tree: Any, context: Any) -> bool:
        return all(p.check_with_tuple_as_context(tree, context) for p in self.parts)

    def __str__(self) -> str:
        if len(self.parts) == 1:
            return f"{self.FUNCTOR}{self.parts[0]}"
        return self.FUNCTOR + "(" + " & ".join(str(p) for p in self.parts) + ")"


class AddresseeLabel(Label):
    """``Addressee(X)`` — uses dialogue context (Java ``AddresseeLabel``)."""

    FUNCTOR = "Addressee"

    def __init__(self, arg: str) -> None:
        super().__init__()
        self.arg = arg.strip()

    @classmethod
    def parse(cls, s: str) -> AddresseeLabel | None:
        """Parse ``Addressee(...)`` or return ``None``."""
        low = s.strip().lower()
        if not low.startswith(cls.FUNCTOR.lower() + "("):
            return None
        i = s.index("(")
        inner = s[i + 1 : s.rindex(")")].strip()
        return cls(inner)

    def __hash__(self) -> int:
        return hash((self.FUNCTOR, self.arg))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AddresseeLabel) and self.arg == other.arg

    def check_with_tuple_as_context(self, tree: Any, context: Any) -> bool:
        """True when the current utterance has an addressee matching this spec."""
        addressee = _dialogue_addressee(context)
        if addressee is None:
            return False
        if len(self.arg) == 1 and self.arg.isupper():
            return True
        return self.arg == addressee

    def __str__(self) -> str:
        return f"{self.FUNCTOR}({self.arg})"


class ModalLabel(Label):
    """Modal path + inner label(s), e.g. ``<\\/1\\/0>person(s3)`` (Java ``ModalLabel``)."""

    def __init__(self, modality: Modality, inners: list[Label]) -> None:
        super().__init__()
        self.modality = modality
        self.inners = inners

    def __hash__(self) -> int:
        return hash((tuple(self.modality.ops), tuple(self.inners)))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ModalLabel)
            and self.modality.ops == other.modality.ops
            and self.modality.required == other.modality.required
            and self.inners == other.inners
        )

    def check_with_tuple_as_context(self, tree: Any, context: Any) -> bool:
        """Holds if some node reachable via ``modality`` from the pointer satisfies all ``inners``."""
        pointed: NodeAddress = tree.pointer
        save = tree.pointer
        try:
            for addr in tree.keys():
                if not self.modality.relates(pointed, addr):
                    continue
                tree.pointer = addr
                if all(lab.check_with_tuple_as_context(tree, context) for lab in self.inners):
                    return True
            return False
        finally:
            tree.pointer = save

    def __str__(self) -> str:
        body = (
            str(self.inners[0])
            if len(self.inners) == 1
            else "(" + " & ".join(str(x) for x in self.inners) + ")"
        )
        return str(self.modality) + body


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

    def check_with_tuple_as_context(self, tree: Any, context: Any) -> bool:
        """Negate modal / contextual checks (Java ``NegatedLabel.checkWithTupleAsContext``)."""
        return not self.inner.check_with_tuple_as_context(tree, context)

    def __str__(self) -> str:
        return f"{self.PREFIX}{self.inner}"


class GenericLabel(Label):
    """Fallback for unrecognised label specs — always fails ``check()``."""

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


def _dialogue_addressee(context: Any) -> str | None:
    """Return addressee string from a :class:`~dylan.context.context.Context`, if any."""
    if context is None:
        return None
    fn = getattr(context, "get_current_addressee", None)
    if callable(fn):
        return fn()
    return None


def _parse_modal_inner(rest: str) -> list[Label]:
    """Parse label group after a modality (possibly ``(a & b)``)."""
    rest = rest.strip()
    if rest.startswith("(") and rest.endswith(")") and "&" in rest:
        inner = rest[1:-1]
        return [label_factory_create(x.strip()) for x in inner.split("&")]
    return [label_factory_create(rest)]


def _try_parse_modal_label(s: str) -> ModalLabel | None:
    """If *s* begins with a modality, return :class:`ModalLabel`; else ``None``."""
    s = s.strip()
    if s.startswith("<") and ">" in s:
        r = s.index(">")
        mod_str = s[: r + 1]
        rest = s[r + 1 :].strip()
        try:
            modality = Modality.parse(mod_str)
        except ValueError:
            return None
        if not rest:
            return None
        return ModalLabel(modality, _parse_modal_inner(rest))
    if s.startswith("[") and "]" in s:
        r = s.index("]")
        mod_str = s[: r + 1]
        rest = s[r + 1 :].strip()
        try:
            modality = Modality.parse(mod_str)
        except ValueError:
            return None
        if not rest:
            return None
        return ModalLabel(modality, _parse_modal_inner(rest))
    ops = list(OP_PATTERN.finditer(s))
    if not ops or ops[0].start() != 0:
        return None
    end = 0
    for om in ops:
        if om.start() != end:
            break
        end = om.end()
    mod_str = s[:end].strip()
    rest = s[end:].strip()
    if not rest:
        return None
    try:
        modality = Modality.parse(mod_str)
    except ValueError:
        return None
    return ModalLabel(modality, _parse_modal_inner(rest))


def label_factory_create(string: str, ite: Any = None) -> Label:  # noqa: ARG001
    """Parse label specs used in IF clauses (partial Java ``LabelFactory.create``)."""
    s = string.strip()

    if s.startswith("(") and _DISJUNCTION_SEP in s:
        return GenericLabel(s)

    if s.startswith(_NEG):
        inner_s = s[len(_NEG) :].strip()
        inner = label_factory_create(inner_s, ite)
        return NegatedLabel(inner)

    if s.startswith(Requirement.PREFIX):
        inner_s = s[len(Requirement.PREFIX) :].strip()
        inner = label_factory_create(inner_s, ite)
        return Requirement(inner)

    low = s.lower()
    if low.startswith(AddresseeLabel.FUNCTOR.lower() + "("):
        ad = AddresseeLabel.parse(s)
        if ad is not None:
            return ad

    up = UnaryPredicateLabel.parse(s)
    if up is not None:
        return up

    if low.startswith(FeatureLabel.PREFIX.lower()) and len(s) > 1:
        return FeatureLabel(s[1:].strip())

    if s == "!":
        return BottomLabel()

    if low.startswith("ex."):
        tail = s[3:].strip()
        if tail == "x":
            return ExistentialLabelConjunction([ArbitraryLabel("x")])
        return ExistentialLabelConjunction([label_factory_create(tail)])

    if low.startswith("fo("):
        inner = s[s.index("(") + 1 : s.rindex(")")]
        return FormulaLabel(_formula_for_fo_inner(inner))

    if low.startswith(FormulaLabel.FUNCTOR.lower() + "("):
        inner = s[s.index("(") + 1 : s.rindex(")")]
        return FormulaLabel(_formula_for_fo_inner(inner))

    if low.startswith(TypeLabel.FUNCTOR.lower() + "("):
        return _parse_ty(s)

    modal = _try_parse_modal_label(s)
    if modal is not None:
        return modal

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
