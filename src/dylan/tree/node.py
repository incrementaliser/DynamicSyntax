"""DS tree node (Java `Node`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from dylan.tree.label.labels import FormulaLabel, Label, Requirement, TypeLabel
from dylan.formula.formula import Formula
from dylan.tree.node_address import NodeAddress

if TYPE_CHECKING:
    from dylan.type.dstype import DSType


class Node:
    """Set of labels at one tree position."""

    def __init__(self, address: NodeAddress, labels: list[Label] | None = None) -> None:
        self.address = address
        self.labels: list[Label] = list(labels) if labels is not None else []

    def __iter__(self) -> Iterator[Label]:
        """Iterate labels (Java ``Node`` is iterable)."""
        return iter(self.labels)

    def add_label(self, label: Label) -> bool:
        """Add *label*; replace same-class :class:`TypeLabel` / :class:`FormulaLabel` (Java ``Node.addLabel``)."""
        if isinstance(label, (TypeLabel, FormulaLabel)):
            existing = next((l for l in self.labels if type(l) is type(label)), None)
            if existing is not None:
                self.labels.remove(existing)
        elif label in self.labels:
            return False
        self.labels.append(label)
        return True

    def remove_label(self, label: Label) -> bool:
        """Remove *label* using symmetric equality (Java ``Node.remove``)."""
        for i, lab in enumerate(self.labels):
            if lab == label or label == lab:
                self.labels.pop(i)
                return True
        return False

    def contains(self, label: Label) -> bool:
        """Check if *label* is present (Java ``Node.contains``)."""
        return label in self.labels

    def is_locally_fixed(self) -> bool:
        """Delegate to address (Java ``Node.isLocallyFixed``)."""
        return self.address.is_locally_fixed()

    def merge_from(self, other: Node) -> None:
        """Add all labels from *other* (Java ``Node.merge`` / ``addAll``)."""
        for lab in other.labels:
            self.add_label(lab)

    def get_type_label(self) -> TypeLabel | None:
        """First :class:`TypeLabel` on this node, or ``None``."""
        for lab in self.labels:
            if isinstance(lab, TypeLabel):
                return lab
        return None

    def get_formula_label(self) -> FormulaLabel | None:
        """First :class:`FormulaLabel` on this node, or ``None``."""
        for lab in self.labels:
            if isinstance(lab, FormulaLabel):
                return lab
        return None

    def get_type(self) -> DSType | None:
        """Return the node's Ty label type if present (Java ``Node.getType``)."""
        tyl = self.get_type_label()
        return tyl.type if tyl is not None else None

    def get_formula(self) -> Formula | None:
        """Return the node's Fo formula if present (Java ``Node.getFormula``)."""
        fl = self.get_formula_label()
        return fl.get_formula() if fl is not None else None

    def has_type(self) -> bool:
        """True if a :class:`TypeLabel` is present (Java ``Node.hasType``)."""
        return self.get_type_label() is not None

    def get_required_type(self) -> DSType | None:
        """Type inside a top-level type requirement, if any (Java ``getRequiredType``)."""
        for lab in self.labels:
            if isinstance(lab, Requirement) and isinstance(lab.inner, TypeLabel):
                return lab.inner.type
        return None

    def get_type_requirement(self) -> Requirement | None:
        """First type :class:`Requirement` on this node (Java ``getTypeRequirement``)."""
        for lab in self.labels:
            if isinstance(lab, Requirement) and isinstance(lab.inner, TypeLabel):
                return lab
        return None

    def is_complete(self) -> bool:
        """Return whether this node has no outstanding :class:`Requirement` labels (Java ``isComplete``)."""
        return not any(isinstance(lab, Requirement) for lab in self.labels)

    def get_required_formula(self) -> Formula | None:
        """Formula inside ``?Fo(…)`` requirement, if any (Java ``getRequiredFormula``)."""
        for lab in self.labels:
            if isinstance(lab, Requirement) and isinstance(lab.inner, FormulaLabel):
                return lab.inner.get_formula()
        return None

    def remove_formula_label(self) -> None:
        """Drop the first :class:`FormulaLabel` (Java ``removeFormulaLabel``)."""
        fl = self.get_formula_label()
        if fl is not None:
            self.remove_label(fl)

    def is_unifiable(self, other: Node) -> bool:
        """Address-compatible merge target for an unfixed node (Java ``Node.isUnifiable``)."""
        if not other.address.subsumes(self.address):
            return False
        t = self.get_type() or self.get_required_type()
        if t is not None:
            ot = other.get_type() or other.get_required_type()
            if ot is not None and t != ot:
                return False
        f = self.get_formula() or self.get_required_formula()
        if f is not None:
            of = other.get_formula() or other.get_required_formula()
            if of is not None and not f.subsumes(of):
                return False
        return True

    def _labels_equal_set(self, other: "Node") -> bool:
        """Whether label multisets match (Java ``Node`` extends ``TreeSet``)."""
        if len(self.labels) != len(other.labels):
            return False
        remaining = list(other.labels)
        for lab in self.labels:
            for i, olab in enumerate(remaining):
                if lab == olab or olab == lab:
                    remaining.pop(i)
                    break
            else:
                return False
        return True

    def __eq__(self, other: object) -> bool:
        """Match Java ``Node.equals``: address plus label-set equality."""
        if self is other:
            return True
        if not isinstance(other, Node):
            return NotImplemented
        return self.address == other.address and self._labels_equal_set(other)

    def __hash__(self) -> int:
        """Hash consistent with Java ``Node.hashCode`` (address + label set)."""
        label_key = frozenset(
            (type(lab), lab) for lab in self.labels
        )  # best-effort; labels should implement stable equality
        return hash((self.address, label_key))

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"Node({self.address!s},{self.labels!r})"


Node.addLabel = Node.add_label  # type: ignore[attr-defined]
Node.removeLabel = Node.remove_label  # type: ignore[attr-defined]
Node.isLocallyFixed = Node.is_locally_fixed  # type: ignore[attr-defined]
Node.mergeFrom = Node.merge_from  # type: ignore[attr-defined]
Node.getTypeLabel = Node.get_type_label  # type: ignore[attr-defined]
Node.getFormulaLabel = Node.get_formula_label  # type: ignore[attr-defined]
Node.getType = Node.get_type  # type: ignore[attr-defined]
Node.getFormula = Node.get_formula  # type: ignore[attr-defined]
Node.hasType = Node.has_type  # type: ignore[attr-defined]
Node.getRequiredType = Node.get_required_type  # type: ignore[attr-defined]
Node.getRequiredFormula = Node.get_required_formula  # type: ignore[attr-defined]
Node.getTypeRequirement = Node.get_type_requirement  # type: ignore[attr-defined]
Node.isComplete = Node.is_complete  # type: ignore[attr-defined]
Node.removeFormulaLabel = Node.remove_formula_label  # type: ignore[attr-defined]
Node.isUnifiable = Node.is_unifiable  # type: ignore[attr-defined]
