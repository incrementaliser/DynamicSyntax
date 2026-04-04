"""DS tree node (Java `Node`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from dylan.tree.label.labels import FormulaLabel, Label, Requirement, TypeLabel
from dylan.tree.node_address import NodeAddress

if TYPE_CHECKING:
    from dylan.formula.formula import Formula
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
        """Add *label* if not already present; return whether it was added."""
        if label in self.labels:
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

    def __repr__(self) -> str:
        return f"Node({self.address!s},{self.labels!r})"
