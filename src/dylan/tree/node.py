"""DS tree node (Java `Node`)."""

from __future__ import annotations

from dylan.tree.label.labels import Label
from dylan.tree.node_address import NodeAddress


class Node:
    """Set of labels at one tree position."""

    def __init__(self, address: NodeAddress) -> None:
        self.address = address
        self.labels: list[Label] = []

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

    def __repr__(self) -> str:
        return f"Node({self.address!s},{self.labels!r})"
