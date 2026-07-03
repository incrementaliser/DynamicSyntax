"""Abstract tree feature (Java ``qmul.ds.learn.TreeFeature``).

Subclasses implement :meth:`extract` to pull a ``(label, node_address)`` pair
out of a Dynamic Syntax tree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree


class TreeFeature(ABC):
    """Abstract DS-tree feature extractor (Java parity)."""

    @abstractmethod
    def extract(self, tree: Tree) -> "tuple[object, NodeAddress] | None":
        """Return a ``(label, node_address)`` pair extracted from *tree* (Java ``extract``)."""
        raise NotImplementedError
