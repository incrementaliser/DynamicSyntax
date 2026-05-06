"""Transformer for DAG tuple sets."""

from __future__ import annotations

from typing import Iterable


class DAGTupleSetTransformer:
    """Return compact string renderings of tuple sets."""

    def transform(self, tuple_set: Iterable[object]) -> str:
        """Transform *tuple_set* to display text."""
        return "\n".join(str(item) for item in tuple_set)
