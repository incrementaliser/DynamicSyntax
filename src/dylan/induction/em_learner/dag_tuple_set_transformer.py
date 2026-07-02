"""Transformer for DAG tuple sets (Java ``qmul.ds.learn.DAGTupleSetTransformer``).

Implements the JUNG ``Transformer<DAGTupleSet, String>`` interface as a callable
class that produces a compact ``{type-labels}[size]`` rendering of the tuple set.
"""

from __future__ import annotations

from typing import Sequence

from dylan.dag.dag_tuple import DAGTuple


class DAGTupleSetTransformer:
    """Renders a DAG tuple set as ``{type-labels}[size]`` (Java parity)."""

    def transform(self, ts: "Sequence[DAGTuple]") -> str:
        """Java ``transform``: format *ts* via the pointed node's type/requirement labels."""
        if not ts:
            return ""
        first = ts[0]
        try:
            tree = first.get_tree() if hasattr(first, "get_tree") else getattr(first, "tree", None)
        except Exception:  # noqa: BLE001
            tree = None
        if tree is None:
            return f"{{}}[{len(ts)}]"
        try:
            pointed = tree.get_pointed_node()
        except Exception:  # noqa: BLE001
            return f"{{}}[{len(ts)}]"

        labels: list[str] = []
        for label in getattr(pointed, "labels", []) or []:
            text = str(label)
            cls_name = type(label).__name__.lower()
            if "type" in cls_name and "requirement" not in cls_name:
                labels.append(text)
            elif "requirement" in cls_name and "type" in text.lower():
                labels.append(text)
        body = ",".join(labels)
        return "{" + body + "}[" + str(len(ts)) + "]"

    def __call__(self, ts: "Sequence[DAGTuple]") -> str:
        """Allow instances to be invoked like Java's functional ``Transformer``."""
        return self.transform(ts)
