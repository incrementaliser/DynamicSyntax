"""Tree modality — sequence of ``BasicOperator`` steps (Java ``Modality``)."""

from __future__ import annotations

import re

from dylan.tree.basic_operator import ARROW_DOWN, ARROW_UP, OP_PATTERN, BasicOperator

FORALL_LEFT = "["
FORALL_RIGHT = "]"
EXIST_LEFT = "<"
EXIST_RIGHT = ">"

_MODALITY_RE = re.compile(
    r"("
    + re.escape(FORALL_LEFT) + "|" + re.escape(EXIST_LEFT)
    + r")?((?:" + OP_PATTERN.pattern + r")+)("
    + re.escape(FORALL_RIGHT) + "|" + re.escape(EXIST_RIGHT)
    + r")?",
)


class Modality:
    """A sequence of basic operators, optionally wrapped in ``[]`` or ``<>``."""

    def __init__(self, ops: list[BasicOperator], *, required: bool = False) -> None:
        self.ops = ops
        self.required = required

    @classmethod
    def parse(cls, string: str) -> Modality:
        """Parse a modality string like ``\\/0``, ``/\\1``, ``<\\/0/\\1>``."""
        m = _MODALITY_RE.fullmatch(string.strip())
        if m:
            required = m.group(1) == FORALL_LEFT if m.group(1) else False
            ops = BasicOperator.create_many(m.group(2))
            return cls(ops, required=required)
        raise ValueError(f"unrecognised modality string: {string!r}")

    def instantiate(self) -> Modality:
        """Return a fresh copy (no meta-variables in this stub)."""
        return Modality(list(self.ops), required=self.required)

    def __str__(self) -> str:
        return "".join(str(op) for op in self.ops)
