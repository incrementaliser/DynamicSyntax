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
_META_MODALITY_RE = re.compile(
    r"^(?:"
    + re.escape(FORALL_LEFT)
    + "|"
    + re.escape(EXIST_LEFT)
    + r")?([A-Z][A-Z0-9]*)(?:"
    + re.escape(FORALL_RIGHT)
    + "|"
    + re.escape(EXIST_RIGHT)
    + r")?$",
)


class Modality:
    """A sequence of basic operators, optionally wrapped in ``[]`` or ``<>``."""

    def __init__(self, ops: list[BasicOperator], *, required: bool = False) -> None:
        self.ops = ops
        self.required = required

    @classmethod
    def parse(cls, string: str) -> Modality:
        """Parse a modality string like ``\\/0``, ``/\\1``, ``<\\/0/\\1>``, or ``<Z>`` metavar."""
        s = string.strip()
        m = _MODALITY_RE.fullmatch(s)
        if m:
            required = m.group(1) == FORALL_LEFT if m.group(1) else False
            ops = BasicOperator.create_many(m.group(2))
            return cls(ops, required=required)
        mm = _META_MODALITY_RE.fullmatch(s)
        if mm:
            from dylan.action.meta.meta_modality import MetaModality

            return MetaModality.get(mm.group(1))
        if re.fullmatch(r"[A-Z][A-Z0-9]*", s):
            from dylan.action.meta.meta_modality import MetaModality

            return MetaModality.get(s)
        raise ValueError(f"unrecognised modality string: {string!r}")

    def instantiate(self) -> Modality:
        """Return a fresh copy (no meta-variables in this stub)."""
        return Modality(list(self.ops), required=self.required)

    def inverse(self) -> Modality:
        """Reverse operator sequence with each step inverted (Java ``Modality.inverse``)."""
        inv_ops = [op.inverse() for op in reversed(self.ops)]
        return Modality(inv_ops, required=self.required)

    def __str__(self) -> str:
        bracket_l = FORALL_LEFT if self.required else EXIST_LEFT
        bracket_r = FORALL_RIGHT if self.required else EXIST_RIGHT
        path = "".join(str(op) for op in self.ops)
        return f"{bracket_l}{path}{bracket_r}"

    def relates(self, from_addr: "NodeAddress", to_addr: "NodeAddress") -> bool:
        """Whether *to_addr* is reachable from *from_addr* via this modality (fixed ops only)."""
        return from_addr.modality_path_matches(to_addr, self.ops)
