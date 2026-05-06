"""Single tree-navigation operator (Java ``BasicOperator``)."""

from __future__ import annotations

import re
from dataclasses import dataclass

ARROW_UP = "/\\"
ARROW_DOWN = "\\/"

PATH_UNFIXED = "*"
PATH_LOCAL_UNFIXED = "U"
PATH_0 = "0"
PATH_1 = "1"
PATH_LINK = "L"

OP_PATTERN = re.compile(r"(/\\|\\/)([01L\*UC]*)")


@dataclass(frozen=True, slots=True)
class BasicOperator:
    """One up/down step with an optional path suffix."""

    direction: str
    path: str

    @classmethod
    def parse(cls, string: str) -> BasicOperator:
        """Parse a single operator like ``\\/0`` or ``/\\1``."""
        m = OP_PATTERN.fullmatch(string)
        if not m:
            raise ValueError(f"unrecognised operator string: {string!r}")
        return cls(m.group(1), m.group(2))

    @classmethod
    def create_many(cls, string: str) -> list[BasicOperator]:
        """Parse a sequence of operators like ``\\/0/\\1`` (Java ``BasicOperator.create``)."""
        ops: list[BasicOperator] = []
        for m in OP_PATTERN.finditer(string):
            ops.append(cls(m.group(1), m.group(2)))
        return ops

    def is_down(self) -> bool:
        """Return true for a downward operator."""
        return self.direction == ARROW_DOWN

    def is_up(self) -> bool:
        """Return true for an upward operator."""
        return self.direction == ARROW_UP

    def is_link(self) -> bool:
        """Return true for a link-path operator."""
        return self.path == "L"

    def is_fixed(self) -> bool:
        """True unless path is Kleene star or local unfixed (Java ``BasicOperator.isFixed``)."""
        return self.path not in (PATH_UNFIXED, PATH_LOCAL_UNFIXED)

    def is_star(self) -> bool:
        """Return true for an unfixed-star operator."""
        return self.path == PATH_UNFIXED

    def is_u(self) -> bool:
        """Return true for a local-unfixed operator."""
        return self.path == PATH_LOCAL_UNFIXED

    def inverse(self) -> BasicOperator:
        """Swap up/down while keeping the path suffix (Java ``BasicOperator.inverse``)."""
        inv_dir = ARROW_DOWN if self.is_up() else ARROW_UP
        return BasicOperator(inv_dir, self.path)

    def __str__(self) -> str:
        """Return Java operator syntax."""
        return self.direction + self.path


DOWN_0 = BasicOperator(ARROW_DOWN, PATH_0)

BasicOperator.isDown = BasicOperator.is_down  # type: ignore[attr-defined]
BasicOperator.isUp = BasicOperator.is_up  # type: ignore[attr-defined]
BasicOperator.isLink = BasicOperator.is_link  # type: ignore[attr-defined]
BasicOperator.isFixed = BasicOperator.is_fixed  # type: ignore[attr-defined]
BasicOperator.isStar = BasicOperator.is_star  # type: ignore[attr-defined]
BasicOperator.isU = BasicOperator.is_u  # type: ignore[attr-defined]
BasicOperator.create = BasicOperator.create_many  # type: ignore[attr-defined]
