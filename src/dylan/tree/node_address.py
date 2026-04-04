"""010*-style node addresses (Java ``NodeAddress``)."""

from __future__ import annotations

from dataclasses import dataclass

PATH_0 = "0"
PATH_1 = "1"
PATH_LINK = "L"
PATH_UNFIXED = "*"
PATH_LOCAL_UNFIXED = "U"
PATH_CONTEXT = "C"
ROOT = "0"


@dataclass(frozen=True, slots=True)
class NodeAddress:
    """String DS address (default root ``0``).

    Addresses are strings over ``{0,1,L,*,U,C}`` rooted at ``"0"``.
    Down-0 from ``"0"`` yields ``"00"``; down-1 yields ``"01"``, etc.
    """

    address: str = ROOT

    def is_root(self) -> bool:
        return self.address == ROOT

    def down(self, path: str) -> NodeAddress:
        """Go down by appending *path* (Java ``NodeAddress.down``)."""
        return NodeAddress(self.address + path)

    def down0(self) -> NodeAddress:
        return self.down(PATH_0)

    def down1(self) -> NodeAddress:
        return self.down(PATH_1)

    def down_link(self) -> NodeAddress:
        return self.down(PATH_LINK)

    def down_star(self) -> NodeAddress:
        return self.down(PATH_UNFIXED)

    def down_local_unfixed(self) -> NodeAddress:
        """Append local-unfixed step (Java ``NodeAddress.downLocalUnfixed``)."""
        return self.down(PATH_LOCAL_UNFIXED)

    def down_char(self, ch: str) -> NodeAddress:
        """Append a single path character ``0``, ``1``, ``L``, ``*``, or ``U`` (Java ``down(String)``)."""
        return self.down(ch)

    def is_locally_fixed(self) -> bool:
        """False when the address ends in Kleene star or local-unfixed (Java ``isLocallyFixed``)."""
        a = self.address
        return not (a.endswith(PATH_UNFIXED) or a.endswith(PATH_LOCAL_UNFIXED))

    def up(self, path: str | None = None) -> NodeAddress | None:
        """Go up (Java ``NodeAddress.up``).

        Without *path*: strip last character (plain parent).
        With *path*: strip that suffix only if the address ends with it.
        Returns ``None`` if the operation is invalid.
        """
        if path is None:
            if len(self.address) < 2:
                return None
            return NodeAddress(self.address[:-1])
        if not self.address.endswith(path):
            return None
        i = self.address.rfind(path)
        if i < 0:
            return None
        return NodeAddress(self.address[:i])

    def modality_path_matches(self, other: NodeAddress, ops: list["BasicOperator"]) -> bool:
        """Whether *other* is reachable from ``self`` following *ops* (fixed operators only; Java ``NodeAddress.to``)."""
        if not ops:
            return self.address == other.address
        op, *rest = ops
        if not op.is_fixed():
            return False
        nxt = self.go_op(op)
        if nxt is None:
            return False
        return nxt.modality_path_matches(other, rest)

    def go_op(self, op: "BasicOperator") -> NodeAddress | None:
        """Navigate one ``BasicOperator`` step (Java ``NodeAddress.go(BasicOperator)``)."""
        if op.is_down():
            if not op.path:
                raise RuntimeError("must specify down path")
            return self.down(op.path)
        if op.is_up():
            if not op.path:
                return self.up()
            return self.up(op.path)
        return None

    def go_modality(self, mod: "Modality") -> NodeAddress | None:
        """Navigate a full ``Modality`` path (Java ``NodeAddress.go(Modality)``)."""
        na: NodeAddress | None = self
        for op in mod.ops:
            if na is None:
                return None
            na = na.go_op(op)
        return na

    def __str__(self) -> str:
        return self.address


# Late imports to avoid circular dependency
from dylan.tree.basic_operator import BasicOperator  # noqa: E402
from dylan.tree.modality import Modality  # noqa: E402

__all__ = ["NodeAddress"]
