"""Lattice edge carrying a record-type increment label (Java ``TypeLatticeIncrement``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dylan.formula.ttr_label import TTRLabel
    from dylan.formula.ttr_record_type import TTRRecordType


class TypeLatticeIncrement:
    """Edge in a :class:`TypeLattice` carrying a TTR increment, an *incrementOn* label, and a polarity."""

    def __init__(
        self,
        increment: "TTRRecordType | None | TypeLatticeIncrement" = None,
        increment_on: "TTRLabel | None" = None,
        id_: int = 0,
    ) -> None:
        """Construct an edge from increment, label, and id; or copy from another edge."""
        if isinstance(increment, TypeLatticeIncrement):
            other = increment
            self.id: int = other.id
            self.increment = other.increment
            self.seen: bool = other.seen
            self.increment_on = other.increment_on
            self.positive: bool = other.positive
            return
        self.id = id_
        self.increment = increment
        self.seen = False
        self.increment_on = increment_on
        self.positive = True

    @staticmethod
    def get_new_edge(*args: object) -> "TypeLatticeIncrement":
        """Mirror Java ``getNewEdge(rt, label, idPool)`` and ``getNewEdge(edge, idPool)``."""
        from dylan.formula.ttr_label import TTRLabel
        from dylan.formula.ttr_record_type import TTRRecordType

        if (
            len(args) == 3
            and isinstance(args[0], TTRRecordType)
            and isinstance(args[1], TTRLabel)
            and isinstance(args[2], list)
        ):
            inc, label, id_pool = args
            new_id = len(id_pool) + 1
            edge = TypeLatticeIncrement(inc, label, new_id)
            id_pool.append(new_id)
            return edge
        if (
            len(args) == 2
            and isinstance(args[0], TypeLatticeIncrement)
            and isinstance(args[1], list)
        ):
            src, id_pool = args
            new_id = len(id_pool) + 1
            edge = TypeLatticeIncrement(src.increment, src.increment_on, new_id)
            edge.positive = src.positive
            id_pool.append(new_id)
            return edge
        raise TypeError(f"TypeLatticeIncrement.get_new_edge: bad args {args!r}")

    def set_id(self, id_: int) -> None:
        """Set the integer id (Java ``setID``)."""
        self.id = id_

    def get_increment(self) -> "TTRRecordType | None":
        """Return the increment record type carried by this edge (Java ``getIncrement``)."""
        return self.increment

    def has_been_seen(self) -> bool:
        """True after :meth:`set_seen` has been called (Java ``hasBeenSeen``)."""
        return self.seen

    def set_seen(self, b: bool) -> None:
        """Mark this edge as visited or not (Java ``setSeen``)."""
        self.seen = b

    def is_positive(self) -> bool:
        """Edges are positive when going forward; negative when going back (Java ``isPositive``)."""
        return self.positive

    def __eq__(self, other: object) -> bool:
        """Equal when ids match (Java ``equals``)."""
        return isinstance(other, TypeLatticeIncrement) and self.id == other.id

    def __hash__(self) -> int:
        """Hash by id (Java ``hashCode``)."""
        return hash(self.id)

    def __str__(self) -> str:
        """Render with sign indicating polarity (Java ``toString``)."""
        sign = "+" if self.positive else "-"
        return f"{sign}{self.increment}"


TypeLatticeIncrement.getNewEdge = staticmethod(TypeLatticeIncrement.get_new_edge)  # type: ignore[method-assign]
TypeLatticeIncrement.setID = TypeLatticeIncrement.set_id  # type: ignore[attr-defined]
TypeLatticeIncrement.getIncrement = TypeLatticeIncrement.get_increment  # type: ignore[attr-defined]
TypeLatticeIncrement.hasBeenSeen = TypeLatticeIncrement.has_been_seen  # type: ignore[attr-defined]
TypeLatticeIncrement.setSeen = TypeLatticeIncrement.set_seen  # type: ignore[attr-defined]
TypeLatticeIncrement.isPositive = TypeLatticeIncrement.is_positive  # type: ignore[attr-defined]
