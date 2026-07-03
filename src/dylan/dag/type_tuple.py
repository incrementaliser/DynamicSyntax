"""Lattice node holding a record-type and accumulated increment (Java ``TypeTuple``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dylan.formula.ttr_record_type import TTRRecordType


class TypeTuple:
    """A vertex in a :class:`TypeLattice` carrying a TTR record type plus running increment."""

    def __init__(self, t: "TTRRecordType | None" = None, id_: int = 0) -> None:
        """Construct a tuple with optional initial type *t* and integer id."""
        from dylan.formula.ttr_record_type import TTRRecordType

        self.id: int = id_
        self.type: TTRRecordType = t if t is not None else TTRRecordType()
        self.increment_so_far: TTRRecordType = TTRRecordType()

    @staticmethod
    def get_new_tuple(*args: object) -> "TypeTuple":
        """Mirror Java overloads ``TypeTuple.getNewTuple(idPool)``, ``(rt, idPool)``, ``(dest, idPool)``."""
        from dylan.formula.ttr_record_type import TTRRecordType

        if len(args) == 1 and isinstance(args[0], list):
            id_pool = args[0]
            new_id = len(id_pool) + 1
            tt = TypeTuple(TTRRecordType(), new_id)
            id_pool.append(new_id)
            return tt
        if len(args) == 2 and isinstance(args[0], TTRRecordType):
            rt, id_pool = args
            assert isinstance(id_pool, list)
            new_id = len(id_pool) + 1
            tt = TypeTuple(rt, new_id)
            id_pool.append(new_id)
            return tt
        if len(args) == 2 and isinstance(args[0], TypeTuple):
            dest, id_pool = args
            assert isinstance(id_pool, list)
            new_id = len(id_pool) + 1
            tt = TypeTuple(dest.type, new_id)
            tt.increment_so_far = dest.increment_so_far
            id_pool.append(new_id)
            return tt
        raise TypeError(f"TypeTuple.get_new_tuple: bad args {args!r}")

    def get_type(self) -> "TTRRecordType":
        """Return this tuple's record type (Java ``getType``)."""
        return self.type

    def set_type(self, rt: "TTRRecordType") -> None:
        """Replace this tuple's type (Java ``setType``)."""
        self.type = rt

    def get_increment_so_far(self) -> "TTRRecordType":
        """Return the increment accumulated up to this tuple (Java ``getIncrementSoFar``)."""
        return self.increment_so_far

    def two_way_subsumes(self, other: "TypeTuple") -> bool:
        """Symmetric subsumption check (Java ``twoWaySubsumes``)."""
        return other.type.subsumes_basic(self.type) and self.type.subsumes_basic(other.type)

    def __eq__(self, other: object) -> bool:
        """Equal when the integer id matches (Java ``equals``)."""
        return isinstance(other, TypeTuple) and self.id == other.id

    def __hash__(self) -> int:
        """Hash by id only (Java ``hashCode``)."""
        return hash(self.id) * 31 + 1

    def __str__(self) -> str:
        """Pretty form delegates to the record type (Java ``toString``)."""
        return str(self.type)


TypeTuple.getNewTuple = staticmethod(TypeTuple.get_new_tuple)  # type: ignore[method-assign]
TypeTuple.getType = TypeTuple.get_type  # type: ignore[attr-defined]
TypeTuple.setType = TypeTuple.set_type  # type: ignore[attr-defined]
TypeTuple.getIncrementSoFar = TypeTuple.get_increment_so_far  # type: ignore[attr-defined]
TypeTuple.twoWaySubsumes = TypeTuple.two_way_subsumes  # type: ignore[attr-defined]
