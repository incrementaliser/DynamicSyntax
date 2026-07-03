"""Generator feature carrying TTR or DS type info (Java ``qmul.ds.learn.Feature``)."""

from __future__ import annotations

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.type.dstype import DSType


class Feature:
    """Either a :class:`TTRRecordType` or a :class:`DSType` requirement (Java ``Feature``)."""

    def __init__(
        self,
        rt_or_dstype: "TTRRecordType | DSType",
        is_requirement: bool = False,
    ) -> None:
        """Initialise from a TTR record type or a DS type (with optional ``isRequirement`` flag)."""
        self.rt: TTRRecordType | None = None
        self.ds_type: DSType | None = None
        self.is_requirement: bool = is_requirement
        if isinstance(rt_or_dstype, TTRRecordType):
            self.rt = rt_or_dstype
        elif isinstance(rt_or_dstype, DSType):
            self.ds_type = rt_or_dstype
        else:
            raise TypeError(f"Feature requires TTRRecordType or DSType, got {type(rt_or_dstype)}")

    def get_record_type(self) -> "TTRRecordType | None":
        """Return the wrapped record type, if any (Java field accessor)."""
        return self.rt

    def get_ds_type(self) -> "DSType | None":
        """Return the wrapped DS type, if any (Java field accessor)."""
        return self.ds_type

    # ---------------- equality / hashing ----------------

    def __eq__(self, other: object) -> bool:
        """Java ``equals``: compare by RT or DSType + ``isRequirement``."""
        if not isinstance(other, Feature):
            return False
        if self.rt is not None:
            return self.rt == other.rt
        if self.ds_type is not None:
            return self.ds_type == other.ds_type and self.is_requirement == other.is_requirement
        raise ValueError("Both rt and ds_type are None on this feature")

    def __hash__(self) -> int:
        """Java ``hashCode``: hash by ``rt`` or ``str(ds_type)``."""
        if self.rt is not None:
            return hash(self.rt)
        if self.ds_type is not None:
            return hash(str(self))
        return 0

    def __str__(self) -> str:
        """Java ``toString`` -> ``rt`` text, or ``?<ds>`` for a requirement type."""
        if self.rt is not None:
            return str(self.rt)
        if self.ds_type is not None:
            return f"?{self.ds_type}" if self.is_requirement else str(self.ds_type)
        raise ValueError("Both rt and ds_type are None on this feature")

    # ---------------- comparison (Java ``compareTo``) ----------------

    def __lt__(self, other: "Feature") -> bool:
        """Java ``compareTo`` reversed: smaller (richer) RT is "greater"."""
        if self == other:
            return False
        if self.rt is not None and other.rt is not None:
            try:
                return -self.rt.compare_to(other.rt) < 0  # type: ignore[attr-defined]
            except AttributeError:
                return str(self.rt) > str(other.rt)
        if self.ds_type is not None and other.ds_type is not None:
            return str(self) < str(other)
        if self.rt is not None:
            return False  # rt > ds_type
        if self.ds_type is not None:
            return True
        raise ValueError("Either this or other feature is null")


Feature.getRecordType = Feature.get_record_type  # type: ignore[attr-defined]
Feature.getDSType = Feature.get_ds_type  # type: ignore[attr-defined]
