"""DS types: basic (e, t, …) and constructed (e>t). Ported from qmul.ds.type."""

from __future__ import annotations

import logging
import re
from abc import ABC
from dataclasses import dataclass
from typing import ClassVar

logger = logging.getLogger(__name__)

TYPE_LEFT = "("
TYPE_RIGHT = ")"
TYPE_SEP = ">"
UNICODE_TYPE_SEP = "\u2192"
BASIC_TYPE_PATTERN = "e|es|cn|t|cnev"
_METAVAR_RE = re.compile(r"^[V-Z][0-9]*$")


class DSType(ABC):
    """Abstract dynamic-syntax type (Java `DSType`)."""

    e: ClassVar["BasicType"]
    t: ClassVar["BasicType"]
    cn: ClassVar["BasicType"]
    es: ClassVar["BasicType"]
    cnev: ClassVar["BasicType"]

    def instantiate(self) -> DSType:
        """Return ``self``; metavariable subclasses override (Java ``DSType.instantiate``)."""
        return self

    def java_hash_code(self) -> int:
        """Java ``DSType.hashCode`` default: ``toString().hashCode()`` (subclasses override)."""
        from dylan.tree.label.labels import java_string_hashcode

        return java_string_hashcode(str(self))

    def to_unicode_string(self) -> str:
        """Return Unicode arrow rendering of this type."""
        return str(self).replace(TYPE_SEP, UNICODE_TYPE_SEP)

    def get_final_type(self) -> DSType:
        """Return the right-most basic result type (Java ``DSType.getFinalType``)."""
        return self

    def to_unique_int(self) -> int:
        """Return a stable integer fingerprint (Java ``DSType.toUniqueInt``)."""
        return 0

    def clone(self) -> DSType:
        """Return a deep copy by re-parsing the printed form (Java ``DSType.clone``)."""
        parsed = DSType.parse(str(self))
        assert parsed is not None
        return parsed

    def get_types_subj_first(self) -> list["BasicType"]:
        """Return basic argument types subject-first (Java ``DSType.getTypesSubjFirst``); empty for abstract base."""
        return []

    @staticmethod
    def create(*args: object) -> "DSType":
        """Mirror Java ``DSType.create(String)`` and ``DSType.create(DSType, DSType)``."""
        if len(args) == 1 and isinstance(args[0], str):
            return BasicType(args[0].strip())
        if len(args) == 2 and isinstance(args[0], DSType) and isinstance(args[1], DSType):
            return ConstructedType(args[0], args[1])
        raise TypeError(f"DSType.create: bad args {args!r}")


@dataclass(frozen=True, slots=True)
class BasicType(DSType):
    """Atomic type such as ``e`` or ``t``."""

    name: str

    def __post_init__(self) -> None:
        """Strip whitespace in-place since dataclass is frozen."""
        object.__setattr__(self, "name", self.name.strip())

    def __str__(self) -> str:
        """Return the bare type name."""
        return self.name

    def java_hash_code(self) -> int:
        """Java ``BasicType.hashCode``: ``31 * 1 + type.hashCode()``."""
        from dylan.tree.label.labels import _java_int_add, java_string_hashcode

        return _java_int_add(31, java_string_hashcode(self.name))

    def __hash__(self) -> int:
        """Hash by name to match Java ``BasicType.hashCode``."""
        return hash(self.name)

    def to_unique_int(self) -> int:
        """Hash of the name (Java ``BasicType.toUniqueInt``)."""
        return hash(self.name)

    def get_types_subj_first(self) -> list["BasicType"]:
        """Singleton ``[self]`` (Java ``BasicType.getTypesSubjFirst``)."""
        return [self]


@dataclass(frozen=True, slots=True)
class ConstructedType(DSType):
    """Function type ``from>to``."""

    from_type: DSType
    to_type: DSType

    def instantiate(self) -> ConstructedType:
        """Recursively instantiate sub-types (Java ``ConstructedType.instantiate``)."""
        return ConstructedType(self.from_type.instantiate(), self.to_type.instantiate())

    def get_final_type(self) -> DSType:
        """Walk ``to`` to the right-most basic type (Java ``ConstructedType.getFinalType``)."""
        return self.to_type.get_final_type()

    def get_from(self) -> DSType:
        """Return the left/argument type (Java ``ConstructedType.getFrom``)."""
        return self.from_type

    def get_to(self) -> DSType:
        """Return the right/result type (Java ``ConstructedType.getTo``)."""
        return self.to_type

    def get_types_subj_first(self) -> list[BasicType]:
        """Subject-first argument list, e.g. ``e>(e>t)`` -> ``[t,e,e]`` (Java ``ConstructedType.getTypesSubjFirst``)."""
        out: list[BasicType] = []
        out.extend(self.to_type.get_types_subj_first())
        out.extend(self.from_type.get_types_subj_first())
        return out

    def to_unique_int(self) -> int:
        """Sum sub-type hashes (Java ``ConstructedType.toUniqueInt``)."""
        return self.from_type.to_unique_int() + self.to_type.to_unique_int()

    def __str__(self) -> str:
        """Render ``from>to`` with parentheses around constructed sub-types."""
        left = f"({self.from_type})" if isinstance(self.from_type, ConstructedType) else str(self.from_type)
        right = f"({self.to_type})" if isinstance(self.to_type, ConstructedType) else str(self.to_type)
        return f"{left}{TYPE_SEP}{right}"

    def java_hash_code(self) -> int:
        """Java ``ConstructedType.hashCode``: fold ``from`` then ``to`` with prime 31."""
        from dylan.tree.label.labels import _java_int_add

        def _mul31(x: int) -> int:
            r = (31 * x) & 0xFFFFFFFF
            if r >= 0x80000000:
                r -= 0x100000000
            return r

        result = 1
        from_h = 0 if self.from_type is None else self.from_type.java_hash_code()
        to_h = 0 if self.to_type is None else self.to_type.java_hash_code()
        result = _java_int_add(_mul31(result), from_h)
        result = _java_int_add(_mul31(result), to_h)
        return result

    def __hash__(self) -> int:
        """Hash from sub-types so that ``a>b == a>b``."""
        return hash((self.from_type, self.to_type))


DSType.e = BasicType("e")
DSType.t = BasicType("t")
DSType.cn = BasicType("cn")
DSType.es = BasicType("es")
DSType.cnev = BasicType("cnev")

DSType.et = ConstructedType(DSType.e, DSType.t)  # type: ignore[attr-defined]
DSType.eet = ConstructedType(DSType.e, DSType.et)  # type: ignore[attr-defined]


def _split_top_level(string: str) -> tuple[str, str] | None:
    n = 0
    for i, ch in enumerate(string):
        rest = string[i:]
        if rest.startswith(TYPE_LEFT):
            n += 1
        elif rest.startswith(TYPE_RIGHT):
            n -= 1
        elif rest.startswith(TYPE_SEP) and n == 0:
            return string[:i], rest[len(TYPE_SEP) :]
    if string.startswith(TYPE_LEFT) and string.endswith(TYPE_RIGHT):
        return _split_top_level(string[1:-1])
    return None


def _parse_meta_type(string: str) -> DSType | None:
    # TODO(verify): full MetaType port; parser tests use only basic/constructed types.
    if _METAVAR_RE.match(string):
        from dylan.action.meta_stub import MetaType

        return MetaType.get(string)
    return None


def DSType_parse(string: str) -> DSType | None:  # noqa: N802
    """Parse a type string like ``e``, ``e>t``, ``(e>t)>t``."""
    string = string.strip()
    if TYPE_SEP in string:
        parts = _split_top_level(string)
        if parts is None:
            return None
        left, right = parts
        if not right:
            return BasicType(left)
        lt = DSType.parse(left)
        rt = DSType.parse(right)
        if lt is None or rt is None:
            return None
        return ConstructedType(lt, rt)
    meta = _parse_meta_type(string)
    if meta is not None:
        return meta
    if re.fullmatch(BASIC_TYPE_PATTERN, string):
        logger.debug("creating basic type from %s", string)
        return BasicType(string)
    logger.debug("string was %s bad type spec", string)
    return None


DSType.parse = staticmethod(DSType_parse)  # type: ignore[method-assign]

# camelCase Java compatibility aliases
DSType.toUnicodeString = DSType.to_unicode_string  # type: ignore[attr-defined]
DSType.getFinalType = DSType.get_final_type  # type: ignore[attr-defined]
DSType.toUniqueInt = DSType.to_unique_int  # type: ignore[attr-defined]
DSType.getTypesSubjFirst = DSType.get_types_subj_first  # type: ignore[attr-defined]
ConstructedType.getFrom = ConstructedType.get_from  # type: ignore[attr-defined]
ConstructedType.getTo = ConstructedType.get_to  # type: ignore[attr-defined]
