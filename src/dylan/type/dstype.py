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
        return self

    def to_unicode_string(self) -> str:
        return str(self).replace(TYPE_SEP, UNICODE_TYPE_SEP)

    def get_final_type(self) -> DSType:
        return self

    def to_unique_int(self) -> int:
        return 0

    def clone(self) -> DSType:
        parsed = DSType.parse(str(self))
        assert parsed is not None
        return parsed


@dataclass(frozen=True, slots=True)
class BasicType(DSType):
    """Atomic type such as ``e`` or ``t``."""

    name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", self.name.strip())

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def to_unique_int(self) -> int:
        return hash(self.name)


@dataclass(frozen=True, slots=True)
class ConstructedType(DSType):
    """Function type ``from>to``."""

    from_type: DSType
    to_type: DSType

    def instantiate(self) -> ConstructedType:
        return ConstructedType(self.from_type.instantiate(), self.to_type.instantiate())

    def get_final_type(self) -> DSType:
        return self.to_type.get_final_type()

    def __str__(self) -> str:
        left = f"({self.from_type})" if isinstance(self.from_type, ConstructedType) else str(self.from_type)
        right = f"({self.to_type})" if isinstance(self.to_type, ConstructedType) else str(self.to_type)
        return f"{left}{TYPE_SEP}{right}"

    def __hash__(self) -> int:
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
