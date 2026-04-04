"""Typed metavariable cells with Java-style side-effecting ``equals`` (Java ``MetaElement``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

X = TypeVar("X")

_POOL: dict[str, "MetaElement[Any]"] = {}


@dataclass
class MetaElement(Generic[X]):
    """Named metavariable; comparing with ``==`` to a compatible value may bind it (Java ``MetaElement``)."""

    name: str
    cls_key: str
    value: X | None = None
    backtrack: set[str] = field(default_factory=set)
    last: X | None = None

    @staticmethod
    def get(name: str, py_cls: type) -> MetaElement[Any]:
        """Return pooled meta-element keyed by class and name (Java ``MetaElement.get``)."""
        key = f"{py_cls.__name__}:{name}"
        if key not in _POOL:
            _POOL[key] = MetaElement(name=name, cls_key=key)
        return _POOL[key]

    def get_value(self) -> X | None:
        """Return the bound value, if any."""
        return self.value

    def set_value(self, v: X | None) -> None:
        """Assign *v* directly."""
        self.value = v

    def reset(self) -> None:
        """Clear binding and backtrack state (Java ``MetaElement.reset``)."""
        self.value = None
        self.backtrack.clear()
        self.last = None

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if other is None:
            return False
        if self.value is None:
            key = str(other)
            if key in self.backtrack:
                return False
            self.value = other  # type: ignore[assignment]
        return self.value == other

    def __repr__(self) -> str:
        return f"{self.name}={self.value!s}" if self.value is not None else self.name


def reset_meta_element_pool() -> None:
    """Clear every metavariable and drop pool entries (Java ``MetaElement.resetPool``)."""
    for m in _POOL.values():
        m.reset()
    _POOL.clear()
