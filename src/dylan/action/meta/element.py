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
    tried_values: set[str] = field(default_factory=set)
    last: X | None = None
    py_cls: type | None = None

    @staticmethod
    def get(name: str, py_cls: type) -> MetaElement[Any]:
        """Return pooled meta-element keyed like Java ``cls.toString() + name``."""
        key = f"{py_cls.__module__}.{py_cls.__qualname__}{name}"
        if key not in _POOL:
            _POOL[key] = MetaElement(name=name, cls_key=key, py_cls=py_cls)
        return _POOL[key]

    @staticmethod
    def get_bound_meta(py_cls: type) -> MetaElement[Any]:
        """Shared bound slot for existential ``Ex.fo(x)`` (Java ``MetaElement.getBoundMeta``)."""
        key = f"{py_cls.__module__}.{py_cls.__qualname__}:BOUND_META"
        if key not in _POOL:
            _POOL[key] = MetaElement(name="META", cls_key=key, py_cls=py_cls)
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
        self.tried_values.clear()
        self.last = None

    def backtrack(self) -> bool:
        """Unbind while remembering the value so it is not retried (Java ``MetaElement.backtrack``)."""
        if self.value is None or str(self.value) in self.tried_values:
            return False
        self.tried_values.add(str(self.value))
        self.last = self.value
        self.value = None
        return True

    def unbacktrack(self) -> None:
        """Restore the value remembered by :meth:`backtrack` (Java ``MetaElement.unbacktrack``)."""
        if self.last is not None:
            self.value = self.last
            self.tried_values.discard(str(self.last))
            self.last = None

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if other is None:
            return False
        if self.py_cls is not None and not isinstance(other, self.py_cls):
            return False
        if self.value is None:
            key = str(other)
            if key in self.tried_values:
                return False
            self.value = other  # type: ignore[assignment]
        return self.value == other

    def __repr__(self) -> str:
        return f"{self.name}={self.value!s}" if self.value is not None else self.name


def reset_bound_metas() -> None:
    """Reset only existential bound-meta cells (Java ``MetaElement.resetBoundMetas``)."""
    for m in _POOL.values():
        if m.name == "META":
            m.reset()


def reset_all_meta_bindings() -> None:
    """Clear values on every pooled meta (keys unchanged); use in tests / between utterances when needed."""
    for m in _POOL.values():
        m.reset()


def reset_meta_element_pool() -> None:
    """Clear every metavariable and drop pool entries (Java ``MetaElement.resetPool``)."""
    for m in _POOL.values():
        m.reset()
    _POOL.clear()
