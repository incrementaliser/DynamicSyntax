"""Perturbation sample utility."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Generic, Iterable, TypeVar

T = TypeVar("T")


@dataclass
class PerturbationSample(Generic[T]):
    """Random sample helper used by learner experiments."""

    items: list[T] = field(default_factory=list)
    seed: int | None = None

    def sample(self, n: int) -> list[T]:
        """Return up to *n* sampled items."""
        rng = Random(self.seed)
        if n >= len(self.items):
            return list(self.items)
        return rng.sample(self.items, n)

    @classmethod
    def from_iterable(cls, items: Iterable[T], seed: int | None = None) -> PerturbationSample[T]:
        """Build a sample helper from *items*."""
        return cls(list(items), seed)
