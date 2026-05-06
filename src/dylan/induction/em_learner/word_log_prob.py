"""Display pair for a word hypothesis probability."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(order=True, frozen=True, slots=True)
class WordLogProb:
    """Word/log-probability pair sorted by log probability."""

    word: str
    log_prob: float

    def first(self) -> str:
        """Return the word."""
        return self.word

    def second(self) -> float:
        """Return the log probability."""
        return self.log_prob
