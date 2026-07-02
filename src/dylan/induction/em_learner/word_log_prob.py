"""Word/log-probability pair (Java ``qmul.ds.learn.WordLogProb``)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WordLogProb:
    """Java ``WordLogProb`` (extends ``Pair<String, Double>``); compares descending by ``prob``."""

    word: str
    prob: float

    def get_word(self) -> str:
        """Return the word string (Java ``getWord``)."""
        return self.word

    def get_prob(self) -> float:
        """Return the probability (Java ``getProb``)."""
        return self.prob

    def first(self) -> str:
        """Return the first element of the underlying pair (Java ``first``)."""
        return self.word

    def second(self) -> float:
        """Return the second element of the underlying pair (Java ``second``)."""
        return self.prob

    def __lt__(self, other: "WordLogProb") -> bool:
        """Java ``compareTo``: descending probability ordering."""
        return self.prob > other.prob

    def __le__(self, other: "WordLogProb") -> bool:
        """Java ``compareTo``: descending probability ordering (<=)."""
        return self.prob >= other.prob

    def __gt__(self, other: "WordLogProb") -> bool:
        """Java ``compareTo``: descending probability ordering (>)."""
        return self.prob < other.prob

    def __ge__(self, other: "WordLogProb") -> bool:
        """Java ``compareTo``: descending probability ordering (>=)."""
        return self.prob <= other.prob

    def __str__(self) -> str:
        """Java ``toString`` -> ``<word> <prob>``."""
        return f"{self.word} {self.prob}"


WordLogProb.getWord = WordLogProb.get_word  # type: ignore[attr-defined]
WordLogProb.getProb = WordLogProb.get_prob  # type: ignore[attr-defined]
