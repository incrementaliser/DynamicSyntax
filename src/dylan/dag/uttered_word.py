"""Word token with speaker (Java `UtteredWord`)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UtteredWord:
    """Surface word plus dialogue participant (and addressee for dyadic parsing)."""

    word: str | None
    speaker: str
    addressee: str = "you"

    def __post_init__(self) -> None:
        if self.word is not None:
            object.__setattr__(self, "word", self.word.lower())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UtteredWord):
            return False
        return (
            self.word == other.word
            and self.speaker == other.speaker
            and self.addressee == other.addressee
        )
