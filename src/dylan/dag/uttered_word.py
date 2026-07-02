"""Word token with speaker (Java `UtteredWord`)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UtteredWord:
    """Surface word plus dialogue participant (and addressee for dyadic parsing)."""

    word: str | None
    speaker: str = "Self"
    addressee: str = "you"

    def __post_init__(self) -> None:
        """Normalize surface word to lower case like Java parsing paths."""
        if self.word is not None:
            object.__setattr__(self, "word", self.word.lower())

    def __eq__(self, other: object) -> bool:
        """Compare by word, speaker, and addressee."""
        if not isinstance(other, UtteredWord):
            return False
        return (
            self.word == other.word
            and self.speaker == other.speaker
            and self.addressee == other.addressee
        )

    def get_word(self) -> str | None:
        """Return the surface word."""
        return self.word

    def get_speaker(self) -> str:
        """Return the speaker id."""
        return self.speaker

    def get_addressee(self) -> str:
        """Return the addressee id."""
        return self.addressee

    def __str__(self) -> str:
        """Return Java-like debug text."""
        return f"{self.speaker}:{self.word}->{self.addressee}"


UtteredWord.word_ = UtteredWord.get_word  # type: ignore[attr-defined]
UtteredWord.getWord = UtteredWord.get_word  # type: ignore[attr-defined]
UtteredWord.speaker_ = UtteredWord.get_speaker  # type: ignore[attr-defined]
UtteredWord.getSpeaker = UtteredWord.get_speaker  # type: ignore[attr-defined]
UtteredWord.addressee_ = UtteredWord.get_addressee  # type: ignore[attr-defined]
UtteredWord.getAddressee = UtteredWord.get_addressee  # type: ignore[attr-defined]
