"""Utterance and lightweight tokenisation (replaces Stanford `HasWord` usage)."""

from __future__ import annotations

from dataclasses import dataclass, field

from dylan.dag.uttered_word import UtteredWord

DEFAULT_SPEAKER = "Dylan"
RELEASE_TURN_TOKEN = "<rt>"
WAIT_TOKEN = "<wait>"


@dataclass
class Utterance:
    """Sequence of `UtteredWord` with a speaker id (Java `Utterance`, simplified)."""

    speaker: str
    words: list[UtteredWord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.words)


def whitespace_tokenize(text: str) -> list[str]:
    """Split on whitespace for tests and default parsing."""
    return [w for w in text.strip().lower().split() if w]


def utterance_from_text(speaker: str, text: str) -> Utterance:
    """Build an `Utterance` from raw text using `whitespace_tokenize`."""
    return Utterance(
        speaker=speaker,
        words=[UtteredWord(w, speaker) for w in whitespace_tokenize(text)],
    )
