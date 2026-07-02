"""Common helpers for the EM induction learner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Word:
    """Small Stanford ``Word``/``HasWord`` equivalent."""

    text: str

    def word(self) -> str:
        """Return the surface form."""
        return self.text

    def __str__(self) -> str:
        """Return the surface form."""
        return self.text


def as_word(value: str | Word | Any) -> Word:
    """Convert Java-style word-like objects to :class:`Word`."""
    if isinstance(value, Word):
        return value
    word_fn = getattr(value, "word", None)
    if callable(word_fn):
        return Word(str(word_fn()))
    return Word(str(value))


def sentence_from_text(text: str) -> list[Word]:
    """Split *text* into a Java ``Sentence<Word>`` style list."""
    return [Word(part) for part in text.strip().split() if part]


def words_to_string(words: Iterable[str | Word | Any]) -> str:
    """Return a space-separated sentence for a word sequence."""
    return " ".join(as_word(word).word() for word in words)


def resolve_path(path: str | Path) -> Path:
    """Return *path* as a :class:`Path` without requiring it to exist."""
    return path if isinstance(path, Path) else Path(path)


def action_key(action: Any) -> str:
    """Stable per-action key for candidate dedup / intersection (includes lexical payload).

    :class:`~dylan.induction.em_learner.lexical_hypothesis.LexicalHypothesis` overrides
    ``__str__`` to the surface name only, so the generic ``str(action)`` suffix would
    collapse distinct induced effects into one key and wrongly treat sequences as duplicates.
    """
    from dylan.action.lexical_action import LexicalAction
    from dylan.induction.em_learner.lexical_hypothesis import LexicalHypothesis

    if isinstance(action, LexicalHypothesis):
        eff = getattr(action, "effect", None)
        eff_s = str(eff) if eff is not None else "None"
        return f"LexicalHypothesis:{action.get_name()}:{eff_s}"
    if isinstance(action, LexicalAction):
        lines = tuple(getattr(action, "_source_lines", None) or ())
        return f"LexicalAction:{action.word}:{lines!r}"
    name_fn = getattr(action, "get_name", None)
    if callable(name_fn):
        return f"{type(action).__name__}:{name_fn()}:{action}"
    name = getattr(action, "name", None)
    return f"{type(action).__name__}:{name}:{action}"
