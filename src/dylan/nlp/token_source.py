"""`TokenSource` protocol and default whitespace tokenisation."""

from __future__ import annotations

from typing import Protocol, Sequence

from dylan.nlp.types import whitespace_tokenize


class TokenSource(Protocol):
    """Maps raw text to tokens (`DSParser` boundary; replaces Stanford token APIs)."""

    def tokenize(self, text: str) -> Sequence[str]:
        """Split `text` into surface tokens."""
        ...


class WhitespaceTokenSource:
    """Splits on whitespace (default for tests and minimal pipelines)."""

    def tokenize(self, text: str) -> list[str]:
        """Return lowercased non-empty whitespace-separated tokens."""
        return whitespace_tokenize(text)


def try_nltk_tokenize(text: str) -> list[str] | None:
    """Use NLTK word tokenizer when installed; otherwise return None."""
    try:
        from nltk import word_tokenize  # type: ignore[import-untyped]
    except ImportError:
        return None
    return [w.lower() for w in word_tokenize(text)]


def try_spacy_tokenize(text: str, model: str = "en_core_web_sm") -> list[str] | None:
    """Use spaCy when installed; otherwise return None."""
    try:
        import spacy  # type: ignore[import-untyped]
    except ImportError:
        return None
    nlp = spacy.load(model)
    doc = nlp(text)
    return [t.text.lower() for t in doc if not t.is_space]
