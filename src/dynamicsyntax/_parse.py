"""High-level :func:`parse` using bundled grammars and ``dylan`` parser core."""

from __future__ import annotations

from contextlib import AbstractContextManager
from importlib import resources
from pathlib import Path
from typing import TypeAlias

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.nlp.types import DEFAULT_SPEAKER, utterance_from_text
from dylan.parser.interactive_context_parser import InteractiveContextParser

# Default TTR English grammar snapshot shipped under ``grammars/`` (see package-data).
_BACKEND_GRAMMAR_SUBDIR: dict[str, str] = {
    "ttr": "grammars/2015-english-ttr",
}

_AsFileCtx: TypeAlias = AbstractContextManager[Path]


def _grammar_as_file(backend: str) -> tuple[_AsFileCtx, str]:
    """Return a context manager yielding a filesystem path to the bundled grammar dir."""
    sub = _BACKEND_GRAMMAR_SUBDIR.get(backend)
    if sub is None:
        allowed = ", ".join(sorted(_BACKEND_GRAMMAR_SUBDIR))
        raise ValueError(f"unknown backend {backend!r}; use one of: {allowed}")
    root = resources.files("dynamicsyntax")
    node = root / sub
    if not node.is_dir():
        raise FileNotFoundError(f"bundled grammar missing: {sub!r} under dynamicsyntax package")
    return resources.as_file(node), sub


def parse(
    sentence: str,
    backend: str,
    /,
    *,
    speaker: str = DEFAULT_SPEAKER,
) -> TTRRecordType | None:
    """Parse *sentence* with a bundled grammar and return evaluated TTR record semantics.

    :param sentence: Whitespace-tokenised surface string (lowercased by the tokenizer).
    :param backend: ``\"ttr\"`` uses the bundled ``2015-english-ttr`` snapshot.
    :param speaker: Dialogue participant id passed to the parser (default matches ``dylan``).
    :returns: :class:`~dylan.formula.ttr_record_type.TTRRecordType` on success, or ``None``
        if any token fails to parse.

    The grammar files are loaded from the installed package via :mod:`importlib.resources`;
    each call constructs a fresh parser so temporary extraction paths from wheels stay valid.
    """
    stripped = sentence.strip()
    if not stripped:
        return None
    ctx, _subdir = _grammar_as_file(backend)
    with ctx as grammar_path:
        parser = InteractiveContextParser(grammar_path)
        parser.init()
        parser.new_sentence()
        utt = utterance_from_text(speaker, stripped)
        if not parser.parse_utterance(utt):
            return None
        return parser.get_final_semantics()
