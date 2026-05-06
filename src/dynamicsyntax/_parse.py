"""High-level :func:`parse` using bundled grammars and ``dylan`` parser core."""

from __future__ import annotations

from pathlib import Path

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.nlp.types import DEFAULT_SPEAKER, utterance_from_text
from dylan.parser.interactive_context_parser import InteractiveContextParser

from dynamicsyntax._session import resolved_grammar_path, session_parser
from dynamicsyntax.parse_result import ParseResult


def _parse_at_path(grammar_path: Path, sentence: str, *, speaker: str) -> ParseResult:
    """Run init / new_sentence / parse at *grammar_path* and build a :class:`ParseResult`."""
    parser = InteractiveContextParser(grammar_path)
    parser.init()
    parser.new_sentence()
    utt = utterance_from_text(speaker, sentence.strip())
    ok = parser.parse_utterance(utt)
    tree = parser.get_best_tuple().get_tree()
    semantics: TTRRecordType | None = parser.get_final_semantics() if ok else None
    return ParseResult(ok=ok, semantics=semantics, tree=tree)


def parse(
    sentence: str,
    grammar: str | Path | None = None,
    /,
    *,
    speaker: str = DEFAULT_SPEAKER,
) -> ParseResult:
    """Parse *sentence* and return a :class:`~dynamicsyntax.parse_result.ParseResult`.

    :param sentence: Whitespace-tokenised surface string (lowercased by the tokenizer).
    :param grammar: Bundled id or alias (e.g. ``\"ttr\"``), a grammar directory path, or
        ``None`` to use the parser from :func:`~dynamicsyntax.load_grammar`.
    :param speaker: Dialogue participant id passed to the parser (default matches ``dylan``).
    :returns: :class:`~dynamicsyntax.parse_result.ParseResult` with ``ok``, ``semantics``,
        and ``tree``; ``semantics`` is ``None`` on failure or blank input.
    :raises ValueError: If *grammar* is ``None`` but no grammar was loaded.
    :raises FileNotFoundError: If *grammar* is unknown or not a directory.

    Bundled grammars are read via :mod:`importlib.resources`; each one-shot parse with an
    explicit *grammar* uses a fresh parser under a short-lived extract path.
    """
    stripped = sentence.strip()
    if not stripped:
        return ParseResult(ok=False, semantics=None, tree=None)

    if grammar is not None:
        with resolved_grammar_path(grammar) as grammar_path:
            return _parse_at_path(grammar_path, stripped, speaker=speaker)

    parser = session_parser()
    if parser is None:
        raise ValueError("no grammar loaded; call load_grammar(...) first or pass grammar= to parse(...)")
    parser.init()
    parser.new_sentence()
    utt = utterance_from_text(speaker, stripped)
    ok = parser.parse_utterance(utt)
    tree = parser.get_best_tuple().get_tree()
    semantics = parser.get_final_semantics() if ok else None
    return ParseResult(ok=ok, semantics=semantics, tree=tree)
