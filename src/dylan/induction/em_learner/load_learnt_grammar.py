"""Load learned grammar/lexicon files (Java ``qmul.ds.learn.LoadLearntGrammar``)."""

from __future__ import annotations

import logging
from pathlib import Path

from dylan.action.grammar import Grammar
from dylan.action.lexicon import Lexicon

logger = logging.getLogger(__name__)


class LoadLearntGrammar(Lexicon):
    """Lexicon subclass that knows how to bootstrap a :class:`TestParser` from learnt files."""

    def __init__(self, path: "str | Path | None" = None, top_n: int = 0) -> None:
        """Optionally load a learned grammar directory at *path*."""
        if path is not None and Path(path).is_dir():
            super().__init__(path, top_n)
        else:
            super().__init__()
        self.path = Path(path) if path is not None else None

    @staticmethod
    def test_from_text(grammar_path: "str | Path", top_n: int = 1):
        """Java ``testFromText``: build a :class:`TestParser` from text-based grammar files."""
        from dylan.induction.em_learner.test_parser import TestParser

        path = Path(grammar_path)
        lex = Lexicon(path)
        comp = Grammar(path)
        return TestParser(lex, comp, top_n=top_n)

    @staticmethod
    def test_from_binary(grammar_path: "str | Path", top_n: int = 1):
        """Java ``testFromBinary``: text-only fallback in Python (no pickled lexicon)."""
        logger.warning("test_from_binary: binary lexicon load not ported; falling back to text loader")
        return LoadLearntGrammar.test_from_text(grammar_path, top_n)


LoadLearntGrammar.testFromText = staticmethod(LoadLearntGrammar.test_from_text)  # type: ignore[method-assign]
LoadLearntGrammar.testFromBinary = staticmethod(LoadLearntGrammar.test_from_binary)  # type: ignore[method-assign]
