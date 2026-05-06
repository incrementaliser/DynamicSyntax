"""Load learned grammar/lexicon files."""

from __future__ import annotations

from pathlib import Path

from dylan.action.lexicon import Lexicon


class LoadLearntGrammar(Lexicon):
    """Lexicon subclass for learned grammar compatibility."""

    def __init__(self, path: str | Path | None = None, top_n: int = 0) -> None:
        """Load a learned grammar file or resource directory when available."""
        super().__init__(path, top_n) if path is not None and Path(path).is_dir() else super().__init__()
        self.path = Path(path) if path is not None else None


LoadLearntGrammar.loadLearntGrammar = LoadLearntGrammar  # type: ignore[attr-defined]
