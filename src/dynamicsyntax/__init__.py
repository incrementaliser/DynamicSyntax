"""Public facade for the *dynamicsyntax* distribution (high-level :func:`parse`)."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from dylan.action.lexicon import NotebookMultilineText
from dylan.formula.latex.build_result import LaTeXBuildResult
from dylan.formula.manim.models import ManimBuildResult
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.parser.interactive_context_parser import InteractiveContextParser

from dynamicsyntax._manim import to_manim
from dynamicsyntax._parse import parse
from dynamicsyntax._session import get_datasets, get_grammars, load_grammar
from dynamicsyntax.parse_result import ParseResult

try:
    __version__: str = version("dynamicsyntax")
except PackageNotFoundError:  # pragma: no cover - editable checkout without metadata
    __version__ = "0.0.0"

__all__ = [
    "InteractiveContextParser",
    "LaTeXBuildResult",
    "ManimBuildResult",
    "NotebookMultilineText",
    "ParseResult",
    "TTRRecordType",
    "__version__",
    "get_datasets",
    "get_grammars",
    "load_grammar",
    "parse",
    "to_manim",
]
