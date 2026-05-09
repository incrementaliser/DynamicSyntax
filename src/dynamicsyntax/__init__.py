"""Public facade for the *dynamicsyntax* distribution (high-level :func:`parse` and :func:`icp`)."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from dylan.action.lexicon import NotebookMultilineText
from dylan.formula.latex.build_result import LaTeXBuildResult
from dylan.formula.manim.models import ManimBuildResult
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.parser.interactive_context_parser import DEFAULT_NAME, InteractiveContextParser

from dynamicsyntax._manim import to_manim
from dynamicsyntax._parse import parse
from dynamicsyntax._session import get_datasets, get_grammars
from dynamicsyntax.parse_result import ParseResult

try:
    __version__: str = version("dynamicsyntax")
except PackageNotFoundError:  # pragma: no cover - editable checkout without metadata
    __version__ = "0.0.0"


def icp(
    grammar: str | Path | None = None,
    *,
    repairing: bool = False,
    top_n: int | tuple[str, ...] = 3,
    participants: tuple[str, ...] = (DEFAULT_NAME,),
) -> InteractiveContextParser:
    """Return an :class:`~dylan.parser.interactive_context_parser.InteractiveContextParser` (short alias ``icp``).

    With no *grammar*, returns an unloaded parser; call :meth:`~dylan.parser.interactive_context_parser.InteractiveContextParser.set_grammar`
    before :meth:`~dylan.parser.interactive_context_parser.InteractiveContextParser.parse`.
    """
    return InteractiveContextParser(grammar, repairing=repairing, top_n=top_n, participants=participants)


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
    "icp",
    "parse",
    "to_manim",
]
