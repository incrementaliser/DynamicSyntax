"""Public facade for the *dynamicsyntax* distribution (high-level :func:`parse`)."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.parser.interactive_context_parser import InteractiveContextParser

from dynamicsyntax._parse import parse
from dynamicsyntax._session import get_datasets, get_grammars, load_grammar
from dynamicsyntax.parse_result import ParseResult

try:
    __version__: str = version("dynamicsyntax")
except PackageNotFoundError:  # pragma: no cover - editable checkout without metadata
    __version__ = "0.0.0"

__all__ = [
    "InteractiveContextParser",
    "ParseResult",
    "TTRRecordType",
    "__version__",
    "get_datasets",
    "get_grammars",
    "load_grammar",
    "parse",
]
