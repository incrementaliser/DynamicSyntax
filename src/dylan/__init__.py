"""DyLan Dynamic Syntax + TTR core (Python port of qmul.ds)."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.parser.interactive_context_parser import InteractiveContextParser

__all__ = ["TTRRecordType", "InteractiveContextParser", "__version__"]

try:
    __version__: str = version("dynamicsyntax")
except PackageNotFoundError:  # pragma: no cover - editable checkout without metadata
    __version__ = "0.0.0"
