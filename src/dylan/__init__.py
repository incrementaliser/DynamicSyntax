"""DyLan Dynamic Syntax + TTR core (Python port of qmul.ds)."""

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.parser.interactive_context_parser import InteractiveContextParser

__all__ = ["TTRRecordType", "InteractiveContextParser", "__version__"]

__version__ = "0.1.1"
