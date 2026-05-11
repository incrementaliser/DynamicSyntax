"""Bind formula-layer loguru emission to :class:`~dylan.parser.interactive_context_parser.InteractiveContextParser` ``log_level``."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Generator

# ``unset``: not inside an ICP-driven parse/complete (preserve legacy visibility).
_FORMULA_SESSION_LEVEL: ContextVar[str] = ContextVar("_formula_session_level", default="unset")


def parser_emits_formula_debug() -> bool:
    """Whether formula code should emit ``logger.debug`` (session uses verbose parser logging)."""
    lv = _FORMULA_SESSION_LEVEL.get()
    if lv == "unset":
        return True
    if lv == "off":
        return False
    return lv == "warning"


def parser_emits_formula_error() -> bool:
    """Whether formula code should emit ``logger.error``."""
    lv = _FORMULA_SESSION_LEVEL.get()
    if lv == "unset":
        return True
    if lv == "off":
        return False
    return lv in ("error", "warning")


def parser_emits_formula_warning() -> bool:
    """Whether formula code should emit ``logger.warning``."""
    lv = _FORMULA_SESSION_LEVEL.get()
    if lv == "unset":
        return True
    return lv == "warning"


@contextmanager
def icp_parser_formula_log_context(level: str) -> Generator[None, None, None]:
    """Scope formula evaluation so ``ttr_path`` and similar respect parser *log_level*."""
    tok: Token = _FORMULA_SESSION_LEVEL.set(level)
    try:
        yield
    finally:
        _FORMULA_SESSION_LEVEL.reset(tok)
