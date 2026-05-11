"""Central logging setup using loguru (replaces log4j usage in Java)."""

from __future__ import annotations

import logging
import sys
from typing import Final

from loguru import logger

_DEFAULT_FORMAT: Final = "<level>{level}</level> {module}: {message}"

# True after :func:`configure_logging` or :func:`ensure_library_loguru_stderr` has run.
_LIBRARY_LOGURU_CONFIGURED: bool = False


def _stderr_without_icp(record: dict) -> bool:
    """Exclude parser-bound records so :class:`~dylan.parser.interactive_context_parser.InteractiveContextParser` sinks own them."""
    return "icp_id" not in record["extra"]


def configure_logging(level: str = "INFO") -> None:
    """Configure one stderr sink for the process; strips default loguru handler first.

    Records carrying ``extra["icp_id"]`` are omitted here so per-parser sinks handle them.
    """
    global _LIBRARY_LOGURU_CONFIGURED
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=_DEFAULT_FORMAT,
        filter=_stderr_without_icp,
    )
    _LIBRARY_LOGURU_CONFIGURED = True


def ensure_library_loguru_stderr(level: str = "WARNING") -> None:
    """Install filtered stderr once so notebooks avoid loguru's default DEBUG sink for all modules.

    Without this, :meth:`~dylan.parser.interactive_context_parser.InteractiveContextParser`'s
    bound logger still prints when ``log_level`` is ``\"off\"`` (no per-parser sinks yet).
    Idempotent if :func:`configure_logging` ran earlier (e.g. tests).
    """
    global _LIBRARY_LOGURU_CONFIGURED
    if _LIBRARY_LOGURU_CONFIGURED:
        return
    configure_logging(level)


def sync_dylan_stdlib_level_for_icp(log_level: str) -> None:
    """Set stdlib ``logging`` level on the ``dylan`` namespace from parser ``log_level``.

    Lexicon/grammar/DAG still use stdlib loggers; ``off`` maps to ``ERROR`` so load/parse stay quiet.
    """
    lg = logging.getLogger("dylan")
    if log_level == "off":
        lg.setLevel(logging.ERROR)
    elif log_level == "error":
        lg.setLevel(logging.ERROR)
    elif log_level == "warning":
        lg.setLevel(logging.WARNING)