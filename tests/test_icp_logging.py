"""Tests for :class:`~dylan.parser.interactive_context_parser.InteractiveContextParser` loguru settings."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from dylan.dag.uttered_word import UtteredWord
from dylan.parser.interactive_context_parser import InteractiveContextParser

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "parser_minimal"


def _icp_filter(icp_id: str):
    """Build a loguru filter that matches a parser's bound ``icp_id``."""

    def _f(record: dict) -> bool:
        return record["extra"].get("icp_id") == icp_id

    return _f


def test_log_level_off_emits_no_icp_lines() -> None:
    """With ``log_level=off``, no per-parser loguru sinks are registered."""
    parser = InteractiveContextParser(FIXTURE, log_level="off")
    assert parser._icp_log_handler_ids == []


def test_log_level_error_captures_unknown_word() -> None:
    """``log_level=error`` records lexicon errors."""
    parser = InteractiveContextParser(FIXTURE, log_level="error", log_output="terminal")
    captured: list[str] = []
    hid = logger.add(
        lambda m: captured.append(m.record["message"]),
        level="ERROR",
        filter=_icp_filter(parser._icp_id),
    )
    try:
        parser.init()
        assert parser.parse_word(UtteredWord("not_a_real_word_zz", "Dylan", "you")) is None
    finally:
        logger.remove(hid)
    assert any("Word not in Lexicon" in msg for msg in captured)


def test_log_level_warning_suppresses_info() -> None:
    """``log_level=warning`` drops INFO lines such as ``Parsed``."""
    parser = InteractiveContextParser(FIXTURE, log_level="warning", log_output="terminal")
    captured: list[str] = []
    hid = logger.add(
        lambda m: captured.append((m.record["level"].name, m.record["message"])),
        level="WARNING",
        filter=_icp_filter(parser._icp_id),
    )
    try:
        parser.init()
        assert parser.parse_word(UtteredWord("test", "Dylan", "you")) is not None
    finally:
        logger.remove(hid)
    levels = {lvl for lvl, _ in captured}
    assert "INFO" not in levels
    assert not any("Parsed" in msg for _, msg in captured)


def test_icp_logging_suite_exits_after_multiple_terminal_parsers() -> None:
    """Regression: two terminal-log parsers must not deadlock loguru sink removal at GC."""
    parsers = [
        InteractiveContextParser(FIXTURE, log_level="error", log_output="terminal"),
        InteractiveContextParser(FIXTURE, log_level="warning", log_output="terminal"),
    ]
    for parser in parsers:
        parser.close()


def test_log_output_file_writes_under_log_dir(tmp_path: Path) -> None:
    """``log_output=file`` with *log_dir* creates a log file and does not require a terminal sink."""
    log_dir = tmp_path / "logs"
    parser = InteractiveContextParser(
        FIXTURE,
        log_level="error",
        log_output="file",
        log_dir=log_dir,
    )
    parser.init()
    parser.parse_word(UtteredWord("not_a_real_word_zz", "Dylan", "you"))
    files = list(log_dir.glob("icp_*.log"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "Word not in Lexicon" in text
