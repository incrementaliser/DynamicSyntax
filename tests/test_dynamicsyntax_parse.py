"""Smoke tests for the ``dynamicsyntax`` distribution facade."""

from __future__ import annotations

import pytest

import dynamicsyntax as ds
from dylan.formula.ttr_record_type import TTRRecordType


def test_import_dynamicsyntax_version() -> None:
    """Package exposes a PEP 440 version string."""
    assert isinstance(ds.__version__, str)
    assert ds.__version__


def test_get_grammars_includes_bundled_and_alias() -> None:
    """Bundled grammar dir and ``ttr`` alias are listed."""
    g = ds.get_grammars()
    assert "2015-english-ttr" in g
    assert "ttr" in g


def test_get_datasets_empty_placeholder() -> None:
    """No bundled datasets yet."""
    assert ds.get_datasets() == []


def test_parse_ttr_bundled_grammar() -> None:
    """End-to-end parse via bundled ``2015-english-ttr`` grammar."""
    p = ds.parse("a man arrives", "ttr")
    assert p.ok
    assert isinstance(p.semantics, TTRRecordType)
    s = str(p.semantics)
    assert "man(" in s and "arrive" in s
    assert p.tree is not None
    assert "00" in p.address_order


def test_parse_empty_returns_failed_result() -> None:
    """Whitespace-only input yields no semantics."""
    p = ds.parse("   ", "ttr")
    assert not p.ok
    assert p.semantics is None


def test_parse_unknown_grammar_raises() -> None:
    """Invalid grammar id raises :class:`FileNotFoundError`."""
    with pytest.raises(FileNotFoundError, match="unknown bundled grammar"):
        ds.parse("hello", "not-a-backend")


def test_load_grammar_then_parse() -> None:
    """Session-style parse reuses ``load_grammar``."""
    ds.load_grammar("ttr")
    p = ds.parse("a man arrives")
    assert p.ok
    assert isinstance(p.semantics, TTRRecordType)


def test_parse_without_grammar_and_without_load_raises() -> None:
    """Calling ``parse(s)`` with no prior ``load_grammar`` is rejected."""
    from dynamicsyntax._session import clear_grammar_session

    clear_grammar_session()
    try:
        with pytest.raises(ValueError, match="no grammar loaded"):
            ds.parse("hello")
    finally:
        ds.load_grammar("ttr")


def test_vis_prints_address_order(capsys: pytest.CaptureFixture[str]) -> None:
    """``ParseResult.vis`` prints the GUI address-order tree text."""
    p = ds.parse("a man arrives", "ttr")
    p.vis()
    out = capsys.readouterr().out
    assert "00" in out and "man" in out.lower()
