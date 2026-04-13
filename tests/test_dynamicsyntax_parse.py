"""Smoke tests for the ``dynamicsyntax`` distribution facade."""

from __future__ import annotations

import pytest

import dynamicsyntax as ds
from dylan.formula.ttr_record_type import TTRRecordType


def test_import_dynamicsyntax_version() -> None:
    """Package exposes a PEP 440 version string."""
    assert isinstance(ds.__version__, str)
    assert ds.__version__


def test_parse_ttr_bundled_grammar() -> None:
    """End-to-end parse via bundled ``2015-english-ttr`` grammar."""
    sem = ds.parse("a man arrives", "ttr")
    assert isinstance(sem, TTRRecordType)
    s = str(sem)
    assert "man(" in s and "arrive" in s


def test_parse_empty_returns_none() -> None:
    """Whitespace-only input yields no semantics."""
    assert ds.parse("   ", "ttr") is None


def test_parse_unknown_backend_raises() -> None:
    """Invalid *backend* is rejected with :class:`ValueError`."""
    with pytest.raises(ValueError, match="unknown backend"):
        ds.parse("hello", "not-a-backend")
