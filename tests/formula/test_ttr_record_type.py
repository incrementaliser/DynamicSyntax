"""Tests for `TTRRecordType` parsing and equality (plan § Tests for TTRRecordType)."""

from __future__ import annotations

from dylan.formula.ttr_record_type import TTRRecordType


def test_parse_empty_record() -> None:
    r = TTRRecordType.parse("[]")
    assert r is not None
    assert r.is_empty()


def test_parse_single_field() -> None:
    r = TTRRecordType.parse("[x:es]")
    assert r is not None
    assert len(r.fields) == 1


def test_parse_malformed_returns_none() -> None:
    assert TTRRecordType.parse("not a record") is None


def test_round_trip_str() -> None:
    r = TTRRecordType.parse("[]")
    assert r is not None
    assert TTRRecordType.parse(str(r)) == r


def test_remove_head_idempotent_on_empty() -> None:
    r = TTRRecordType.parse("[]")
    assert r is not None
    h = r.remove_head()
    assert h.is_empty()
