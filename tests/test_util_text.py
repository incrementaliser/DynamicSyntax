"""Utility string helpers."""

from __future__ import annotations

from dylan.util.text import casefold_equal


def test_casefold_equal() -> None:
    assert casefold_equal("a", "A")
    assert not casefold_equal("a", "b")
