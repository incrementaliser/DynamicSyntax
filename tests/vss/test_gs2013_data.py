"""Tests for GS2013 dataset loading."""

from __future__ import annotations

from dylan.vss.gs2013_data import convert_gs2013_to_ks_format, load_sentence_pairs


def test_convert_and_load_pairs() -> None:
    """KS conversion runs and yields paired annotations."""
    path = convert_gs2013_to_ks_format()
    assert path.is_file()
    pairs = load_sentence_pairs(path)
    assert len(pairs) > 100
    p0 = pairs[0]
    assert p0.first.subj == p0.second.subj
    assert p0.first.landmark == p0.second.landmark
    assert p0.gold_category in (0, 1)
