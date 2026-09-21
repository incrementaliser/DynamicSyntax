"""Tests for discrete support encoding."""

from __future__ import annotations

from dylan.nesy.support import EncodedExample, collect_global_hyp_catalog, hyp_key


def test_hyp_key() -> None:
    """Stable word/hyp key format."""
    assert hyp_key("pickup", 3) == "pickup::3"


def test_collect_global_hyp_catalog() -> None:
    """Union of hyp ids across examples."""
    from dylan.induction.em_learner.common import Word

    examples = [
        EncodedExample(
            sentence="a b",
            words=[Word("a"), Word("b")],
            legal_tuples=[[1, 2]],
            word_hyp_ids={"a": [1], "b": [2]},
        ),
        EncodedExample(
            sentence="a c",
            words=[Word("a"), Word("c")],
            legal_tuples=[[1, 3], [4, 3]],
            word_hyp_ids={"a": [1, 4], "c": [3]},
        ),
    ]
    cat = collect_global_hyp_catalog(examples)
    assert cat["a"] == [1, 4]
    assert cat["b"] == [2]
    assert cat["c"] == [3]
