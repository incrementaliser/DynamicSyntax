"""Evaluation smoke tests with demo embeddings."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from dylan.vss.embedding_store import DemoEmbeddingStore
from dylan.vss.evaluate import evaluate_gs2013
from dylan.vss.gs2013_data import load_sentence_pairs
from dylan.vss.types import CompositionMethod, EvaluationMode, UnderspecMethod


def test_evaluate_demo_subset() -> None:
    """Run evaluation on two synthetic pairs using DemoEmbeddingStore."""
    store = DemoEmbeddingStore()
    pairs = load_sentence_pairs()[:2]
    # Patch pairs to use demo lemmas only
    from dylan.vss.types import GS2013Pair, GS2013Sentence

    demo_pairs = [
        GS2013Pair(
            first=GS2013Sentence("1", "alpha", "like", "draw", "beta"),
            second=GS2013Sentence("2", "alpha", "draw", "like", "beta"),
            gold_category=0,
        )
    ]
    result = evaluate_gs2013(
        store,
        mode=EvaluationMode.tensor_only,
        pairs=demo_pairs,
    )
    acc = result.accuracy(UnderspecMethod.identity, CompositionMethod.baseline, 0)
    assert 0.0 <= acc <= 1.0
