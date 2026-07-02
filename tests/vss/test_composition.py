"""Unit tests for VSS composition operators."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dylan.vss.composition import baseline_stages, cosine_distance, interpret_sentence, pick_category
from dylan.vss.embedding_store import DemoEmbeddingStore
from dylan.vss.types import CompositionMethod
from dylan.vss.underspec import compose_incremental
from dylan.vss.types import UnderspecMethod


def test_interpret_sentence_shapes() -> None:
    """gs yields matrix; ks/ko yield vectors on demo-sized tensors."""
    store = DemoEmbeddingStore()
    s = store.get_noun("alpha")
    v = store.get_verb_tensor("like")
    o = store.get_noun("beta")
    reps = interpret_sentence(s, v, o)
    assert reps[CompositionMethod.gs].tensor.dim() == 2
    assert reps[CompositionMethod.ks].tensor.numel() == 4
    assert reps[CompositionMethod.ko].tensor.numel() == 4


def test_incremental_identity_three_stages() -> None:
    """Identity underspec produces three incremental stages."""
    store = DemoEmbeddingStore()
    s = store.get_noun("alpha")
    v = store.get_verb_tensor("like")
    o = store.get_noun("gamma")
    incr = compose_incremental(
        s,
        v,
        o,
        candidate_verbs=[v],
        candidate_objects=[o],
        method=UnderspecMethod.identity,
    )
    assert len(incr) == 3


def test_cosine_distance_and_pick() -> None:
    """Cosine distance is zero for identical vectors; pick_category breaks ties."""
    a = torch.tensor([1.0, 0.0, 0.0])
    assert cosine_distance(a, a) == pytest.approx(0.0, abs=1e-5)
    assert pick_category(0.1, 0.2) == 0
    assert pick_category(0.2, 0.1) == 1
    assert pick_category(0.5, 0.5) == -1


def test_baseline_stages_count() -> None:
    """Additive baseline returns three vectors."""
    store = DemoEmbeddingStore()
    stages = baseline_stages(
        store.get_noun("alpha"),
        store.get_verb_vector("like"),
        store.get_noun("beta"),
    )
    assert len(stages) == 3
