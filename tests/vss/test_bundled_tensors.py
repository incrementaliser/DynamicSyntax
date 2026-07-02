"""Tests for bundled word2vec verb tensors (skipped if archive missing)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from dylan.vss.embedding_store import BundledWord2VecStore, _DEFAULT_TENSOR_ZIP

pytestmark = pytest.mark.skipif(
    not _DEFAULT_TENSOR_ZIP.is_file(),
    reason="word2vec_tensors_trained.zip not present",
)


def test_bundled_verb_tensor_shape() -> None:
    """Default store loads 300×300 verb matrices from the tensor zip."""
    store = BundledWord2VecStore(load_vectors=False)
    assert len(store._verb_tensors) > 0
    t = store.get_verb_tensor("accept")
    assert t.shape == (300, 300)


def test_bundled_default_paths() -> None:
    """Tensor zip lives beside the vss package."""
    assert Path(_DEFAULT_TENSOR_ZIP).name == "word2vec_tensors_trained.zip"
