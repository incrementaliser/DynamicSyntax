"""Tests for unsupervised API stubs."""

from __future__ import annotations

from pathlib import Path

import pytest

from dylan.vss.embedding_store import DemoEmbeddingStore
from dylan.vss.nesy.dataset import LatentParseDataset
from dylan.vss.nesy.learner import NesyDSVSSLearner

_VSS_GRAMMAR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "dylan"
    / "vss"
    / "resources"
    / "vss-transitive"
)


def test_fit_unsupervised_raises() -> None:
    """Unsupervised training is documented but not implemented yet."""
    learner = NesyDSVSSLearner(_VSS_GRAMMAR, store=DemoEmbeddingStore())
    data = LatentParseDataset.from_sentences(["table draw eye"], grammar_path=_VSS_GRAMMAR)
    with pytest.raises(NotImplementedError, match="Unsupervised"):
        learner.fit_unsupervised(data)
