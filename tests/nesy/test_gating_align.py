"""Gating alignment tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.em_learner.record_type_corpus import RecordTypeCorpus
from dylan.nesy.gating import SoftmaxGate, WordHypLogits
from dylan.nesy.vss_features import build_aligned_lexicon, predicates_from_rt


def test_softmax_gate_shapes() -> None:
    """Gate maps feature dim → n_hyps and sums to 1."""
    g = SoftmaxGate(2, 4)
    z = torch.randn(2)
    w = g(z)
    assert w.shape == (4,)
    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-5)


def test_word_hyp_logits_with_gate() -> None:
    """Gated log-probs align with catalog size."""
    params = WordHypLogits({"pickup": [1, 2, 3]}, feature_dim=2, use_gate=True)
    ids, log_p = params.hyp_log_probs("pickup", z=torch.tensor([0.5, 0.5]))
    assert ids == [1, 2, 3]
    assert log_p.shape == (3,)
    assert torch.allclose(log_p.exp().sum(), torch.tensor(1.0), atol=1e-5)


def test_word_hyp_logits_reserved_word_keys() -> None:
    """Surface words that collide with nn.Module attrs still allocate cleanly."""
    params = WordHypLogits(
        {"get": [1, 2], "to": [3], "add": [4, 5, 6]},
        feature_dim=2,
        use_gate=True,
    )
    ids, log_p = params.hyp_log_probs("get", z=None)
    assert ids == [1, 2]
    assert log_p.shape == (2,)
    assert params.n_parameters() > 0


def test_predicates_from_babyds_rt() -> None:
    """Gold BabyDS-style RT yields event/entity constants."""
    rt = TTRRecordType.parse(
        "[e0==state_holding : es|r0 : [x0 : e|p0==obj_ball(x0) : t|head==x0 : e]"
        "|head==e0 : es|x1==epsilon(r0.head, r0) : e|p2==obj(e0, x1) : t]"
    )
    assert rt is not None
    preds = predicates_from_rt(rt)
    consts = {c for c, _ in preds}
    assert "state_holding" in consts


def test_build_aligned_lexicon_smoke() -> None:
    """Lexicon builder returns entities and at least one event matrix."""
    path = Path("data/BabyDS/class1_train_100.txt")
    if not path.is_file():
        pytest.skip("BabyDS train file missing")
    corpus = RecordTypeCorpus(corpus_path=path)
    lex = build_aligned_lexicon(corpus)
    assert lex.word_space.dim >= 2
    assert lex._entities or lex._matrices
