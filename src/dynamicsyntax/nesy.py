"""High-level facade for neuro-symbolic (cirkit / SPL) grammar induction."""

from __future__ import annotations

from dylan.nesy.compile import HAS_CIRKIT, EnumeratedSupportCircuit, try_build_cirkit_categorical
from dylan.nesy.evaluate import evaluate_learned_lexicon, metrics_to_dict
from dylan.nesy.gating import SoftmaxGate, WordHypLogits
from dylan.nesy.learner import CircuitWordLearner
from dylan.nesy.support import EncodedExample, collect_global_hyp_catalog
from dylan.nesy.vss_features import build_aligned_lexicon

__all__ = [
    "HAS_CIRKIT",
    "CircuitWordLearner",
    "EncodedExample",
    "EnumeratedSupportCircuit",
    "SoftmaxGate",
    "WordHypLogits",
    "build_aligned_lexicon",
    "collect_global_hyp_catalog",
    "evaluate_learned_lexicon",
    "metrics_to_dict",
    "try_build_cirkit_categorical",
]
