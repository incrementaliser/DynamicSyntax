"""Neuro-symbolic DS-TTR induction via probabilistic circuits (cirkit / SPL).

This package implements the BabyDS MVP described in
``docs/ds-ttr-pc-design-note.md``: compile the finite legal support from
:class:`~dylan.induction.em_learner.ttr_hypothesiser.TTRHypothesiser`,
gate sum weights with DS-VSS features, train by maximum likelihood, and
export ``lexicon-top-N.txt`` for the standard induction evaluator.

Requires the optional ``nesy`` extra (``libcirkit`` + torch). Mutually
exclusive with the ``video`` extra due to a scipy pin conflict.
"""

from __future__ import annotations

from dylan.nesy.evaluate import evaluate_learned_lexicon
from dylan.nesy.learner import CircuitWordLearner

__all__ = [
    "CircuitWordLearner",
    "evaluate_learned_lexicon",
]
