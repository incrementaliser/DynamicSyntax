"""Seeded TTR hypothesiser compatibility class."""

from __future__ import annotations

from dylan.induction.em_learner.ttr_hypothesiser import TTRHypothesiser


class SeededTTRHypothesiser(TTRHypothesiser):
    """Thin seeded variant of :class:`TTRHypothesiser`."""
