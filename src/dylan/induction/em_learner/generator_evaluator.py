"""Evaluator for generator learners."""

from __future__ import annotations

from dylan.induction.em_learner.generator_learner import GeneratorLearner


class GeneratorEvaluator:
    """Evaluate generator learner probabilities."""

    def __init__(self, learner: GeneratorLearner | None = None) -> None:
        """Create evaluator for *learner*."""
        self.learner = learner or GeneratorLearner()

    def score(self, examples: list[tuple[str, str]]) -> float:
        """Return average probability assigned to examples."""
        if not examples:
            return 0.0
        return sum(self.learner.probability(ctx, word) for ctx, word in examples) / len(examples)
