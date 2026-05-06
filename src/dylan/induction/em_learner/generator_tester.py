"""Test harness for generator learners."""

from __future__ import annotations

from dylan.induction.em_learner.generator_evaluator import GeneratorEvaluator
from dylan.induction.em_learner.generator_learner import GeneratorLearner


class GeneratorTester:
    """Small wrapper combining a generator learner and evaluator."""

    def __init__(self, learner: GeneratorLearner | None = None) -> None:
        """Create a tester."""
        self.learner = learner or GeneratorLearner()
        self.evaluator = GeneratorEvaluator(self.learner)

    def test(self, examples: list[tuple[str, str]]) -> float:
        """Return evaluator score for *examples*."""
        return self.evaluator.score(examples)
