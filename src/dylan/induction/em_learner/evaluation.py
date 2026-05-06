"""Evaluation helpers for learned induction outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from dylan.formula.ttr_record_type import TTRRecordType


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Precision/recall/F-score tuple."""

    precision: float
    recall: float
    f_score: float


class Evaluation:
    """Static evaluation methods matching Java ``Evaluation``."""

    @staticmethod
    def precision(gold: TTRRecordType, predicted: TTRRecordType) -> float:
        """Return field precision of *predicted* against *gold*."""
        if predicted.num_fields() == 0:
            return 0.0
        hits = sum(1 for field in predicted.get_fields() if gold.has_field(field))
        return hits / predicted.num_fields()

    @staticmethod
    def recall(gold: TTRRecordType, predicted: TTRRecordType) -> float:
        """Return field recall of *predicted* against *gold*."""
        if gold.num_fields() == 0:
            return 0.0
        hits = sum(1 for field in gold.get_fields() if predicted.has_field(field))
        return hits / gold.num_fields()

    @classmethod
    def evaluate(cls, gold: TTRRecordType, predicted: TTRRecordType) -> EvaluationResult:
        """Return precision, recall and harmonic mean."""
        p = cls.precision(gold, predicted)
        r = cls.recall(gold, predicted)
        f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        return EvaluationResult(p, r, f)

    @staticmethod
    def find_best_ttr_interpretation(
        gold: TTRRecordType,
        candidates: Iterable[TTRRecordType],
    ) -> TTRRecordType | None:
        """Return candidate with best F-score against *gold*."""
        return max(candidates, key=lambda candidate: Evaluation.evaluate(gold, candidate).f_score, default=None)


Evaluation.findBestTTRInterpretation = Evaluation.find_best_ttr_interpretation  # type: ignore[attr-defined]
