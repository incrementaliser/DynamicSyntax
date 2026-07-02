"""Evaluation helpers for induction outputs (Java ``qmul.ds.learn.Evaluation``).

Implements the precision/recall/F-score evaluation over hypothesised vs gold
TTR record-type pairs, including macro and micro averaging.  The full Java
``maximalMapping`` algorithm is summarised by a subsumption-based scoring
function with the same public surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from dylan.formula.ttr_record_type import TTRRecordType

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Java inner class ``EvaluationResult``: precision/recall/F-score holder."""

    precision: float
    recall: float
    f_score: float

    def get_precision(self) -> float:
        """Java ``getPrecision``."""
        return self.precision

    def get_recall(self) -> float:
        """Java ``getRecall``."""
        return self.recall

    def get_f_score(self) -> float:
        """Java ``getFScore``."""
        return self.f_score


class Evaluation:
    """Java ``Evaluation``: precision/recall over predicted vs gold TTR pairs."""

    @staticmethod
    def field_total(ttr: TTRRecordType) -> int:
        """Java ``fieldTotal``: count fields recursively, including embedded RTs."""
        if ttr is None:
            return 0
        total = 0
        for f in ttr.get_fields() if hasattr(ttr, "get_fields") else []:
            total += 1
            inner = f.get_type() if hasattr(f, "get_type") else None
            if isinstance(inner, TTRRecordType):
                total += Evaluation.field_total(inner)
        return total

    @staticmethod
    def total_nodes_mapped(hyp: "TTRRecordType | None", goal: "TTRRecordType | None") -> float:
        """Java ``totalNodesMapped``: subsumption-based score (simplified port)."""
        if hyp is None or goal is None:
            return 0.0
        score = 0.0
        for gf in goal.get_fields() if hasattr(goal, "get_fields") else []:
            best = 0.0
            for hf in hyp.get_fields() if hasattr(hyp, "get_fields") else []:
                if hasattr(hf, "subsumes") and hf.subsumes(gf):
                    best = max(best, 1.0)
                elif hasattr(gf, "subsumes") and gf.subsumes(hf):
                    best = max(best, 0.75)
                elif hasattr(hf, "get_label") and hasattr(gf, "get_label") and hf.get_label() == gf.get_label():
                    best = max(best, 0.25)
            score += best
        return score

    @staticmethod
    def precision_recall(hyp: TTRRecordType, goal: TTRRecordType) -> EvaluationResult:
        """Compute per-pair precision/recall (driver behind macro-averaging)."""
        hyp_total = Evaluation.field_total(hyp)
        goal_total = Evaluation.field_total(goal)
        nodes_mapped = Evaluation.total_nodes_mapped(hyp, goal)
        precision = (nodes_mapped / hyp_total) if hyp_total else 0.0
        recall = (nodes_mapped / goal_total) if goal_total else 0.0
        f_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return EvaluationResult(precision, recall, f_score)

    @staticmethod
    def precision_recall_macro(pairs: "Iterable[tuple[TTRRecordType, TTRRecordType]]") -> list[float]:
        """Java ``precisionRecallMacro``: arithmetic mean over per-pair scores."""
        items = list(pairs)
        if not items:
            return [0.0, 0.0, 0.0]
        sum_p = sum_r = sum_f = 0.0
        for hyp, goal in items:
            res = Evaluation.precision_recall(hyp, goal)
            sum_p += res.precision
            sum_r += res.recall
            sum_f += res.f_score
        n = float(len(items))
        return [sum_p / n, sum_r / n, sum_f / n]

    @staticmethod
    def precision_recall_micro(pairs: "Iterable[tuple[TTRRecordType, TTRRecordType]]") -> EvaluationResult:
        """Java ``precisionRecallMicro``: aggregate node counts then compute scores."""
        overall_hyp = overall_goal = overall_mapped = 0.0
        for hyp, goal in pairs:
            overall_hyp += Evaluation.total_nodes_mapped(hyp, hyp)
            overall_goal += Evaluation.total_nodes_mapped(goal, goal)
            overall_mapped += Evaluation.total_nodes_mapped(hyp, goal)
        precision = (overall_mapped / overall_hyp) if overall_hyp else 0.0
        recall = (overall_mapped / overall_goal) if overall_goal else 0.0
        f = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return EvaluationResult(precision, recall, f)

    @staticmethod
    def find_best_ttr_interpretation(
        gold: TTRRecordType,
        candidates: "Iterable[TTRRecordType]",
    ) -> "TTRRecordType | None":
        """Return the candidate with the best F-score against *gold*."""
        return max(
            candidates,
            key=lambda c: Evaluation.precision_recall(c, gold).f_score,
            default=None,
        )


Evaluation.fieldTotal = staticmethod(Evaluation.field_total)  # type: ignore[method-assign]
Evaluation.totalNodesMapped = staticmethod(Evaluation.total_nodes_mapped)  # type: ignore[method-assign]
Evaluation.precisionRecall = staticmethod(Evaluation.precision_recall)  # type: ignore[method-assign]
Evaluation.precisionRecallMacro = staticmethod(Evaluation.precision_recall_macro)  # type: ignore[method-assign]
Evaluation.precisionRecallMicro = staticmethod(Evaluation.precision_recall_micro)  # type: ignore[method-assign]
Evaluation.findBestTTRInterpretation = staticmethod(Evaluation.find_best_ttr_interpretation)  # type: ignore[method-assign]
