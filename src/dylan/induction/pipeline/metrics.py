"""Evaluation result containers and TSV export."""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SplitMetrics:
    """Scores for one corpus split at one top-N rank."""

    precision: float
    recall: float
    f1: float
    coverage: float
    exact_match: float
    parsed_count: int
    total_count: int
    exact_match_count: int
    failed_parses: list[str] = field(default_factory=list)
    failed_exact_matches: list[str] = field(default_factory=list)

    def as_percent_dict(self) -> dict[str, float]:
        """Return P/R/F1/coverage/EM as percentages."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "coverage": self.coverage,
            "exact_match": self.exact_match,
        }


@dataclass
class EvalResult:
    """Nested metrics: top_n -> split_name -> :class:`SplitMetrics`."""

    by_top_n: dict[int, dict[str, SplitMetrics]] = field(default_factory=dict)
    dataset_sizes: dict[str, int] = field(default_factory=dict)
    fold_results: list["EvalResult"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(
        self,
        top_n: int,
        split_name: str,
        metrics: SplitMetrics,
    ) -> None:
        """Store metrics for *split_name* at *top_n*."""
        self.by_top_n.setdefault(top_n, {})[split_name] = metrics
        self.dataset_sizes[split_name] = metrics.total_count

    def get(self, top_n: int, split_name: str) -> SplitMetrics | None:
        """Return metrics for *split_name* at *top_n*, if present."""
        return self.by_top_n.get(top_n, {}).get(split_name)

    def split_names(self) -> list[str]:
        """Return ordered unique split names across all top-N entries."""
        names: list[str] = []
        for split_map in self.by_top_n.values():
            for name in split_map:
                if name not in names:
                    names.append(name)
        return names

    def top_ns(self) -> list[int]:
        """Return sorted top-N ranks present in the result."""
        return sorted(self.by_top_n)

    def mean_over_folds(self) -> "EvalResult":
        """Average fold_results into a single :class:`EvalResult` (mean of each metric)."""
        if not self.fold_results:
            return self
        aggregated = EvalResult(metadata=dict(self.metadata))
        aggregated.metadata["n_folds"] = len(self.fold_results)
        top_ns = sorted({n for fr in self.fold_results for n in fr.top_ns()})
        splits = []
        for fr in self.fold_results:
            for name in fr.split_names():
                if name not in splits:
                    splits.append(name)
        for n in top_ns:
            for split in splits:
                samples = [fr.get(n, split) for fr in self.fold_results]
                present = [m for m in samples if m is not None]
                if not present:
                    continue
                aggregated.add(
                    n,
                    split,
                    SplitMetrics(
                        precision=_mean(m.precision for m in present),
                        recall=_mean(m.recall for m in present),
                        f1=_mean(m.f1 for m in present),
                        coverage=_mean(m.coverage for m in present),
                        exact_match=_mean(m.exact_match for m in present),
                        parsed_count=int(_mean(m.parsed_count for m in present)),
                        total_count=int(_mean(m.total_count for m in present)),
                        exact_match_count=int(_mean(m.exact_match_count for m in present)),
                    ),
                )
                aggregated.metadata.setdefault("std", {}).setdefault(n, {})[split] = {
                    "f1": _stdev(m.f1 for m in present),
                    "coverage": _stdev(m.coverage for m in present),
                    "exact_match": _stdev(m.exact_match for m in present),
                }
        aggregated.fold_results = list(self.fold_results)
        return aggregated


def _mean(values: Any) -> float:
    """Return arithmetic mean of *values*."""
    seq = list(values)
    return float(statistics.fmean(seq)) if seq else 0.0


def _stdev(values: Any) -> float:
    """Return sample standard deviation, or 0 for a single value."""
    seq = list(values)
    if len(seq) < 2:
        return 0.0
    return float(statistics.stdev(seq))


def write_metrics_tsv(result: EvalResult, path: str | Path) -> None:
    """Write Java-like eval_result.tsv columns to *path*."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "TopN",
        "ResultsOn",
        "Size",
        "Precision",
        "Recall",
        "F1",
        "Coverage",
        "ExactMatch",
        "ParsedCount",
        "ExactMatchCount",
    ]
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for top_n in result.top_ns():
            for split_name, metrics in result.by_top_n[top_n].items():
                writer.writerow(
                    {
                        "TopN": top_n,
                        "ResultsOn": split_name,
                        "Size": metrics.total_count,
                        "Precision": f"{metrics.precision:.4f}",
                        "Recall": f"{metrics.recall:.4f}",
                        "F1": f"{metrics.f1:.4f}",
                        "Coverage": f"{metrics.coverage:.4f}",
                        "ExactMatch": f"{metrics.exact_match:.4f}",
                        "ParsedCount": metrics.parsed_count,
                        "ExactMatchCount": metrics.exact_match_count,
                    },
                )
