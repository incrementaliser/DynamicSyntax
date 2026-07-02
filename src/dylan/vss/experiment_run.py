"""Loguru logging, artifact paths, and analysis reports for VSS paper reproduction runs."""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from dylan.logging_config import configure_logging
from dylan.vss.types import CompositionMethod, EvaluationMode, GS2013EvaluationResult, UnderspecMethod

_VSS_DIR = Path(__file__).resolve().parent
_DEFAULT_RUNS_ROOT = _VSS_DIR / "output" / "runs"

_STAGE_LABELS = ("S", "S-V", "S-V-O")
_COMPOSITION_LABELS: dict[str, str] = {
    CompositionMethod.gs.value: "G&S",
    CompositionMethod.ks.value: "copy-subj",
    CompositionMethod.ko.value: "copy-obj",
    CompositionMethod.baseline.value: "add",
}
_INCR_LABELS: dict[str, str] = {
    UnderspecMethod.identity.value: "identity",
    UnderspecMethod.sum.value: "sum",
    UnderspecMethod.directsum.value: "directsum",
}
_COMPOSITIONAL = frozenset(
    {CompositionMethod.gs.value, CompositionMethod.ks.value, CompositionMethod.ko.value}
)


@dataclass(frozen=True, slots=True)
class ExperimentRunConfig:
    """Configuration for one VSS experiment reproduction run."""

    output_dir: Path
    log_level: str = "INFO"
    run_id: str | None = None
    save_json: bool = True
    save_csv: bool = True
    save_report: bool = True
    log_to_stderr: bool = True


@dataclass
class ExperimentRunContext:
    """Holds output paths, log sink ids, and timing for a single experiment run."""

    config: ExperimentRunConfig
    run_id: str
    started_at: datetime
    log_path: Path
    json_path: Path
    report_path: Path
    csv_path: Path
    plots_dir: Path
    _sink_ids: list[int] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        output_dir: Path | None = None,
        *,
        log_level: str = "INFO",
        run_id: str | None = None,
    ) -> ExperimentRunContext:
        """Create a timestamped run directory under *output_dir* or the default runs root."""
        started = datetime.now(timezone.utc)
        rid = run_id or started.strftime("%Y%m%d-%H%M%S")
        base = output_dir if output_dir is not None else _DEFAULT_RUNS_ROOT
        run_dir = (base / rid).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = ExperimentRunConfig(output_dir=run_dir, log_level=log_level, run_id=rid)
        return cls(
            config=cfg,
            run_id=rid,
            started_at=started,
            log_path=run_dir / "run.log",
            json_path=run_dir / "results.json",
            report_path=run_dir / "analysis_report.md",
            csv_path=run_dir / "accuracy_table.csv",
            plots_dir=run_dir / "plots",
        )

    def setup_logging(self) -> None:
        """Configure loguru: stderr (optional) plus a dedicated run log file."""
        configure_logging(level=self.config.log_level)
        for sid in self._sink_ids:
            try:
                logger.remove(sid)
            except ValueError:
                pass
        self._sink_ids.clear()
        fmt_file = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}"
        )
        fmt_stderr = "<level>{level:<8}</level> | <cyan>{name}</cyan>: {message}"
        self._sink_ids.append(
            logger.add(
                self.log_path,
                level=self.config.log_level,
                format=fmt_file,
                encoding="utf-8",
                enqueue=True,
            )
        )
        if self.config.log_to_stderr:
            self._sink_ids.append(
                logger.add(sys.stderr, level=self.config.log_level, format=fmt_stderr)
            )
        logger.bind(run_id=self.run_id).info("Experiment run logging initialized: {}", self.log_path)

    def teardown_logging(self) -> None:
        """Remove run-specific loguru sinks."""
        for sid in self._sink_ids:
            try:
                logger.remove(sid)
            except ValueError:
                pass
        self._sink_ids.clear()

    def record_timing(self, label: str, seconds: float) -> None:
        """Store a named timing entry for the analysis report."""
        self.timings[label] = seconds

    def plots_path_for(self, mode: str, stem: str = "accuracy") -> Path:
        """Return a plot path under ``plots/`` for *mode*, creating the directory."""
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        return self.plots_dir / f"{stem}_{mode}.png"


def result_to_dict(result: GS2013EvaluationResult) -> dict[str, Any]:
    """Serialize evaluation counts for JSON export."""
    grid: dict[str, Any] = {}
    for im_key, comp_grid in result.by_incremental.items():
        grid[im_key] = {}
        for cm_key, stages in comp_grid.items():
            grid[im_key][cm_key] = [
                {
                    "stage": _STAGE_LABELS[i],
                    "total": s.total,
                    "correct": s.correct,
                    "incorrect": s.incorrect,
                    "accuracy": s.accuracy,
                }
                for i, s in enumerate(stages)
            ]
    return {
        "mode": result.mode.value,
        "metadata": result.metadata,
        "accuracy": grid,
    }


def format_accuracy_table(result: GS2013EvaluationResult) -> str:
    """Format results like jolli ``testSentences`` (incremental x composition x stage)."""
    lines: list[str] = []
    lines.append(f"mode: {result.mode.value}")
    meta = result.metadata
    lines.append(
        f"pairs_total={meta.get('pairs_total', '?')} "
        f"skipped={meta.get('pairs_skipped', 0)} "
        f"parse_failures={meta.get('parse_failures', 0)}"
    )
    lines.append("")
    lines.append(
        f"{'underspec':<12} {'composition':<12} {'stage':<6} "
        f"{'total':>8} {'correct':>8} {'wrong':>8} {'accuracy':>10}"
    )
    lines.append("-" * 72)
    for im_key, comp_grid in sorted(result.by_incremental.items()):
        for cm_key, stages in sorted(comp_grid.items()):
            for stage_idx, acc in enumerate(stages):
                lines.append(
                    f"{_INCR_LABELS.get(im_key, im_key):<12} "
                    f"{_COMPOSITION_LABELS.get(cm_key, cm_key):<12} "
                    f"{_STAGE_LABELS[stage_idx]:<6} "
                    f"{acc.total:8.0f} {acc.correct:8.0f} {acc.incorrect:8.0f} "
                    f"{acc.accuracy:10.4f}"
                )
    return "\n".join(lines)


def best_methods_summary(result: GS2013EvaluationResult) -> str:
    """Summarise best accuracy per incremental stage."""
    lines: list[str] = ["", "Best per stage (composition / underspec):"]
    for stage_idx, stage_name in enumerate(_STAGE_LABELS):
        best_acc = -1.0
        best_label = ""
        for im_key, comp_grid in result.by_incremental.items():
            for cm_key, stages in comp_grid.items():
                acc = stages[stage_idx].accuracy
                if acc > best_acc:
                    best_acc = acc
                    best_label = (
                        f"{_INCR_LABELS.get(im_key, im_key)}/"
                        f"{_COMPOSITION_LABELS.get(cm_key, cm_key)}"
                    )
        lines.append(f"  {stage_name}: {best_label} ({best_acc:.4f})")
    return "\n".join(lines)


def _analysis_section_compositional_vs_baseline(result: GS2013EvaluationResult) -> list[str]:
    """Compare best compositional method vs additive baseline at S-V-O."""
    lines: list[str] = ["", "### Compositional vs additive (S-V-O)", ""]
    best_comp = 0.0
    best_comp_label = ""
    baseline_acc = 0.0
    stage_idx = 2
    for im_key, comp_grid in result.by_incremental.items():
        for cm_key, stages in comp_grid.items():
            acc = stages[stage_idx].accuracy
            if cm_key == CompositionMethod.baseline.value:
                if acc > baseline_acc:
                    baseline_acc = acc
            elif cm_key in _COMPOSITIONAL and acc > best_comp:
                best_comp = acc
                best_comp_label = f"{_INCR_LABELS.get(im_key, im_key)}/{_COMPOSITION_LABELS.get(cm_key, cm_key)}"
    delta = best_comp - baseline_acc
    lines.append(f"- Best compositional: **{best_comp_label}** ({best_comp:.4f})")
    lines.append(f"- Additive baseline (best underspec): **{baseline_acc:.4f}**")
    lines.append(f"- Delta (compositional - add): **{delta:+.4f}**")
    if delta > 0:
        lines.append("- Compositional methods outperform the additive baseline at S-V-O.")
    elif delta < 0:
        lines.append("- Additive baseline matches or beats compositional methods at S-V-O.")
    else:
        lines.append("- Compositional and additive accuracies tie at S-V-O.")
    return lines


def _analysis_section_incrementality(result: GS2013EvaluationResult) -> list[str]:
    """Note methods whose accuracy rises from S-V to S-V-O (paper discussion)."""
    lines: list[str] = ["", "### Incremental trend (S-V -> S-V-O)", ""]
    for im_key, comp_grid in sorted(result.by_incremental.items()):
        for cm_key, stages in sorted(comp_grid.items()):
            sv = stages[1].accuracy
            svo = stages[2].accuracy
            trend = svo - sv
            label = f"{_INCR_LABELS.get(im_key, im_key)}/{_COMPOSITION_LABELS.get(cm_key, cm_key)}"
            direction = "up" if trend > 0.01 else ("down" if trend < -0.01 else "flat")
            lines.append(f"- {label}: {sv:.4f} -> {svo:.4f} ({direction}, {trend:+.4f})")
    return lines


def build_analysis_report(
    results: dict[str, GS2013EvaluationResult],
    ctx: ExperimentRunContext,
    *,
    extra_sections: list[str] | None = None,
    cli_args: dict[str, Any] | None = None,
) -> str:
    """Build a Markdown analysis report for all evaluation modes in *results*."""
    lines: list[str] = [
        "# DS-VSS GS2013 reproduction report",
        "",
        f"- **Run id:** `{ctx.run_id}`",
        f"- **Started (UTC):** {ctx.started_at.isoformat()}",
        f"- **Output directory:** `{ctx.config.output_dir}`",
        "",
        "## Timings (seconds)",
        "",
    ]
    for label, sec in sorted(ctx.timings.items()):
        lines.append(f"- {label}: {sec:.2f}")
    if cli_args:
        lines.extend(["", "## Run configuration", "", "```json", json.dumps(cli_args, indent=2), "```"])
    for mode_key, result in results.items():
        lines.extend(
            [
                "",
                f"## Results: {mode_key}",
                "",
                "```",
                format_accuracy_table(result),
                "```",
                "",
                best_methods_summary(result).replace("Best per stage", "### Best per stage"),
            ]
        )
        lines.extend(_analysis_section_compositional_vs_baseline(result))
        lines.extend(_analysis_section_incrementality(result))
        meta = result.metadata
        lines.extend(
            [
                "",
                "### Run metadata",
                "",
                f"- pairs_total: {meta.get('pairs_total', '?')}",
                f"- pairs_skipped: {meta.get('pairs_skipped', 0)}",
                f"- parse_failures: {meta.get('parse_failures', 0)}",
                "",
                "> Stage **S** should be near **0.50** (chance). Paper Sec. 5 compares S-V and S-V-O.",
            ]
        )
    if extra_sections:
        lines.extend(["", "## Additional notes", ""])
        lines.extend(extra_sections)
    lines.append("")
    return "\n".join(lines)


def write_accuracy_csv(
    results: dict[str, GS2013EvaluationResult],
    path: Path,
) -> None:
    """Write a flat CSV of all accuracy cells for spreadsheet analysis."""
    fieldnames = [
        "mode",
        "underspec",
        "composition",
        "stage",
        "total",
        "correct",
        "incorrect",
        "accuracy",
    ]
    rows: list[dict[str, Any]] = []
    for mode_key, result in results.items():
        for im_key, comp_grid in result.by_incremental.items():
            for cm_key, stages in comp_grid.items():
                for stage_idx, acc in enumerate(stages):
                    rows.append(
                        {
                            "mode": mode_key,
                            "underspec": im_key,
                            "composition": cm_key,
                            "stage": _STAGE_LABELS[stage_idx],
                            "total": acc.total,
                            "correct": acc.correct,
                            "incorrect": acc.incorrect,
                            "accuracy": acc.accuracy,
                        }
                    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_run_artifacts(
    ctx: ExperimentRunContext,
    results: dict[str, GS2013EvaluationResult],
    *,
    extra_report_sections: list[str] | None = None,
    cli_args: dict[str, Any] | None = None,
) -> None:
    """Persist JSON, CSV, Markdown report, and log summary for *results*."""
    cfg = ctx.config
    if cfg.save_json:
        payload = {
            "run_id": ctx.run_id,
            "started_at": ctx.started_at.isoformat(),
            "timings": ctx.timings,
            "cli_args": cli_args or {},
            "results": {k: result_to_dict(v) for k, v in results.items()},
        }
        ctx.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote JSON results: {}", ctx.json_path)
    if cfg.save_csv:
        write_accuracy_csv(results, ctx.csv_path)
        logger.info("Wrote CSV table: {}", ctx.csv_path)
    if cfg.save_report:
        report = build_analysis_report(
            results,
            ctx,
            extra_sections=extra_report_sections,
            cli_args=cli_args,
        )
        ctx.report_path.write_text(report, encoding="utf-8")
        logger.info("Wrote analysis report: {}", ctx.report_path)


def log_evaluation_progress(
    pair_index: int,
    pair_total: int,
    *,
    skipped: int,
    every: int = 100,
) -> None:
    """Emit periodic progress lines during long GS2013 evaluation loops."""
    if pair_total <= 0:
        return
    if pair_index % every != 0 and pair_index != pair_total:
        return
    logger.info(
        "Evaluation progress: {}/{} pairs (skipped so far: {})",
        pair_index,
        pair_total,
        skipped,
    )
