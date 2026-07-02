"""Tests for VSS experiment logging and report generation."""

from __future__ import annotations

from pathlib import Path

from dylan.vss.evaluate import _init_accuracy_grid
from dylan.vss.experiment_run import (
    ExperimentRunContext,
    build_analysis_report,
    format_accuracy_table,
    result_to_dict,
    save_run_artifacts,
    write_accuracy_csv,
)
from dylan.vss.types import EvaluationMode, GS2013EvaluationResult, MethodAccuracy


def _minimal_result() -> GS2013EvaluationResult:
    """Build a tiny result grid for report tests."""
    grid = _init_accuracy_grid()
    grid["identity"]["gs"][0] = MethodAccuracy(total=10, correct=5, incorrect=5)
    grid["identity"]["gs"][2] = MethodAccuracy(total=10, correct=7, incorrect=3)
    return GS2013EvaluationResult(
        mode=EvaluationMode.tensor_only,
        by_incremental=grid,
        metadata={"pairs_total": 10, "pairs_skipped": 0},
    )


def test_save_run_artifacts(tmp_path: Path) -> None:
    """Report, JSON, and CSV are written under the run directory."""
    ctx = ExperimentRunContext.create(tmp_path / "runs", run_id="test-run")
    ctx.setup_logging()
    results = {"tensor_only": _minimal_result()}
    save_run_artifacts(ctx, results, cli_args={"max_pairs": 10})
    assert ctx.json_path.is_file()
    assert ctx.report_path.is_file()
    assert ctx.csv_path.is_file()
    assert "tensor_only" in ctx.report_path.read_text(encoding="utf-8")
    ctx.teardown_logging()


def test_format_and_dict() -> None:
    """Table formatting and JSON dict include mode and stage labels."""
    result = _minimal_result()
    table = format_accuracy_table(result)
    assert "identity" in table
    assert "S-V-O" in table
    data = result_to_dict(result)
    assert data["mode"] == "tensor_only"
    assert "identity" in data["accuracy"]
