"""End-of-run Rich report and optional Textual TUI."""

from __future__ import annotations

from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from dylan.induction.pipeline.config import InductionConfig
from dylan.induction.pipeline.metrics import EvalResult
from dylan.induction.pipeline.timing import format_hh_mm_ss


def _scores_table(result: EvalResult, title: str) -> Table:
    """Build a Rich table of P/R/F1/coverage/EM for all splits × top-N."""
    splits = result.split_names()
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Top-N", justify="right")
    for split in splits:
        table.add_column(f"{split} P", justify="right")
        table.add_column(f"{split} R", justify="right")
        table.add_column(f"{split} F1", justify="right")
        table.add_column(f"{split} Cov", justify="right")
        table.add_column(f"{split} EM", justify="right")

    for top_n in result.top_ns():
        row: list[str] = [str(top_n)]
        for split in splits:
            m = result.get(top_n, split)
            if m is None:
                row.extend(["—"] * 5)
            else:
                row.extend(
                    [
                        f"{m.precision:.2f}",
                        f"{m.recall:.2f}",
                        f"{m.f1:.2f}",
                        f"{m.coverage:.2f}",
                        f"{m.exact_match:.2f}",
                    ],
                )
        table.add_row(*row)
    return table


def _config_text(config: InductionConfig) -> str:
    """Format resolved config as YAML text."""
    return yaml.safe_dump(config.to_dict(), sort_keys=False, default_flow_style=False)


def _metadata_text(result: EvalResult) -> str:
    """Format run metadata (including timings) for the end of the report."""
    meta = result.metadata
    lines: list[str] = []
    for key in (
        "run_dir",
        "split",
        "seed",
        "git_commit",
        "dataset_sizes",
        "train_time_s",
        "eval_time_s",
        "train_time",
        "eval_time",
        "n_folds",
    ):
        if key not in meta:
            continue
        val = meta[key]
        if key.endswith("_time_s") and isinstance(val, (int, float)):
            lines.append(f"{key}: {val:.3f} ({format_hh_mm_ss(float(val))})")
        else:
            lines.append(f"{key}: {val}")
    # Any remaining simple keys
    skip = {
        "std",
        "run_dir",
        "split",
        "seed",
        "git_commit",
        "dataset_sizes",
        "train_time_s",
        "eval_time_s",
        "train_time",
        "eval_time",
        "n_folds",
    }
    for key, val in meta.items():
        if key in skip or isinstance(val, (dict, list)):
            continue
        if f"{key}:" in "\n".join(lines):
            continue
        lines.append(f"{key}: {val}")
    return "\n".join(lines) if lines else "(none)"


def build_report_text(result: EvalResult, config: InductionConfig) -> str:
    """Return a plain-text report (tables + config + metadata) for ``report.txt``."""
    console = Console(record=True, width=120, force_terminal=False)
    _print_report(console, result, config)
    return console.export_text()


def print_report(
    result: EvalResult,
    config: InductionConfig,
    *,
    console: Console | None = None,
) -> None:
    """Print the Rich report to the given (or default) console."""
    _print_report(console or Console(), result, config)


def _print_report(console: Console, result: EvalResult, config: InductionConfig) -> None:
    """Shared Rich rendering for CLI and text export."""
    console.print()
    console.print(Panel(Text("Induction report", style="bold"), expand=False))

    train_s = result.metadata.get("train_time_s")
    eval_s = result.metadata.get("eval_time_s")
    if isinstance(train_s, (int, float)) or isinstance(eval_s, (int, float)):
        timing = Table(title="Timing", show_header=True)
        timing.add_column("Phase")
        timing.add_column("HH-MM-SS", justify="right")
        timing.add_column("Seconds", justify="right")
        if isinstance(train_s, (int, float)):
            timing.add_row("Train", format_hh_mm_ss(float(train_s)), f"{float(train_s):.2f}")
        if isinstance(eval_s, (int, float)):
            timing.add_row("Eval", format_hh_mm_ss(float(eval_s)), f"{float(eval_s):.2f}")
        console.print(timing)

    console.print(_scores_table(result, "Scores (percentages)"))
    if result.fold_results:
        console.print(
            f"\n[dim]Averaged over {len(result.fold_results)} folds "
            f"(per-fold details under run_dir/fold_*).[/dim]",
        )
        std = result.metadata.get("std")
        if isinstance(std, dict) and std:
            std_table = Table(title="Std-dev across folds (F1 / Coverage / EM)", show_header=True)
            std_table.add_column("Top-N", justify="right")
            std_table.add_column("Split")
            std_table.add_column("F1 σ", justify="right")
            std_table.add_column("Cov σ", justify="right")
            std_table.add_column("EM σ", justify="right")
            for top_n, split_map in sorted(std.items()):
                for split_name, vals in split_map.items():
                    std_table.add_row(
                        str(top_n),
                        str(split_name),
                        f"{vals.get('f1', 0):.2f}",
                        f"{vals.get('coverage', 0):.2f}",
                        f"{vals.get('exact_match', 0):.2f}",
                    )
            console.print(std_table)
    console.print()
    console.print(Panel(_config_text(config), title="Resolved config", border_style="cyan"))
    console.print()
    console.print(Panel(_metadata_text(result), title="Metadata", border_style="green"))


def write_report_file(
    result: EvalResult,
    config: InductionConfig,
    path: str | Path,
) -> None:
    """Write the plain-text report to *path*."""
    Path(path).write_text(build_report_text(result, config), encoding="utf-8")


def launch_report_tui(result: EvalResult, config: InductionConfig) -> None:
    """Open a simple Textual app to browse the report (blocks until quit)."""
    from textual.app import App, ComposeResult
    from textual.widgets import Footer, Header, Static

    report = build_report_text(result, config)

    class ReportApp(App[None]):
        """Minimal Textual viewer for the induction report."""

        BINDINGS = [("q", "quit", "Quit")]

        def compose(self) -> ComposeResult:
            """Lay out header, scrollable report, and footer."""
            yield Header()
            yield Static(report, id="report")
            yield Footer()

    ReportApp().run()
