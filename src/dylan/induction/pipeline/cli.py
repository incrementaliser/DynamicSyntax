"""CLI entry point for the TTR induction train/evaluate pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dylan.induction.pipeline.config import load_config
from dylan.induction.pipeline.runner import run_induction


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for ``dsttr-induction``."""
    parser = argparse.ArgumentParser(
        description="Run a YAML-configured TTR induction train/evaluate pipeline.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to induction YAML config",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config field with a dotted path (repeatable), e.g. data.seed=48",
    )
    parser.add_argument(
        "--report-tui",
        action="store_true",
        help="Open a Textual TUI to browse the end report after the run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load config, run induction, and return a process exit code."""
    args = _parse_args(argv)
    if not args.config.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1
    try:
        config = load_config(args.config, overrides=args.overrides or None)
        result = run_induction(config, report_tui=args.report_tui)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    run_dir = result.metadata.get("run_dir", "")
    if run_dir:
        print(f"Done. Artifacts in {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
