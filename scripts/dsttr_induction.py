"""Thin wrapper so ``uv run python scripts/dsttr_induction.py`` works."""

from __future__ import annotations

from dylan.induction.pipeline.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
