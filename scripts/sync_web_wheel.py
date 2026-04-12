#!/usr/bin/env python3
"""Copy the latest ``dynamicsyntax-*.whl`` from ``dist/`` to ``web/dist/package.whl`` for the Pyodide shell."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> None:
    """Find the newest wheel under *dist/* and copy it to *web/dist/package.whl*."""
    root = Path(__file__).resolve().parents[1]
    dist_dir = root / "dist"
    wheels = sorted(
        dist_dir.glob("dynamicsyntax-*.whl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not wheels:
        print("No dynamicsyntax-*.whl in dist/; run `uv build` first.", file=sys.stderr)
        raise SystemExit(1)
    out = root / "web" / "dist" / "package.whl"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(wheels[0], out)
    print(f"Copied {wheels[0].name} -> {out}")


if __name__ == "__main__":
    main()
