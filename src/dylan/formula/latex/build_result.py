"""Structured return value for LaTeX generation and optional compilation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LaTeXBuildResult:
    """Full LaTeX source plus optional paths after ``compile=True``."""

    tex: str
    tex_path: Path | None
    pdf_path: Path | None
    png_path: Path | None
    exit_code: int | None
