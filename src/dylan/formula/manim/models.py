"""Structured return values for Manim scene generation / rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManimBuildResult:
    """Generated Manim scene source plus optional render output."""

    scene_code: str
    scene_path: Path | None
    video_path: Path | None
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""

