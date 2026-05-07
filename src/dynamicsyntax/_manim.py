"""Convenience facade for parsing a sentence and producing a Manim animation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from dylan.nlp.types import DEFAULT_SPEAKER
from dylan.formula.manim.models import ManimBuildResult

from dynamicsyntax._parse import parse


def to_manim(
    sentence: str,
    grammar: str | Path | None = None,
    /,
    *,
    speaker: str = DEFAULT_SPEAKER,
    output_path: Path | None = None,
    write_scene: Path | None = None,
    quality: Literal["l", "m", "h", "p", "k"] = "m",
    preview: bool = False,
    render: bool = True,
    class_name: str | None = None,
) -> ManimBuildResult:
    """Parse *sentence* with action tracing and render (or return) a generated Manim scene."""
    result = parse(sentence, grammar, speaker=speaker, trace=True)
    return result.to_manim(
        output_path=output_path,
        write_scene=write_scene,
        quality=quality,
        preview=preview,
        render=render,
        class_name=class_name,
    )

