"""Render generated Manim scene code via the Manim Community command line."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import importlib.util
from pathlib import Path

from dylan.formula.manim.models import ManimBuildResult


def _find_rendered_video(media_dir: Path) -> Path | None:
    """Return the newest MP4 emitted under *media_dir*, if any."""
    videos = sorted(media_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return videos[0] if videos else None


def render_manim_scene(
    scene_code: str,
    *,
    class_name: str,
    output_path: Path | None,
    write_scene: Path | None,
    quality: str,
    preview: bool,
    render: bool,
) -> ManimBuildResult:
    """Write generated scene code and optionally render it with ``python -m manim``."""
    scene_path: Path | None = None
    if write_scene is not None:
        write_scene.parent.mkdir(parents=True, exist_ok=True)
        write_scene.write_text(scene_code, encoding="utf-8")
        scene_path = write_scene

    if not render:
        return ManimBuildResult(
            scene_code=scene_code,
            scene_path=scene_path,
            video_path=None,
            exit_code=None,
        )

    manim_exe = shutil.which("manim")
    manim_module = importlib.util.find_spec("manim")
    if manim_exe is None and manim_module is None:
        raise FileNotFoundError("Manim is not installed; install with `uv pip install -e \".[video]\"`")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        generated_scene = work / "dynamic_syntax_manim_scene.py"
        generated_scene.write_text(scene_code, encoding="utf-8")
        media_dir = work / "media"
        cmd = [sys.executable, "-m", "manim"] if manim_module is not None else [str(manim_exe)]
        cmd.extend([f"-q{quality}", "--media_dir", str(media_dir)])
        if preview:
            cmd.append("-p")
        cmd.extend([str(generated_scene), class_name])
        proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True, check=False)
        video = _find_rendered_video(media_dir)
        final_video: Path | None = None
        if video is not None:
            if output_path is None:
                output_path = Path.cwd() / video.name
            output_path = output_path.resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(video, output_path)
            final_video = output_path
        return ManimBuildResult(
            scene_code=scene_code,
            scene_path=scene_path,
            video_path=final_video,
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )

