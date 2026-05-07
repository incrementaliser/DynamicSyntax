"""Run ``latexmk`` / ``pdflatex`` on a directory that already contains ``main.tex`` and ``*.sty``."""

from __future__ import annotations

import shutil
import subprocess
from importlib import resources
from pathlib import Path


def copy_latex_assets(dest_dir: Path) -> None:
    """Copy shipped ``*.sty`` files next to the document under *dest_dir*."""
    root = resources.files("dylan.formula.latex")
    for name in ("dsttr.sty", "rtrees.sty", "avm.sty"):
        node = root / name
        if node.is_file():
            dest = dest_dir / name
            dest.write_bytes(node.read_bytes())


def compile_main_tex(
    work_dir: Path,
    *,
    main_name: str = "main.tex",
    pdf_name: str = "main.pdf",
) -> tuple[int, Path | None]:
    """Compile *work_dir* / *main_name*; return ``(exit_code, pdf_path_if_ok)``."""
    main = work_dir / main_name
    if not main.is_file():
        raise FileNotFoundError(f"missing LaTeX main file: {main}")
    latexmk = shutil.which("latexmk")
    if latexmk is not None:
        cmd = [
            latexmk,
            "-pdf",
            "-interaction=nonstopmode",
            str(main),
        ]
        proc = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        pdf = work_dir / pdf_name
        if proc.returncode == 0 and pdf.is_file():
            return proc.returncode, pdf
        return proc.returncode, None
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise FileNotFoundError("neither latexmk nor pdflatex found on PATH")
    cmd = [pdflatex, "-interaction=nonstopmode", f"-output-directory={work_dir}", str(main)]
    last = 0
    for _ in range(3):
        proc = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True, check=False)
        last = proc.returncode
        if proc.returncode != 0:
            break
    pdf = work_dir / pdf_name
    if last == 0 and pdf.is_file():
        return last, pdf
    return last, None


def pdf_to_png(pdf_path: Path, png_path: Path) -> bool:
    """Rasterise first page of *pdf_path* to *png_path* using ``pdftoppm`` or ImageMagick."""
    ppm = shutil.which("pdftoppm")
    if ppm is not None:
        stem = png_path.with_suffix("")
        proc = subprocess.run(
            [ppm, "-png", "-singlefile", str(pdf_path), str(stem)],
            capture_output=True,
            text=True,
            check=False,
        )
        candidate = stem.with_suffix(".png")
        if proc.returncode == 0 and candidate.is_file():
            if candidate.resolve() != png_path.resolve():
                shutil.move(str(candidate), png_path)
            return True
    magick = shutil.which("magick")
    if magick is not None:
        proc = subprocess.run(
            [magick, str(pdf_path) + "[0]", str(png_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode == 0 and png_path.is_file()
    convert = shutil.which("convert")
    if convert is not None:
        proc = subprocess.run(
            [convert, "-density", "150", str(pdf_path) + "[0]", str(png_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode == 0 and png_path.is_file()
    return False
