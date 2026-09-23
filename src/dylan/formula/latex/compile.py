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


def latex_log_excerpt(work_dir: Path, *, main_stem: str = "main", limit: int = 40) -> str:
    """Return the last *limit* lines of the LaTeX log in *work_dir*, if any."""
    log = work_dir / f"{main_stem}.log"
    if not log.is_file():
        return ""
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-limit:])


def compile_main_tex(
    work_dir: Path,
    *,
    main_name: str = "main.tex",
    pdf_name: str = "main.pdf",
) -> tuple[int, Path | None]:
    """Compile *work_dir* / *main_name*; return ``(exit_code, pdf_path_if_ok)``.

    DS trees are PSTricks pictures, so the preferred route is ``latexmk -pdfps``
    (latex, dvips, ps2pdf). ``pdflatex`` does not draw those trees.
    """
    main = work_dir / main_name
    if not main.is_file():
        raise FileNotFoundError(f"missing LaTeX main file: {main}")
    pdf = work_dir / pdf_name
    latexmk = shutil.which("latexmk")
    if latexmk is not None:
        cmd = [
            latexmk,
            "-pdfps",
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
        if proc.returncode == 0 and pdf.is_file():
            return proc.returncode, pdf
        return proc.returncode, None
    latex = shutil.which("latex")
    dvips = shutil.which("dvips")
    ps2pdf = shutil.which("ps2pdf")
    if latex is None or dvips is None or ps2pdf is None:
        raise FileNotFoundError(
            "neither latexmk nor latex+dvips+ps2pdf found on PATH "
            "(PSTricks trees need the dvips route)",
        )
    stem = main.stem
    last = 0
    for _ in range(2):
        proc = subprocess.run(
            [latex, "-interaction=nonstopmode", main.name],
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        last = proc.returncode
        if proc.returncode != 0:
            return last, None
    dvi = work_dir / f"{stem}.dvi"
    ps = work_dir / f"{stem}.ps"
    dvips_proc = subprocess.run(
        [dvips, "-o", ps.name, dvi.name],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if dvips_proc.returncode != 0 or not ps.is_file():
        return dvips_proc.returncode or 1, None
    ps_proc = subprocess.run(
        [ps2pdf, ps.name, pdf.name],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if ps_proc.returncode == 0 and pdf.is_file():
        return 0, pdf
    return ps_proc.returncode or 1, None


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
