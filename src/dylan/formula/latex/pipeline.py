"""Assemble standalone TeX and optional ``latexmk`` / PNG pipeline."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from dylan.formula.latex.build_result import LaTeXBuildResult
from dylan.formula.latex.compile import compile_main_tex, copy_latex_assets, pdf_to_png
from dylan.formula.latex.document import build_standalone_document


def _default_pdf_destination(
    write_tex: Path | None,
    image_path: Path | None,
) -> Path:
    """Pick a persistent path for the compiled PDF when the caller does not set one."""
    if write_tex is not None:
        return write_tex.with_suffix(".pdf")
    if image_path is not None:
        return image_path.with_suffix(".pdf")
    fd, name = tempfile.mkstemp(prefix="dynamicsyntax_", suffix=".pdf")
    os.close(fd)
    return Path(name)


def run_latex_pipeline(
    body: str,
    *,
    title: str,
    write_tex: Path | None,
    do_compile: bool,
    image_path: Path | None,
    pdf_out: Path | None,
) -> LaTeXBuildResult:
    """Build full document from *body*, optionally write/compile and rasterise to PNG."""
    full_tex = build_standalone_document(body, title=title)
    tex_path: Path | None = None
    pdf_path: Path | None = None
    png_path: Path | None = None
    exit_code: int | None = None

    if write_tex is not None:
        write_tex.parent.mkdir(parents=True, exist_ok=True)
        write_tex.write_text(full_tex, encoding="utf-8")
        tex_path = write_tex

    if not do_compile:
        return LaTeXBuildResult(
            tex=full_tex,
            tex_path=tex_path,
            pdf_path=None,
            png_path=None,
            exit_code=None,
        )

    dest_pdf = pdf_out if pdf_out is not None else _default_pdf_destination(write_tex, image_path)
    dest_pdf = dest_pdf.resolve()
    dest_pdf.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        copy_latex_assets(work)
        main = work / "main.tex"
        main.write_text(full_tex, encoding="utf-8")
        code, pdf = compile_main_tex(work)
        exit_code = code
        if pdf is None or not pdf.is_file():
            raise RuntimeError(
                f"LaTeX compilation failed (exit {code}); "
                "install latexmk or pdflatex and ensure dsttr/rtrees dependencies resolve.",
            )
        shutil.copy(pdf, dest_pdf)
        pdf_path = dest_pdf

        if image_path is not None:
            image_path = image_path.resolve()
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if image_path.suffix.lower() == ".png":
                if not pdf_to_png(dest_pdf, image_path):
                    raise RuntimeError(
                        "PNG export requested but neither pdftoppm nor magick/convert succeeded.",
                    )
                png_path = image_path
            elif image_path.suffix.lower() == ".pdf":
                shutil.copy(dest_pdf, image_path)
                pdf_path = image_path

    return LaTeXBuildResult(
        tex=full_tex,
        tex_path=tex_path,
        pdf_path=pdf_path,
        png_path=png_path,
        exit_code=exit_code,
    )
