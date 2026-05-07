"""Wrap fragment LaTeX in a standalone ``article`` document with DS-TTR style files."""

from __future__ import annotations

def latex_preamble(*, include_dsttr: bool = True) -> str:
    """Return shared preamble lines (fallback ``\\arrow``, geometry, ``dsttr`` bundle)."""
    lines = [
        r"\usepackage[a4paper,margin=1cm]{geometry}",
        r"\usepackage{graphicx}",
        r"\providecommand{\arrow}[1]{$\Rightarrow$}",
    ]
    if include_dsttr:
        lines.append(r"\input{dsttr.sty}")
    return "\n".join(lines)


def build_standalone_document(body: str, *, title: str, include_dsttr: bool = True) -> str:
    """Return a full ``article`` document string embedding *body*."""
    esc_title = title.replace("\\", r"\textbackslash{}").replace("&", r"\&")
    pre = latex_preamble(include_dsttr=include_dsttr)
    return "\n".join(
        [
            r"\documentclass{article}",
            pre,
            r"\begin{document}",
            rf"\section*{{{esc_title}}}",
            body,
            r"\end{document}",
        ],
    )

