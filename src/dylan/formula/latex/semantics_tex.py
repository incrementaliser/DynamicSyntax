"""LaTeX body for final TTR semantics (uses :meth:`~dylan.formula.ttr_record_type.TTRRecordType.to_latex`)."""

from __future__ import annotations

from dylan.formula.ttr_record_type import TTRRecordType


def semantics_figure_tex(rt: TTRRecordType) -> str:
    """Return a centered display-math block for record *rt*."""
    inner = rt.to_latex()
    return "\n".join([r"\begin{center}", r"\[" + inner + r"\]", r"\end{center}"])
