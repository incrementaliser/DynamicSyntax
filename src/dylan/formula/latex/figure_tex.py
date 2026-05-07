"""Incremental parse figure: row of trees separated by arrow columns."""

from __future__ import annotations

from dylan.tree.tree import Tree

from dylan.formula.latex.escape import latex_escape_math
from dylan.formula.latex.tree_tex import tree_environment_tex


def trace_figure_tex(
    trees: tuple[Tree, ...],
    step_labels: tuple[str, ...],
) -> str:
    """Lay out *trees* horizontally with ``\\arrow`` columns labeled by *step_labels*.

    ``len(step_labels)`` must equal ``len(trees) - 1`` (one label per consumed word step).
    """
    if len(trees) < 1:
        return ""
    if len(step_labels) != len(trees) - 1:
        raise ValueError(
            f"step_labels length {len(step_labels)} must be len(trees)-1 == {len(trees) - 1}",
        )
    cols: list[str] = []
    for i, tr in enumerate(trees):
        cols.append(tree_environment_tex(tr))
        if i < len(step_labels):
            lab = latex_escape_math(step_labels[i])
            arrow_col = (
                r"\begin{tabular}{c}"
                r"\arrow{1} \\"
                rf"\textrm{{\footnotesize {lab}}}"
                r"\end{tabular}"
            )
            cols.append(arrow_col)
    body = "\n&\n".join(cols)
    ncols = len(cols)
    spec = "c" * ncols
    return "\n".join(
        [
            r"\begin{figure*}[ht]\centering",
            rf"\begin{{tabular}}{{{spec}}}",
            body,
            r"\end{tabular}",
            r"\end{figure*}",
        ],
    )
