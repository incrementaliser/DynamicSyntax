"""Incremental parse figure: trees stacked with the step that produced each one."""

from __future__ import annotations

from dylan.tree.tree import Tree

from dylan.formula.latex.escape import latex_escape_math
from dylan.formula.latex.tree_tex import tree_environment_tex


def _step_row(label: str) -> str:
    """Return a centred arrow row whose caption is the actions and word for one step."""
    safe = latex_escape_math(label)
    return "\n".join(
        [
            r"\begin{tabular}{c}",
            r"\parbox{0.92\linewidth}{\centering\footnotesize " + safe + r"}\\[0.35em]",
            r"$\Rightarrow$",
            r"\end{tabular}",
        ],
    )


def trace_figure_tex(
    trees: tuple[Tree, ...],
    step_labels: tuple[str, ...],
    *,
    sentence: str = "",
) -> str:
    """Lay out *trees* vertically, with an arrow caption between snapshots.

    ``len(step_labels)`` must equal ``len(trees) - 1`` (one label per consumed word).
    Side-by-side rows overflow the page once node formulae are real records, so each
    snapshot is its own row. *sentence* is used in the caption when given.
    """
    if len(trees) < 1:
        return ""
    if len(step_labels) != len(trees) - 1:
        raise ValueError(
            f"step_labels length {len(step_labels)} must be len(trees)-1 == {len(trees) - 1}",
        )
    rows: list[str] = [tree_environment_tex(trees[0])]
    for index, label in enumerate(step_labels):
        rows.append(_step_row(label))
        rows.append(tree_environment_tex(trees[index + 1]))
    body = "\n\\\\\n".join(rows)
    caption = ""
    if sentence.strip():
        shown = latex_escape_math(sentence.strip())
        caption = r"\caption{Incremental DS-TTR parse of \emph{" + shown + "}}"
    parts = [
        r"\begin{figure*}[ht]\centering",
        r"\begin{tabular}{c}",
        body,
        r"\end{tabular}",
    ]
    if caption:
        parts.append(caption)
    parts.append(r"\end{figure*}")
    return "\n".join(parts)
