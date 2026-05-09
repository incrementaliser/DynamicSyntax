"""Parse outcome wrapper for the :mod:`dynamicsyntax` facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TextIO

from dylan.action.lexicon import NotebookMultilineText
from dylan.formula.latex.build_result import LaTeXBuildResult
from dylan.formula.latex.figure_tex import trace_figure_tex
from dylan.formula.latex.pipeline import run_latex_pipeline
from dylan.formula.latex.semantics_tex import semantics_figure_tex
from dylan.formula.latex.tree_tex import tree_to_rtrees_tex
from dylan.formula.manim.models import ManimBuildResult
from dylan.formula.manim.render import render_manim_scene
from dylan.formula.manim.template import build_manim_scene_code, scene_class_name
from dylan.formula.manim.tree_scene import serialize_action_steps
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.gui.formatting import format_ds_tree
from dylan.parser.interactive_context_parser import InteractiveContextParser
from dylan.tree.tree import Tree
from dynamicsyntax.parse_trace import ParseActionStep


@dataclass(frozen=True)
class ParseResult:
    """Outcome of :func:`dynamicsyntax.parse` or :meth:`InteractiveContextParser.parse`: semantics, tree, trace, optional parser."""

    ok: bool
    semantics: TTRRecordType | None
    tree: Tree | None
    sentence: str = ""
    trace_trees: tuple[Tree, ...] = field(default_factory=tuple)
    trace_step_labels: tuple[str, ...] = field(default_factory=tuple)
    action_steps: tuple[ParseActionStep, ...] = field(default_factory=tuple)
    parser: InteractiveContextParser | None = None  # set by parse() when a parser ran (not blank-only skips)

    @property
    def address_order(self) -> str:
        """Address-ordered tree text (same panel as the Flet GUI ``address_order`` view)."""
        if self.tree is None:
            return ""
        return format_ds_tree(self.tree)

    def vis(self) -> None:
        """Print the address-order parse tree (GUI ``address_order`` panel); no-op message if no tree."""
        if self.tree is None:
            print("(no parse tree)")
            return
        print(format_ds_tree(self.tree))

    def get_vocab(
        self,
        groupby: Literal["category", "alpha"] = "category",
        *,
        stream: TextIO | None = None,
        backend: Literal["plain", "rich"] = "plain",
        max_cell_width: int | None = 120,
    ) -> NotebookMultilineText:
        """Format loaded lexical entries via :meth:`~dylan.parser.dag_parser.DAGParser.get_vocab` on the parse parser."""
        if self.parser is None:
            raise ValueError(
                "ParseResult has no parser (whitespace-only input skips loading a grammar); "
                "call parse(...) with text or use InteractiveContextParser/Lexicon.get_vocab.",
            )
        return self.parser.get_vocab(
            groupby,
            stream=stream,
            backend=backend,
            max_cell_width=max_cell_width,
        )

    def to_latex(
        self,
        kind: Literal["semantics", "tree", "incremental"],
        *,
        title: str | None = None,
        write_tex: Path | None = None,
        compile_tex: bool = False,
        image_path: Path | None = None,
        pdf_out: Path | None = None,
    ) -> LaTeXBuildResult:
        """Build standalone LaTeX (and optionally PDF/PNG) for semantics, final tree, or incremental trace.

        :param kind: ``\"semantics\"`` uses :meth:`~dylan.formula.ttr_record_type.TTRRecordType.to_latex`;
            ``\"tree\"`` emits one ``rtrees`` figure; ``\"incremental\"`` needs non-empty ``trace_trees``
            from :func:`dynamicsyntax.parse` with ``trace=True``.
        :param title: Section title in the wrapper document (default from *kind*).
        :param write_tex: If set, write the full ``.tex`` source to this path.
        :param compile_tex: If ``True``, run ``latexmk`` or ``pdflatex`` (requires local TeX).
        :param image_path: When set with ``compile_tex=True``, write ``.png`` or ``.pdf`` from the PDF.
        :param pdf_out: Optional explicit output path for the compiled PDF.
        :raises ValueError: When *kind* is incompatible with available data.
        :raises RuntimeError: When ``compile_tex=True`` but the toolchain fails.
        """
        cap = title or f"dynamicsyntax — {kind}"
        if kind == "semantics":
            if not self.ok or self.semantics is None:
                raise ValueError("semantics export requires a successful parse with non-None semantics")
            body = semantics_figure_tex(self.semantics)
        elif kind == "tree":
            if self.tree is None:
                raise ValueError("tree export requires a non-None parse tree")
            body = tree_to_rtrees_tex(self.tree)
        elif kind == "incremental":
            if len(self.trace_trees) < 2:
                raise ValueError(
                    'incremental export requires trace_trees from parse(..., trace=True); '
                    "got fewer than two snapshots",
                )
            if len(self.trace_step_labels) != len(self.trace_trees) - 1:
                raise ValueError("trace_step_labels length must match len(trace_trees) - 1")
            body = trace_figure_tex(self.trace_trees, self.trace_step_labels)
        else:
            raise ValueError(f"unknown kind {kind!r}")
        return run_latex_pipeline(
            body,
            title=cap,
            write_tex=write_tex,
            do_compile=compile_tex,
            image_path=image_path,
            pdf_out=pdf_out,
        )

    def to_manim(
        self,
        *,
        output_path: Path | None = None,
        write_scene: Path | None = None,
        quality: Literal["l", "m", "h", "p", "k"] = "m",
        preview: bool = False,
        render: bool = True,
        class_name: str | None = None,
    ) -> ManimBuildResult:
        """Generate an action-level Manim animation of the parse and optionally render it.

        :param output_path: Optional destination for the rendered ``.mp4``.
        :param write_scene: Optional path to write the generated Python scene source.
        :param quality: Manim quality flag suffix (``l``, ``m``, ``h``, ``p``, or ``k``).
        :param preview: Forward ``-p`` to Manim when rendering.
        :param render: If ``False``, return scene code without invoking Manim.
        :param class_name: Optional Manim scene class name; generated from the sentence by default.
        :raises ValueError: If this result has no action trace (parse with ``trace=True``).
        """
        if not self.action_steps:
            raise ValueError("Manim export requires action_steps; call parse(..., trace=True) first")
        cls = class_name or scene_class_name(self.sentence or "Dynamic Syntax Parse")
        data = serialize_action_steps(
            self.action_steps,
            semantics=self.semantics if self.ok else None,
            sentence=self.sentence,
        )
        scene_code = build_manim_scene_code(data, class_name=cls)
        return render_manim_scene(
            scene_code,
            class_name=cls,
            output_path=output_path,
            write_scene=write_scene,
            quality=quality,
            preview=preview,
            render=render,
        )
