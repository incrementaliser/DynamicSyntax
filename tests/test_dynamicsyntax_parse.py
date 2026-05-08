"""Smoke tests for the ``dynamicsyntax`` distribution facade."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

import dynamicsyntax as ds
from dylan.formula.ttr_record_type import TTRRecordType


def test_import_dynamicsyntax_version() -> None:
    """Package exposes a PEP 440 version string."""
    assert isinstance(ds.__version__, str)
    assert ds.__version__


def test_get_grammars_includes_bundled_and_alias() -> None:
    """Bundled grammar dir and ``ttr`` alias are listed."""
    g = ds.get_grammars()
    assert "2015-english-ttr" in g
    assert "ttr" in g


def test_get_datasets_empty_placeholder() -> None:
    """No bundled datasets yet."""
    assert ds.get_datasets() == []


def test_parse_ttr_bundled_grammar() -> None:
    """End-to-end parse via bundled ``2015-english-ttr`` grammar."""
    p = ds.parse("a man arrives", "ttr")
    assert p.ok
    assert isinstance(p.semantics, TTRRecordType)
    s = str(p.semantics)
    assert "man(" in s and "arrive" in s
    assert p.tree is not None
    assert "00" in p.address_order
    assert p.parser is not None


def test_parse_empty_returns_failed_result() -> None:
    """Whitespace-only input yields no semantics and no parser reference."""
    p = ds.parse("   ", "ttr")
    assert not p.ok
    assert p.semantics is None
    assert p.parser is None
    with pytest.raises(ValueError, match="no parser"):
        p.get_vocab()


def test_parse_list_empty_returns_empty() -> None:
    """Batch parse of an empty list returns ``[]`` and does not require a loaded grammar."""
    from dynamicsyntax._session import clear_grammar_session

    clear_grammar_session()
    try:
        assert ds.parse([], "ttr") == []
        assert ds.parse([]) == []
    finally:
        ds.load_grammar("ttr")


def test_parse_list_singleton_matches_single() -> None:
    """One-element list with explicit grammar matches single-string parse."""
    one = ds.parse("a man arrives", "ttr")
    batch = ds.parse(["a man arrives"], "ttr")
    assert len(batch) == 1
    assert batch[0].ok == one.ok
    assert batch[0].sentence == one.sentence
    assert str(batch[0].semantics) == str(one.semantics)


def test_parse_list_mixed_blank_slot() -> None:
    """Per-item blank strings yield a failed result for that index only."""
    out = ds.parse(["a man arrives", "   "], "ttr")
    assert len(out) == 2
    assert out[0].ok
    assert not out[1].ok
    assert out[1].semantics is None


def test_parse_result_get_vocab_matches_lexicon() -> None:
    """``ParseResult.get_vocab`` matches ``Lexicon.get_vocab`` for the same grammar directory."""
    from dylan.action.lexicon import Lexicon

    from dynamicsyntax._session import resolved_grammar_path

    p = ds.parse("a man arrives", "ttr")
    assert p.parser is not None
    with resolved_grammar_path("ttr") as path:
        ref = Lexicon(path).get_vocab(groupby="alpha")
    assert p.get_vocab(groupby="alpha") == ref


def test_parse_list_blank_slot_keeps_parser_for_get_vocab() -> None:
    """List parse shares one parser; blank slots still expose ``get_vocab``."""
    out = ds.parse(["a man arrives", "   "], "ttr")
    assert out[0].parser is not None
    assert out[0].parser is out[1].parser
    assert "Load statistics" in out[1].get_vocab()


def test_parse_list_trace_matches_individual() -> None:
    """``trace=True`` on a list records the same trace shape as per-sentence parses."""
    sentences = ["a man arrives", "a man arrives"]
    batch = ds.parse(sentences, "ttr", trace=True)
    assert len(batch) == 2
    for sent, got in zip(sentences, batch):
        ref = ds.parse(sent, "ttr", trace=True)
        assert len(got.trace_trees) == len(ref.trace_trees)
        assert got.trace_step_labels == ref.trace_step_labels
        assert len(got.action_steps) == len(ref.action_steps)


def test_parse_list_session_grammar() -> None:
    """List input works with session grammar from ``load_grammar``."""
    ds.load_grammar("ttr")
    batch = ds.parse(["a man arrives", "a man arrives"])
    assert len(batch) == 2
    assert all(r.ok for r in batch)


def test_parse_unknown_grammar_raises() -> None:
    """Invalid grammar id raises :class:`FileNotFoundError`."""
    with pytest.raises(FileNotFoundError, match="unknown bundled grammar"):
        ds.parse("hello", "not-a-backend")


def test_load_grammar_then_parse() -> None:
    """Session-style parse reuses ``load_grammar``."""
    ds.load_grammar("ttr")
    p = ds.parse("a man arrives")
    assert p.ok
    assert isinstance(p.semantics, TTRRecordType)
    assert p.parser is not None


def test_parse_without_grammar_and_without_load_raises() -> None:
    """Calling ``parse(s)`` with no prior ``load_grammar`` is rejected."""
    from dynamicsyntax._session import clear_grammar_session

    clear_grammar_session()
    try:
        with pytest.raises(ValueError, match="no grammar loaded"):
            ds.parse("hello")
    finally:
        ds.load_grammar("ttr")


def test_vis_prints_address_order(capsys: pytest.CaptureFixture[str]) -> None:
    """``ParseResult.vis`` prints the GUI address-order tree text."""
    p = ds.parse("a man arrives", "ttr")
    p.vis()
    out = capsys.readouterr().out
    assert "00" in out and "man" in out.lower()


def test_parse_trace_snapshots() -> None:
    """``trace=True`` records one tree per word plus the initial state."""
    p = ds.parse("a man arrives", "ttr", trace=True)
    assert len(p.trace_trees) == 4
    assert len(p.trace_step_labels) == 3
    assert p.trace_step_labels == ("a", "man", "arrives")


def test_to_latex_semantics_document() -> None:
    """Semantics export wraps a full LaTeX document."""
    p = ds.parse("a man arrives", "ttr")
    r = p.to_latex("semantics", title="test semantics")
    assert "\\documentclass" in r.tex
    assert "dsttr" in r.tex or "input" in r.tex
    assert "\\[" in r.tex


def test_to_latex_tree_has_rtrees() -> None:
    """Tree export includes a ``tree`` environment."""
    p = ds.parse("a man arrives", "ttr")
    r = p.to_latex("tree")
    assert "\\begin{tree}" in r.tex


def test_to_latex_incremental_requires_trace() -> None:
    """Incremental export without ``trace=True`` raises."""
    p = ds.parse("a man arrives", "ttr")
    with pytest.raises(ValueError, match="trace"):
        p.to_latex("incremental")


def test_to_latex_incremental_with_trace() -> None:
    """Incremental layout references ``figure*`` and tabular."""
    p = ds.parse("a man arrives", "ttr", trace=True)
    r = p.to_latex("incremental")
    assert "figure*" in r.tex
    assert "tabular" in r.tex


def test_compile_tex_smoke_if_latexmk_available(tmp_path: Path) -> None:
    """When ``latexmk`` is on PATH, smoke-compile semantics to PDF."""
    import shutil

    if shutil.which("latexmk") is None:
        pytest.skip("latexmk not available")
    p = ds.parse("a man arrives", "ttr")
    tex = tmp_path / "smoke.tex"
    pdf = tmp_path / "smoke.pdf"
    r = p.to_latex("semantics", write_tex=tex, compile_tex=True, pdf_out=pdf)
    assert r.exit_code == 0
    assert r.pdf_path is not None and r.pdf_path.is_file()


def test_parse_action_trace_steps() -> None:
    """``trace=True`` captures action-level steps for Manim export."""
    p = ds.parse("a man arrives", "ttr", trace=True)
    assert p.action_steps
    assert any(step.action_name for step in p.action_steps)


def test_parse_result_to_manim_render_free() -> None:
    """``ParseResult.to_manim(render=False)`` returns generated scene code only."""
    p = ds.parse("a man arrives", "ttr", trace=True)
    r = p.to_manim(render=False)
    assert r.video_path is None
    assert "from manim import" in r.scene_code
    assert "Dynamic Syntax parse" in r.scene_code
    assert "a man arrives" in r.scene_code


def test_top_level_to_manim_render_free() -> None:
    """Top-level ``ds.to_manim`` parses with action trace and returns scene code."""
    r = ds.to_manim("a man arrives", "ttr", render=False)
    assert r.video_path is None
    assert "class AManArrivesScene" in r.scene_code
    assert "a man arrives" in r.scene_code


def test_to_manim_requires_action_trace() -> None:
    """Existing untraced results ask users to parse with ``trace=True``."""
    p = ds.parse("a man arrives", "ttr")
    with pytest.raises(ValueError, match="trace=True"):
        p.to_manim(render=False)


def test_manim_render_smoke_if_available(tmp_path: Path) -> None:
    """When Manim and LaTeX are available, smoke-render a low-quality MP4."""
    if shutil.which("manim") is None and importlib.util.find_spec("manim") is None:
        pytest.skip("manim not available")
    if shutil.which("latex") is None and shutil.which("pdflatex") is None:
        pytest.skip("latex not available for Manim MathTex")
    out = tmp_path / "parse.mp4"
    p = ds.parse("a man arrives", "ttr", trace=True)
    r = p.to_manim(output_path=out, quality="l")
    assert r.exit_code == 0
    assert r.video_path is not None and r.video_path.is_file()
