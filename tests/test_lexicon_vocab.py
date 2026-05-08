"""Tests for `Lexicon` load statistics and `get_vocab` formatting."""

from __future__ import annotations

from pathlib import Path

import pytest

from dylan.action.lexicon import (
    NOTEBOOK_MULTILINE_HTML_MAX_CHARS,
    Lexicon,
    LexiconLoadStats,
    NotebookMultilineText,
    _wrapped_comma_name_lines,
    strip_block_comments,
)
from dylan.action.atomic.effect_factory import EffectFactory
from dylan.action.grammar import Grammar
from dylan.parser.dag_parser import DAGParser
from dylan.parser.interactive_context_parser import InteractiveContextParser


def _write_minimal_grammar(tmp: Path) -> None:
    """Write a tiny grammar with known load outcomes."""
    (tmp / "lexical-actions.txt").write_text(
        "foo_tpl(W)\n"
        "    put(!)\n"
        "\n"
        "two_met(X,Y)\n"
        "    put(!)\n",
        encoding="utf-8",
    )
    (tmp / "lexical-macros.txt").write_text(
        "good_macro()\n"
        "    put(!)\n"
        "\n"
        "bad_header_only()\n",
        encoding="utf-8",
    )
    (tmp / "lexicon.txt").write_text(
        "hello foo_tpl stem\n"
        "badword missing_tpl\n"
        "one two_met only_one\n",
        encoding="utf-8",
    )


def test_failed_words_wrap_ten_per_line_in_get_vocab(tmp_path: Path) -> None:
    """Failed-word names break at ten per line in the stats header."""
    names = tuple(f"w{i}" for i in range(23))
    lines = _wrapped_comma_name_lines(names, per_line=10, indent="  ")
    assert len(lines) == 3
    assert lines[0].count(",") == 9
    assert lines[1].count(",") == 9
    assert lines[2].count(",") == 2


def test_dag_parser_get_vocab_delegates_to_lexicon(tmp_path: Path) -> None:
    """`DAGParser.get_vocab` forwards to the embedded lexicon (same output)."""
    _write_minimal_grammar(tmp_path)
    lex = Lexicon(tmp_path)
    parser = DAGParser(lex, Grammar(tmp_path))
    assert parser.get_vocab(groupby="alpha") == lex.get_vocab(groupby="alpha")


def test_interactive_context_parser_get_vocab_matches_lexicon(tmp_path: Path) -> None:
    """Subclass exposes `get_vocab` via `DAGParser` delegation."""
    _write_minimal_grammar(tmp_path)
    lex = Lexicon(tmp_path)
    parser = InteractiveContextParser(tmp_path)
    assert parser.get_vocab(groupby="category") == lex.get_vocab(groupby="category")


def test_notebook_multiline_text_jupyter_html() -> None:
    """NotebookMultilineText exposes HTML repr so Jupyter shows line breaks (not one escaped string)."""
    t = NotebookMultilineText("a\nb")
    html = t._repr_html_()
    assert "<pre" in html and "white-space: pre-wrap" in html
    assert "a" in html and "b" in html
    bundle = t._repr_mimebundle_()
    assert set(bundle.keys()) == {"text/html", "text/plain"}
    assert bundle["text/plain"] == "a\nb"


def test_notebook_multiline_text_mimebundle_respects_exclude_html() -> None:
    """When IPython excludes ``text/html``, the bundle omits it (avoids heavy escape work)."""
    t = NotebookMultilineText("a\nb")
    bundle = t._repr_mimebundle_(exclude=frozenset({"text/html"}))
    assert set(bundle.keys()) == {"text/plain"}


def test_notebook_multiline_text_mimebundle_respects_include_plain_only() -> None:
    """When ``include`` whitelists only ``text/plain``, skip HTML MIME."""
    t = NotebookMultilineText("x")
    bundle = t._repr_mimebundle_(include=frozenset({"text/plain"}))
    assert set(bundle.keys()) == {"text/plain"}


def test_notebook_multiline_text_oversized_skips_heavy_html_in_mimebundle() -> None:
    """Very long strings stay in ``text/plain`` only so notebooks do not freeze on HTML escape."""
    body = "z" * (NOTEBOOK_MULTILINE_HTML_MAX_CHARS + 1)
    t = NotebookMultilineText(body)
    bundle = t._repr_mimebundle_()
    assert set(bundle.keys()) == {"text/plain"}
    assert bundle["text/plain"] == body
    assert len(bundle["text/plain"]) > NOTEBOOK_MULTILINE_HTML_MAX_CHARS
    html = t._repr_html_()
    assert len(html) < 500
    assert "characters" in html and "text/plain" in html


def test_get_vocab_accepts_positional_groupby(tmp_path: Path) -> None:
    """Notebooks often pass ``get_vocab('alpha')``; groupby is positional before keyword-only stream/backend."""
    _write_minimal_grammar(tmp_path)
    lex = Lexicon(tmp_path)
    assert lex.get_vocab("alpha") == lex.get_vocab(groupby="alpha")


def test_lexicon_load_stats_and_vocab_views(tmp_path: Path) -> None:
    """Load statistics match skips; `get_vocab` supports category and alpha views."""
    _write_minimal_grammar(tmp_path)
    lex = Lexicon(tmp_path)
    st = lex.load_stats
    assert st.word_entries_loaded == 1
    assert st.words_unique == 1
    assert st.words_failed == 2
    assert st.macros_loaded == 1
    assert st.macros_failed == 1
    assert st.words_failed_names == ("badword", "one")
    assert st.macros_failed_names == ("bad_header_only",)

    plain_cat = lex.get_vocab(groupby="category", backend="plain")
    assert f"Lexicon: {tmp_path.name}" in plain_cat
    assert f"Source: {(tmp_path / 'lexicon.txt').resolve()}" in plain_cat
    assert "Failed words:" in plain_cat
    assert "badword, one" in plain_cat
    assert "Failed macros:" in plain_cat
    assert "bad_header_only" in plain_cat
    assert "hello" in plain_cat and "foo_tpl" in plain_cat
    assert "=== foo_tpl ===" in plain_cat
    assert "Load statistics:" in plain_cat
    assert "put(!)" not in plain_cat

    plain_alpha = lex.get_vocab(groupby="alpha", backend="plain")
    assert "hello" in plain_alpha
    assert any(t == "hello" for ln in plain_alpha.splitlines() for t in ln.split())

    assert lex.get_vocab(groupby="category", backend="plain") == plain_cat

    lex.invalidate_vocab_cache()
    assert lex.get_vocab(groupby="category", backend="plain") == plain_cat


def test_get_vocab_empty_lexicon() -> None:
    """Empty lexicon reports zero stats and no rows."""
    lex = Lexicon()
    assert lex.load_stats == LexiconLoadStats(0, 0, 0, 0, 0)
    out = lex.get_vocab()
    assert "Word entries loaded:    0" in out
    assert "(no lexical entries loaded)" in out


def test_init_macro_templates_counts() -> None:
    """Macro parser returns loaded vs incomplete-at-EOF counts."""
    lines = strip_block_comments(
        "a()\n\tput(!)\n\nb()\n".splitlines(),
    )
    loaded, failed, failed_names = EffectFactory.init_macro_templates(lines)
    assert loaded == 1
    assert failed == 1
    assert failed_names == ("b",)


def test_get_vocab_rich_backend(tmp_path: Path) -> None:
    """Rich backend renders without error when `rich` is installed."""
    pytest.importorskip("rich")
    _write_minimal_grammar(tmp_path)
    lex = Lexicon(tmp_path)
    rich_out = lex.get_vocab(backend="rich", groupby="alpha")
    assert "hello" in rich_out
    assert "Load statistics" in rich_out or "Word entries loaded" in rich_out


def test_rich_vocab_stats_block_matches_plain_prefix(tmp_path: Path) -> None:
    """Stats header (Lexicon, Source, load stats, failed lists) is byte-identical to plain before Rich tables."""
    pytest.importorskip("rich")
    _write_minimal_grammar(tmp_path)
    lex = Lexicon(tmp_path)
    plain = lex.get_vocab(groupby="category", backend="plain")
    rich = lex.get_vocab(groupby="category", backend="rich")
    head = plain.split("===", 1)[0]
    assert rich.startswith(head)


def test_get_vocab_invalid_backend_raises(tmp_path: Path) -> None:
    """Unknown ``backend`` values raise a clear error (typos do not silently fall through)."""
    _write_minimal_grammar(tmp_path)
    lex = Lexicon(tmp_path)
    with pytest.raises(ValueError, match="backend must be 'plain' or 'rich'"):
        lex.get_vocab(backend="markdown")  # type: ignore[arg-type]


def test_get_vocab_rich_then_plain_same_lexicon(tmp_path: Path) -> None:
    """Rich then plain on one lexicon remains stable (no accidental global Rich stdout coupling)."""
    pytest.importorskip("rich")
    _write_minimal_grammar(tmp_path)
    lex = Lexicon(tmp_path)
    rich_alpha = lex.get_vocab(groupby="alpha", backend="rich")
    plain_alpha = lex.get_vocab(groupby="alpha", backend="plain")
    assert "hello" in rich_alpha and "hello" in plain_alpha
    assert plain_alpha == Lexicon(tmp_path).get_vocab(groupby="alpha", backend="plain")
