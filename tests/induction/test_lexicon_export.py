"""Tests for learnt lexicon serialization (``Lexicon.write_to_text_file`` / ``get_core_action``)."""

from __future__ import annotations

from pathlib import Path

from dylan.action.atomic.put import Put
from dylan.action.lexical_action import LexicalAction
from dylan.action.lexicon import Lexicon
from dylan.dag.parser_tuple import ParserTuple
from dylan.induction.em_learner.candidate_sequence import CandidateSequence
from dylan.induction.em_learner.common import action_key
from dylan.induction.em_learner.lexical_hypothesis import LexicalHypothesis, _CompositeEffect
from dylan.induction.em_learner.word_hypothesis import WordHypothesis
from dylan.tree.label.labels import label_factory_create
from dylan.tree.tree import Tree


def test_lexicon_write_to_text_file_includes_body(tmp_path: Path) -> None:
    """``write_to_text_file`` must emit ``_source_lines``, not only the surface word."""
    lines = ["IF      ?Ty(e)", "THEN    put(Ty(e))", "ELSE    abort"]
    act = LexicalAction("door", lines, "test_cat")
    setattr(act, "prob", 0.5)
    setattr(act, "rank", 0)
    lex = Lexicon(None)
    lex["door"] = [act]
    out = tmp_path / "out.txt"
    lex.write_to_text_file(out)
    text = out.read_text(encoding="utf-8")
    assert "[0.5,0]" in text
    assert "IF      ?Ty(e)" in text
    assert "THEN    put(Ty(e))" in text


def test_get_core_action_skips_stub_hypothesis_to_the_right() -> None:
    """When a no-effect stub sits after a real hypothesis in sequence order, use the real body (right-to-left scan)."""
    put = Put(label_factory_create("Ty(e)"))
    real = LexicalHypothesis("hypSem(x)", [put], True)
    stub = LexicalHypothesis("w", None, True)
    cs = CandidateSequence(ParserTuple(Tree()), [real, stub], ["w"])
    wh = WordHypothesis(0)
    assert wh.intersect_into(cs)
    core = wh.get_core_action()
    assert isinstance(core, LexicalAction)
    assert core.word == "w"
    body = "\n".join(core._source_lines)
    assert "put(" in body
    assert body.count("THEN    abort") <= 1


def test_action_key_includes_lexical_hypothesis_effect() -> None:
    """Same word with different induced effects must not share one ``action_key`` (intersection dedup)."""
    put_e = Put(label_factory_create("Ty(e)"))
    put_cn = Put(label_factory_create("Ty(cn)"))
    h1 = LexicalHypothesis("hyp", [put_e], True)
    h2 = LexicalHypothesis("hyp", [put_cn], True)
    assert action_key(h1) != action_key(h2)


def test_get_core_action_from_lexical_hypothesis() -> None:
    """Induced :class:`LexicalHypothesis` with ``put`` effects becomes a :class:`LexicalAction` with source lines."""
    put = Put(label_factory_create("Ty(e)"))
    hyp = LexicalHypothesis("hypSem(x)", [put], True)
    cs = CandidateSequence(ParserTuple(Tree()), [hyp], ["w"])
    wh = WordHypothesis(0)
    assert wh.intersect_into(cs)
    core = wh.get_core_action()
    assert isinstance(core, LexicalAction)
    assert core.word == "w"
    assert any("put(" in ln for ln in core._source_lines)


def test_lexicon_export_lines_for_hypothesis_without_source(tmp_path: Path) -> None:
    """Hypothesised entries lack ``_source_lines`` but must still serialize reloadable IF/THEN blocks."""
    put = Put(label_factory_create("Ty(e)"))
    hyp = LexicalHypothesis("w", put, True)
    assert getattr(hyp, "_source_lines", None) is None
    setattr(hyp, "prob", 0.9)
    setattr(hyp, "rank", 0)
    lex = Lexicon(None)
    lex["door"] = [hyp]  # type: ignore[list-item]
    out = tmp_path / "out.txt"
    lex.write_to_text_file(out)
    text = out.read_text(encoding="utf-8")
    assert "[0.9,0]" in text
    assert "IF" in text
    assert any(s in text for s in ("put(Ty(e))", "put("))


def test_composite_effect_constructible_exec_tuple_context() -> None:
    """Multi-effect ``hyp-sem`` must wrap as :class:`_CompositeEffect` without ABC instantiation errors."""
    p1 = Put(label_factory_create("Ty(e)"))
    p2 = Put(label_factory_create("Ty(e)"))
    comp = _CompositeEffect([p1, p2])
    assert comp.effects == [p1, p2]
    _ = comp.exec_tuple_context(Tree(), None)
