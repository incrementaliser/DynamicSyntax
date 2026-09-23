"""LaTeX and display rendering for TTR records and DS node labels."""

from __future__ import annotations

import dynamicsyntax as ds
from dylan.formula.latex.formula_tex import formula_to_latex, record_display_lines
from dylan.formula.manim.tree_scene import serialize_action_steps
from dylan.formula.ttr_record_type import TTRRecordType


def test_record_to_latex_uses_subscript_manifests() -> None:
    """Manifest fields are subscripts in a two-column array, not a rewritten ``:=``."""
    record = TTRRecordType.parse("[x : e|e1==arrive : es|p==man(x) : t|head==e1 : es]")
    assert record is not None
    tex = record.to_latex()
    assert r"\begin{array}" in tex
    assert r"&" in tex
    assert r"\mathit{arrive}" in tex
    assert r"\mathit{man}(x)" in tex
    assert "e_{s}" in tex
    assert r"{e_{1}}_{=\mathit{arrive}}" in tex
    assert r"\mathrel" not in tex
    assert "==" not in tex


def test_lambda_to_latex_is_not_a_caret() -> None:
    """A TTR lambda is ``\\lambda``, not the Java ``R^body`` caret dumped into math."""
    record = TTRRecordType.parse("[x : e|head==x : e]")
    assert record is not None
    from dylan.formula.predicate_argument import Predicate
    from dylan.formula.ttr_infix_expression import TTRInfixExpression
    from dylan.formula.ttr_lambda import TTRLambdaAbstract
    from dylan.formula.variable import Variable

    lam = TTRLambdaAbstract(
        Variable("R1"), TTRInfixExpression(Predicate("++"), Variable("R1"), record)
    )
    tex = formula_to_latex(lam)
    assert tex.startswith(r"\lambda R_{1}.")
    assert r"\mathbin{+\mkern-6mu+}" in tex
    assert "^" not in tex


def test_record_display_breaks_fields() -> None:
    """Plain display keeps one field per line and uses a manifest colon-equals."""
    record = TTRRecordType.parse("[x : e|e1==arrive : es]")
    assert record is not None
    lines = record_display_lines(record)
    assert lines[0].startswith("[ ")
    assert any("arrive" in line and "≔" in line for line in lines)
    assert any("eₛ" in line for line in lines)


def test_manim_payload_shows_each_word_once() -> None:
    """Action steps that share a surface word do not each add that word to the utterance."""
    parsed = ds.parse("a man arrives", "ttr", trace=True)
    data = serialize_action_steps(
        parsed.action_steps, semantics=parsed.semantics, sentence=parsed.sentence
    )
    shown = [step["word"] for step in data["steps"] if step["show_word"]]
    assert shown == ["a", "man", "arrives"]
    nodes = data["steps"][-1]["after"]["nodes"]
    flat = [line for node in nodes for line in node["lines"]]
    assert any(line.startswith("Ty(") or line.startswith("?Ty(") for line in flat)
    assert all(not line.startswith("0000") for line in flat)
    assert any("◆" in line for line in flat)
    assert any("arrive" in line for line in data["semantics_lines"])
