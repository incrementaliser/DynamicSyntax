"""Integration: load fixture grammar and parse one word."""

from __future__ import annotations

from pathlib import Path

import pytest

from dylan.dag.uttered_word import UtteredWord
from dylan.nlp.types import Utterance, utterance_from_text
from dylan.parser.interactive_context_parser import InteractiveContextParser
from dylan.tree.label.labels import Requirement
from dylan.tree.tree import Tree

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "parser_minimal"
GRAMMAR_2026 = Path(__file__).resolve().parents[1] / "resources" / "2026-english-ttr-test"


def _requirement_count(tree: Tree) -> int:
    """Count :class:`~dylan.tree.label.labels.Requirement` labels on *tree*."""
    return sum(1 for node in tree.values() for lab in node.labels if isinstance(lab, Requirement))


def test_parse_single_word_test() -> None:
    p = InteractiveContextParser(FIXTURE)
    p.init()
    dag = p.parse_word(UtteredWord("test", "Dylan"))
    assert dag is not None
    assert not dag.word_stack
    tup = p.get_best_tuple()
    assert tup is not None
    sem = p.get_final_semantics()
    assert sem is not None


def test_parse_utterance_whitespace() -> None:
    p = InteractiveContextParser(FIXTURE)
    p.init()
    utt = utterance_from_text("Dylan", "test")
    assert isinstance(utt, Utterance)
    assert p.parse_utterance(utt) is True


def test_modal_label_two_operator_path_not_truncated() -> None:
    """Regression: Python regex must not collapse ``<\\/1\\/0>`` to a single ``\\/0``."""
    from dylan.tree.label.labels import ModalLabel, label_factory_create

    lab = label_factory_create(r"</\1\/0>person(s3)")
    assert isinstance(lab, ModalLabel)
    assert len(lab.modality.ops) == 2


def test_parse_mini_sentence_2026_grammar() -> None:
    """End-to-end: ``a man knows you`` with bundled 2026 test lexicon."""
    if not (GRAMMAR_2026 / "lexicon.txt").is_file():
        pytest.skip("2026-english-ttr-test grammar not in resources")
    p = InteractiveContextParser(GRAMMAR_2026)
    p.init()
    for w in ("a", "man", "knows", "you"):
        assert p.parse_word(UtteredWord(w, "Dylan", "you")) is not None
    tup = p.get_best_tuple()
    assert tup is not None
    parsed = tup.tree.clone()
    n_req_before = _requirement_count(parsed)
    _, finished = p.complete_tree(parsed)
    assert finished.pointer.is_root()
    assert _requirement_count(finished) < n_req_before
    sem = tup.get_semantics(p.context)
    s = str(sem)
    assert len(s) > 10
    assert "subj(" in s and "obj(" in s and "man(" in s
    assert "pres(" in s and "pres(head)" not in s.replace(" ", "")
    # Addressee metavar ``X`` must unify to ``you`` (Java ``AddresseeLabel`` + ``AtomicFormula``).
    assert "you" in s
    compact = s.replace(" ", "")
    assert "==X:" not in compact and "|X==" not in compact
    assert any(
        "Fo(" in str(lab)
        for node in tup.tree.values()
        for lab in node.labels
    )
