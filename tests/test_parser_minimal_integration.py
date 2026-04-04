"""Integration: load fixture grammar and parse one word."""

from __future__ import annotations

from pathlib import Path

import pytest

from dylan.dag.uttered_word import UtteredWord
from dylan.nlp.types import Utterance, utterance_from_text
from dylan.parser.interactive_context_parser import InteractiveContextParser

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "parser_minimal"


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
