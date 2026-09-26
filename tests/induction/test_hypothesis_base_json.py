"""JSON round-trip for :class:`WordHypothesisBase`."""

from __future__ import annotations

from pathlib import Path

import pytest

from dylan.action.atomic.if_then_else import IfThenElse
from dylan.action.grammar import Grammar
from dylan.action.lexical_action import LexicalAction
from dylan.dag.parser_tuple import ParserTuple
from dylan.induction.em_learner.candidate_sequence import CandidateSequence
from dylan.induction.em_learner.common import Word
from dylan.induction.em_learner.lexical_hypothesis import LexicalHypothesis
from dylan.induction.em_learner.word_hypothesis import WordHypothesis
from dylan.induction.em_learner.word_hypothesis_base import WordHypothesisBase


_LINES = ["IF    ?Ty(e)", "THEN  abort", "ELSE  abort"]


def _lexical(name: str) -> LexicalHypothesis:
    """One abort lexical hypothesis rebuilt the same way JSON load does."""
    return LexicalHypothesis(name, IfThenElse.from_lines(_LINES), True)


def test_split_retains_lexical_action_for_revising_word() -> None:
    """A seed lexical action is kept, as a hypothesis, only for words being revised."""
    action = LexicalAction("door", _LINES, None)
    seq = CandidateSequence(ParserTuple(), [action], "door")
    assert seq.split() == set()
    retained = seq.split({"door"})
    assert len(retained) == 1
    chunk = next(iter(retained))[0]
    assert isinstance(chunk[-1], LexicalHypothesis)


def test_hypothesis_base_json_round_trip(tmp_path: Path) -> None:
    """Weights, ids, and probabilities survive save/load, and intersection still accepts a sequence."""
    grammar_dir = tmp_path / "grammar"
    grammar_dir.mkdir()
    (grammar_dir / "computational-actions.txt").write_text(
        "thin\nIF      ?Ty(e)\nTHEN    abort\nELSE    abort\n\n",
        encoding="utf-8",
    )
    grammar = Grammar(grammar_dir)
    lexical = _lexical("door")
    hypothesis = WordHypothesis(3)
    assert hypothesis.intersect_into(
        CandidateSequence(ParserTuple(), [grammar["thin"], lexical], "door"),
    )
    hypothesis.howmany = 4
    hypothesis.set_prob(1.0)
    base = WordHypothesisBase()
    base.num_training_so_far = 9
    dist_word = Word("door")
    from dylan.induction.em_learner.word_log_prob_distribution import WordLogProbDistribution

    dist = WordLogProbDistribution(dist_word, weight=5.0, max_id=3)
    dist[hypothesis] = hypothesis.get_log_prob()
    base.prior_dist[dist_word] = dist

    path = tmp_path / "hypothesis-base.json"
    base.save_json(path, grammar=grammar)
    loaded = WordHypothesisBase()
    loaded.load_json(path, grammar=grammar)

    assert loaded.num_training_so_far == 9
    loaded_dist = loaded.prior_dist[Word("door")]
    assert loaded_dist.get_weight() == 5.0
    assert loaded_dist.max_id == 3
    restored = loaded_dist.get_all_hyps()
    assert len(restored) == 1
    assert restored[0].hyp_id == 3
    assert restored[0].get_count() == 4
    assert restored[0].get_prob() == pytest.approx(1.0)

    again = CandidateSequence(ParserTuple(), [grammar["thin"], _lexical("door")], "door")
    assert restored[0].intersect_into(again) is True
    assert restored[0].get_count() == 5


def test_load_json_requires_computational_action(tmp_path: Path) -> None:
    """A stored computational action that the grammar does not define is an error."""
    import json

    path = tmp_path / "hypothesis-base.json"
    payload = {
        "version": 1,
        "num_training_so_far": 0,
        "words": [
            {
                "word": "door",
                "weight": 1.0,
                "max_id": 1,
                "hypotheses": [
                    {
                        "hyp_id": 1,
                        "log_prob": 0.0,
                        "howmany": 1,
                        "sequences": [[{"kind": "computational", "name": "missing"}]],
                    },
                ],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        WordHypothesisBase().load_json(path, grammar=Grammar())
