"""Parse-versus-learn gate: entropy, count, and win-stay."""

from __future__ import annotations

from pathlib import Path

from dylan.action.atomic.effect import Effect
from dylan.action.lexical_action import LexicalAction
from dylan.dag.parser_tuple import ParserTuple
from dylan.dag.uttered_word import UtteredWord
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.em_learner.candidate_sequence import CandidateSequence
from dylan.induction.em_learner.common import Word
from dylan.induction.em_learner.explore_gate import (
    ExploreGate,
    entropy_and_count_allow_exploit,
    normalized_hypothesis_entropy,
)
from dylan.induction.em_learner.lexical_hypothesis import LexicalHypothesis
from dylan.induction.em_learner.ttr_hypothesiser import TTRHypothesiser
from dylan.induction.em_learner.ttr_word_learner import TTRWordLearner
from dylan.induction.em_learner.word_hypothesis_base import WordHypothesisBase
from dylan.induction.em_learner.word_log_prob_distribution import WordLogProbDistribution


class _IdentityEffect(Effect):
    """No-op effect so candidate sequences can be split in unit tests."""

    def exec(self, tree, context=None):  # type: ignore[no-untyped-def]
        """Return *tree* unchanged."""
        return tree

    def exec_tuple_context(self, tree, context=None):  # type: ignore[no-untyped-def]
        """Return *tree* unchanged."""
        return tree

    def instantiate(self) -> "_IdentityEffect":
        """Return a fresh copy."""
        return _IdentityEffect()


def _peaked(word: str, *, weight: float) -> WordHypothesisBase:
    """One certain hypothesis for *word* with the given example *weight*."""
    seq = CandidateSequence(ParserTuple(), [LexicalHypothesis(word, _IdentityEffect(), True)], word)
    base = WordHypothesisBase()
    base.add_sequence_tuples(seq.split())
    base.update_dists_end_of_example([word])
    base.prior_dist[Word(word)].set_weight(weight)
    return base


def test_normalized_entropy_bounds() -> None:
    """One hypothesis is 0; two equal hypotheses are 1."""
    peaked = _peaked("door", weight=5)
    assert normalized_hypothesis_entropy(peaked.prior_dist[Word("door")]) == 0.0

    from dylan.induction.em_learner.word_hypothesis import WordHypothesis

    uniform = WordLogProbDistribution(Word("door"), weight=5, max_id=2)
    for hyp_id in (1, 2):
        hyp = WordHypothesis(hyp_id)
        hyp.word = Word("door")
        hyp.set_prob(0.5)
        uniform[hyp] = hyp.get_log_prob()
    assert normalized_hypothesis_entropy(uniform) == 1.0
    assert normalized_hypothesis_entropy(WordLogProbDistribution(Word("door"))) is None


def test_entropy_and_count_truth_table() -> None:
    """Exploit only when both the entropy cap and the example-count floor hold."""
    peaked = _peaked("door", weight=5).prior_dist[Word("door")]
    assert entropy_and_count_allow_exploit(peaked, max_normalized_entropy=0.5, min_word_count=5)
    peaked.set_weight(4)
    assert not entropy_and_count_allow_exploit(peaked, max_normalized_entropy=0.5, min_word_count=5)
    peaked.set_weight(5)
    # Spread the single hypothesis by adding a second equal one via a fresh distribution.
    wide = WordLogProbDistribution(Word("door"), weight=5, max_id=2)
    from dylan.induction.em_learner.word_hypothesis import WordHypothesis

    for hyp_id, prob in ((1, 0.5), (2, 0.5)):
        hyp = WordHypothesis(hyp_id)
        hyp.word = Word("door")
        hyp.set_prob(prob)
        wide[hyp] = hyp.get_log_prob()
    assert not entropy_and_count_allow_exploit(wide, max_normalized_entropy=0.5, min_word_count=5)
    assert not entropy_and_count_allow_exploit(None, max_normalized_entropy=0.5, min_word_count=5)


def test_words_to_update_skips_win_stay_only() -> None:
    """A word searched on any visit is updated; a pure win-stay is not."""
    base = _peaked("door", weight=5)
    gate = ExploreGate(base, max_normalized_entropy=0.5, min_word_count=5)
    gate.note_win_stay("door")
    gate.note_search("open")
    assert gate.words_to_update(["door", "open", "door"]) == [Word("open")]


class _ProbeHypothesiser(TTRHypothesiser):
    """Records whether known actions and new hypotheses were requested."""

    def __init__(self, known_result: bool) -> None:
        """Skip grammar loading and record *known_result* as the action-apply outcome."""
        super().__init__(resource_dir_or_url=None, learner_comp_actions_path=None)
        self.known_result = known_result
        self.known_calls = 0
        self.hyp_calls = 0

    def apply_known_lexical(self) -> bool:
        """Pretend the current lexical action did or did not apply."""
        self.known_calls += 1
        return self.known_result

    def apply_lexical_hypotheses(self, target: object = None) -> None:
        """Record a hypothesise request."""
        _ = target
        self.hyp_calls += 1

    def apply_optional_grammar(self, target: object = None) -> None:
        """Skip optional grammar in this probe."""
        _ = target


def test_gated_lexical_win_stay_and_fail_search() -> None:
    """Eligible words parse when an action applies, and hypothesise when it does not."""
    base = _peaked("door", weight=5)
    stay = _ProbeHypothesiser(True)
    stay.state.word_stack = [UtteredWord("door")]
    stay.seed_lexicon["door"] = [
        LexicalAction("door", ["IF    ?Ty(e)", "THEN  abort", "ELSE  abort"], None)
    ]
    gate = ExploreGate(base, max_normalized_entropy=0.5, min_word_count=5)
    stay._apply_gated_lexical(gate)
    assert stay.known_calls == 1
    assert stay.hyp_calls == 0
    assert gate.words_to_update(["door"]) == []

    search = _ProbeHypothesiser(False)
    search.state.word_stack = [UtteredWord("door")]
    search.seed_lexicon["door"] = stay.seed_lexicon["door"]
    fail = ExploreGate(base, max_normalized_entropy=0.5, min_word_count=5)
    search._apply_gated_lexical(fail)
    assert search.known_calls == 1
    assert search.hyp_calls == 1
    assert fail.words_to_update(["door"]) == [Word("door")]


def test_low_count_seed_word_is_hypothesised() -> None:
    """A seed word under the count floor is hypothesised even if an action would apply."""
    base = _peaked("door", weight=2)
    probe = _ProbeHypothesiser(True)
    probe.state.word_stack = [UtteredWord("door")]
    probe.seed_lexicon["door"] = [
        LexicalAction("door", ["IF    ?Ty(e)", "THEN  abort", "ELSE  abort"], None)
    ]
    gate = ExploreGate(base, max_normalized_entropy=0.5, min_word_count=5)
    probe._apply_gated_lexical(gate)
    assert probe.known_calls == 1
    assert probe.hyp_calls == 1
    assert Word("door") in gate.words_to_update(["door"])


class _ScriptedHypothesiser:
    """Hypothesiser double that records stay versus search from the live gate."""

    def __init__(self, *, fail_actions: bool) -> None:
        """When *fail_actions* is set, even an eligible word is searched."""
        self.seed_lexicon: dict[str, list] = {}
        self.grammar = None
        self.explore_gate: ExploreGate | None = None
        self.fail_actions = fail_actions
        self.words: list[Word] = []

    def load_training_example(self, words: list[Word], target: TTRRecordType) -> None:
        """Remember *words* for :meth:`hypothesise`."""
        _ = target
        self.words = list(words)

    def hypothesise(self) -> list[CandidateSequence]:
        """Mark each word and return a sequence only the learner should keep for searched words."""
        assert self.explore_gate is not None
        sequences: list[CandidateSequence] = []
        for word in self.words:
            if self.explore_gate.entropy_and_count_met(word) and not self.fail_actions:
                self.explore_gate.note_win_stay(word)
            else:
                self.explore_gate.note_search(word)
            sequences.append(
                CandidateSequence(
                    ParserTuple(),
                    [LexicalHypothesis(word.word(), _IdentityEffect(), True)],
                    [word],
                ),
            )
        return sequences

    def get_seed_lexicon(self) -> dict[str, list]:
        """Return the stand-in lexicon."""
        return self.seed_lexicon


def test_learner_does_not_increment_weight_on_win_stay(tmp_path: Path) -> None:
    """A peaked word at the count minimum keeps its weight when its action applies."""
    gdir = tmp_path / "grammar"
    gdir.mkdir()
    (gdir / "computational-actions.txt").write_text(
        "hyp-adj-smoke\nIF      ?Ty(e)\nTHEN    abort\nELSE    abort\n\n",
        encoding="utf-8",
    )
    target = TTRRecordType.parse("[x==door:e|head==x:e]")
    assert target is not None
    from dylan.induction.em_learner.record_type_corpus import RecordTypeCorpus

    corpus = RecordTypeCorpus()
    corpus.add_example("door", target)
    learner = TTRWordLearner(
        seed_resource_dir=None,
        corpus=corpus,
        learner_comp_actions_path=gdir,
        min_word_count=5,
        max_normalized_entropy=0.5,
    )
    learner.hb = _peaked("door", weight=5)
    learner.hypothesiser = _ScriptedHypothesiser(fail_actions=False)  # type: ignore[assignment]
    assert learner.learn_once() is True
    assert learner.hb.prior_dist[Word("door")].get_weight() == 5

    learner.corpus_iterator = iter(corpus)
    learner.hypothesiser = _ScriptedHypothesiser(fail_actions=True)  # type: ignore[assignment]
    assert learner.learn_once() is True
    assert learner.hb.prior_dist[Word("door")].get_weight() == 6
