"""Base hypothesiser for grammar induction."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from dylan.action.action import Action
from dylan.action.grammar import Grammar
from dylan.action.lexicon import Lexicon
from dylan.dag.parser_tuple import ParserTuple
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.em_learner.candidate_sequence import CandidateSequence
from dylan.induction.em_learner.common import Word, sentence_from_text
from dylan.induction.em_learner.dag_induction_state import DAGInductionState
from dylan.tree.tree import Tree


class Hypothesiser:
    """Hypothesise action sequences from a start tree to a target."""

    HYP_ACTION_PREFIX = "hyp"
    HYP_ADJUNCTION_PREFIX = "hyp-adj"
    HYP_ADJ_T_PREFIX = "hyp-adj-t"

    def __init__(
        self,
        resource_dir_or_url: str | Path | None = None,
        top_n: int = 3,
        load_learnt_lexicon: bool = False,
    ) -> None:
        """Load seed lexicon and grammar resources."""
        self.resource_dir = Path(resource_dir_or_url) if resource_dir_or_url is not None else Path(".")
        self.seed_lexicon = Lexicon(self.resource_dir, top_n)
        self.grammar = Grammar(self.resource_dir)
        self.nonoptional_grammar = Grammar()
        self.optional_grammar = Grammar()
        self.separate_grammars(self.grammar)
        self.state = DAGInductionState()
        self.target: Tree | TTRRecordType | None = None
        self.hypotheses: set[CandidateSequence] = set()
        self.load_learnt_lexicon = load_learnt_lexicon

    def separate_grammars(self, grammar: Grammar) -> None:
        """Split grammar into optional and non-optional actions."""
        self.nonoptional_grammar = Grammar()
        self.optional_grammar = Grammar()
        for name, action in grammar.items():
            if action.is_always_good():
                self.nonoptional_grammar[name] = action
            else:
                self.optional_grammar[name] = action

    def load_training_example(self, sentence: str | Iterable[str | Word], target: Tree | TTRRecordType) -> None:
        """Load one training example."""
        words = sentence_from_text(sentence) if isinstance(sentence, str) else [Word(str(w)) for w in sentence]
        self.state = DAGInductionState()
        self.state.word_stack = []
        self.words = words
        self.target = target
        self.hypotheses.clear()

    def hypothesise(self) -> set[CandidateSequence]:
        """Return candidate sequences for the loaded example."""
        if not hasattr(self, "words"):
            return set()
        actions: list[Action] = []
        for word in self.words:
            entries = list(self.seed_lexicon.lookup(word.word()))
            if entries:
                actions.append(entries[0].instantiate())
            else:
                from dylan.induction.em_learner.lexical_hypothesis import LexicalHypothesis

                actions.append(LexicalHypothesis(word.word(), None, True))
        candidate = CandidateSequence(ParserTuple(Tree()), actions, self.words)
        self.hypotheses = {candidate}
        return self.hypotheses

    def local_lexical_hyps(self, target: Tree, fixed_on_target: object) -> set[Action]:
        """Return local lexical hypotheses at a target position."""
        _ = (target, fixed_on_target)
        return set()

    def get_seed_lexicon(self) -> Lexicon:
        """Return seed lexicon."""
        return self.seed_lexicon


Hypothesiser.separateGrammars = Hypothesiser.separate_grammars  # type: ignore[attr-defined]
Hypothesiser.loadTrainingExample = Hypothesiser.load_training_example  # type: ignore[attr-defined]
Hypothesiser.localLexicalHyps = Hypothesiser.local_lexical_hyps  # type: ignore[attr-defined]
Hypothesiser.getSeedLexicon = Hypothesiser.get_seed_lexicon  # type: ignore[attr-defined]
