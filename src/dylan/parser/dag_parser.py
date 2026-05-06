"""Shared parser infrastructure (Java `DAGParser`)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dylan.action.action import Action
from dylan.action.computational_action import ComputationalAction
from dylan.action.grammar import Grammar
from dylan.action.lexicon import Lexicon
from dylan.action.speech_act_inference_grammar import SpeechActInferenceGrammar
from dylan.context.context import Context
from dylan.dag.dag_tuple import DAGTuple
from dylan.dag.groundable_edge import GroundableEdge
from dylan.dag.word_level_context_dag import WordLevelContextDAG
from dylan.formula.formula import Formula
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.tree.tree import Tree

if TYPE_CHECKING:
    from dylan.dag.uttered_word import UtteredWord

logger = logging.getLogger(__name__)

_MAX_NONOPTIONAL_ADJUST_PASSES = 10_000


class DAGParser:
    """Loads grammars and coordinates `Context` + DAG state (partial port)."""

    def __init__(
        self,
        lexicon: Lexicon,
        grammar: Grammar,
        sa_grammar: SpeechActInferenceGrammar | None = None,
    ) -> None:
        self.lexicon = lexicon
        self.nonoptional_grammar = Grammar()
        self.optional_grammar = Grammar()
        self.completion_grammar = Grammar()
        self._separate_grammars(grammar)
        self.sa_grammar = sa_grammar or SpeechActInferenceGrammar(Path("."))
        self.context: Context[DAGTuple, GroundableEdge] | None = None
        self.ready = True

    def _separate_grammars(self, grammar: Grammar) -> None:
        """Partition computational actions like Java `DAGParser.separateGrammars`."""
        for a in grammar.values():
            nm = a.name.lower()
            if nm == "completion" or nm == "merge" or nm.startswith("anticipation"):
                self.completion_grammar[a.name] = a
            if a.is_always_good():
                self.nonoptional_grammar[a.name] = a
            else:
                self.optional_grammar[a.name] = a

    @classmethod
    def from_resource_dir(cls, resource_dir: str | Path) -> DAGParser:
        """Load lexicon + grammars from a directory (Java `DAGParser(String)`)."""
        p = Path(resource_dir)
        return cls(Lexicon(p), Grammar(p), SpeechActInferenceGrammar(p))

    def get_state(self) -> WordLevelContextDAG:
        """Return the active word-level DAG."""
        if self.context is None:
            raise RuntimeError("context not initialised")
        return self.context.get_dag()

    def get_context(self) -> Context[DAGTuple, GroundableEdge]:
        """Return the active parser context."""
        if self.context is None:
            raise RuntimeError("context not initialised")
        return self.context

    def apply_actions(self, t: Tree, actions: list[Action]) -> Tree | None:
        """Sequentially apply `actions` to a clone of `t` (Java `applyActions`)."""
        cur: Tree | None = t.clone()
        for a in actions:
            assert cur is not None
            cur = a.exec(cur, self.context)
            if cur is None:
                return None
        return cur

    def adjust_with_non_optional_grammar(
        self,
        init_pair: tuple[list[Action], Tree],
    ) -> tuple[list[Action], Tree]:
        """Apply ``*`` computational actions until fixpoint (Java ``adjustWithNonOptionalGrammar``)."""
        actions = list(init_pair[0])
        res = init_pair[1]
        for _ in range(_MAX_NONOPTIONAL_ADJUST_PASSES):
            progressed = False
            for ca in sorted(self.nonoptional_grammar.values(), key=lambda x: x.name):
                clone = res.clone()
                nxt = ca.exec(clone, self.context)
                if nxt is not None:
                    actions.append(ca.instantiate())
                    res = nxt
                    progressed = True
                    break
            if not progressed:
                return (actions, res)
        logger.error(
            "adjust_with_non_optional_grammar exceeded %s passes — possible runaway rule loop",
            _MAX_NONOPTIONAL_ADJUST_PASSES,
        )
        return (actions, res)

    def complete_once(self, t: Tree) -> tuple[list[Action], Tree]:
        """Apply star grammar to fixpoint, then try one completion-grammar action (Java ``DAGParser.completeOnce``)."""
        init_actions, init_tree = self.adjust_with_non_optional_grammar(([], t.clone()))
        if init_actions:
            return (init_actions, init_tree)
        for ca in sorted(self.completion_grammar.values(), key=lambda x: x.name):
            completed = ca.exec(init_tree.clone(), self.context)
            if completed is not None:
                return (init_actions + [ca.instantiate()], completed)
        return (init_actions, init_tree)

    def complete_tree(self, res: Tree) -> tuple[list[Action], Tree]:
        """Run ``complete_once`` until quiescence or :meth:`Tree.is_complete` (Java ``DAGParser.complete(Tree)``)."""
        cur_actions: list[Action] = []
        cur_tree = res.clone()
        acc: list[Action] = []
        while True:
            acc.extend(cur_actions)
            if cur_tree.is_complete():
                return (acc, cur_tree)
            cur_actions, cur_tree = self.complete_once(cur_tree)
            if not cur_actions:
                return (acc, cur_tree)

    def get_final_semantics(self) -> TTRRecordType:
        """Semantics at the current tuple after `evaluate` (Java `getFinalSemantics`)."""
        dag = self.get_state()
        sem = dag.get_current_tuple().get_semantics(self.context)
        ev = sem.evaluate()
        if not isinstance(ev, TTRRecordType):
            raise TypeError("expected TTRRecordType from evaluate()")
        return ev

    def init(self) -> None:
        """Reset parser context."""
        if self.context is not None:
            self.context.init()

    def new_sentence(self) -> None:
        """Reset DAG to axiom (Java `newSentence`)."""
        self.get_state().init()

    def complete(self, word: UtteredWord | None = None) -> DAGTuple:
        """Complete the current tree and attach a completion edge."""
        dag = self.get_state()
        actions, tree = self.complete_tree(dag.get_current_tuple().get_tree())
        tup = dag.get_new_tuple(tree)
        edge = dag.get_new_completion_edge(actions, word)
        edge.set_repairable(False)
        dag.add_child(tup, edge)
        return tup

    def get_n_best_final_semantics(self, n: int) -> list[TTRRecordType]:
        """Return up to *n* final semantics from complete leaf tuples."""
        out: list[TTRRecordType] = []
        for tup in self.get_state().get_n_best_final_tuples(n):
            sem = tup.get_semantics(self.context)
            ev = sem.evaluate()
            if isinstance(ev, TTRRecordType):
                out.append(ev)
        return out

    def parse(self, goal: Formula | None = None) -> bool:
        """Delegate to `parse(goal)` on concrete parser (Java `DAGParser.parse()`)."""
        return self.parse_goal(goal)

    def parse_goal(self, goal: Formula | None) -> bool:
        raise NotImplementedError

    def parse_word(self, word: UtteredWord) -> WordLevelContextDAG | None:
        raise NotImplementedError


DAGParser.separateGrammars = DAGParser._separate_grammars  # type: ignore[attr-defined]
DAGParser.fromResourceDir = DAGParser.from_resource_dir  # type: ignore[attr-defined]
DAGParser.getState = DAGParser.get_state  # type: ignore[attr-defined]
DAGParser.getContext = DAGParser.get_context  # type: ignore[attr-defined]
DAGParser.applyActions = DAGParser.apply_actions  # type: ignore[attr-defined]
DAGParser.adjustWithNonOptionalGrammar = DAGParser.adjust_with_non_optional_grammar  # type: ignore[attr-defined]
DAGParser.completeOnce = DAGParser.complete_once  # type: ignore[attr-defined]
DAGParser.completeTree = DAGParser.complete_tree  # type: ignore[attr-defined]
DAGParser.getFinalSemantics = DAGParser.get_final_semantics  # type: ignore[attr-defined]
DAGParser.newSentence = DAGParser.new_sentence  # type: ignore[attr-defined]
DAGParser.getNBestFinalSemantics = DAGParser.get_n_best_final_semantics  # type: ignore[attr-defined]
