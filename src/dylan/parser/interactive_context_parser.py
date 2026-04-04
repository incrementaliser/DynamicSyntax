"""Interactive word-level DAG parser (Java `InteractiveContextParser`)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from dylan.action.action import Action
from dylan.action.lexical_action import LexicalAction
from dylan.action.grammar import Grammar
from dylan.action.lexicon import Lexicon
from dylan.action.speech_act_inference_grammar import SpeechActInferenceGrammar
from dylan.context.context import Context
from dylan.dag.dag_tuple import DAGTuple
from dylan.dag.groundable_edge import GroundableEdge
from dylan.dag.uttered_word import UtteredWord
from dylan.dag.word_level_context_dag import WordLevelContextDAG
from dylan.formula.formula import Formula
from dylan.nlp.types import WAIT_TOKEN, RELEASE_TURN_TOKEN, Utterance
from dylan.parser.dag_parser import DAGParser
from dylan.tree.label.labels import Requirement, TypeLabel
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _repoint_for_verb_lexical(tree: Tree, la: LexicalAction) -> None:
    """Move pointer to ``01`` when applying finite/inf verb templates (``v_*``).

    Noun work often leaves the pointer inside an NP; English ``v_tran_fin`` IF clauses
    are stated relative to the predicate functor node ``01``.
    """
    at = la.get_lexical_action_type() or ""
    if not at.startswith("v_"):
        return
    pred = NodeAddress("01")
    if pred not in tree:
        return
    for lab in tree[pred].labels:
        if isinstance(lab, Requirement) and isinstance(lab.inner, TypeLabel):
            if "e>t" in str(lab.inner.type).replace(" ", ""):
                tree.pointer = pred
                return

_MAX_LEXICAL_ADJUSTMENT_PAIRS = 50_000

DEFAULT_NAME = "Dylan"


def _tree_structural_key(tree: Tree) -> tuple:
    """Hashable key capturing the full tree content + pointer.

    Java uses ``HashSet<Tree>`` backed by ``Tree.equals``/``hashCode``
    (structural equality of the tree-map contents plus pointer).  Python
    ``dict`` objects are unhashable, so we build an equivalent frozen
    snapshot instead.
    """
    items = []
    for addr in sorted(tree.keys(), key=lambda a: a.address):
        node = tree[addr]
        labels_key = tuple(str(lab) for lab in node.labels)
        items.append((addr.address, labels_key))
    return (tree.pointer.address, tuple(items))


NON_REPAIRING_ACTION_TYPES = frozenset({"accept", "reject", "assert", "question"})


class InteractiveContextParser(DAGParser):
    """Best-first DS parser with explicit word-level context DAG (Eshghi et al. 2015)."""

    non_repairing_action_types = list(NON_REPAIRING_ACTION_TYPES)
    RELEASE_TURN = RELEASE_TURN_TOKEN
    WAIT = WAIT_TOKEN

    def __init__(
        self,
        resource_dir: str | Path,
        repairing: bool = False,
        participants: tuple[str, ...] = (DEFAULT_NAME,),
    ) -> None:
        p = Path(resource_dir)
        super().__init__(Lexicon(p), Grammar(p), SpeechActInferenceGrammar(p))
        dag = WordLevelContextDAG()
        parts = participants if participants else (DEFAULT_NAME,)
        self.context = Context(dag, self.sa_grammar, *parts)
        self.context.set_repair_processing(repairing)
        self.forced_restart = False
        self.forced_repair = False
        self.right_edge_indicators: list[str] = []
        self.acks: list[str] = ["uhu"]
        self.repairanda: list[str] = ["uhh", "errm", "err", "er", "uh", "erm", "uhm", "um", "oh"]
        self.forced_repairanda: list[str] = ["sorry", "oops", "wait", "erm"]
        self.restarters: list[str] = ["yeah"]

    @classmethod
    def from_loaded(
        cls,
        lexicon: Lexicon,
        grammar: Grammar,
        *,
        sa: SpeechActInferenceGrammar | None = None,
        participants: tuple[str, ...] = (DEFAULT_NAME,),
    ) -> InteractiveContextParser:
        """Build parser from in-memory `Lexicon` / `Grammar` (Java `Lexicon, Grammar` ctor)."""
        obj = cls.__new__(cls)
        DAGParser.__init__(obj, lexicon, grammar, sa or SpeechActInferenceGrammar(Path(".")))
        dag = WordLevelContextDAG()
        parts = participants if participants else (DEFAULT_NAME,)
        obj.context = Context(dag, obj.sa_grammar, *parts)
        obj.context.set_repair_processing(False)
        obj.forced_restart = False
        obj.forced_repair = False
        obj.right_edge_indicators = []
        obj.acks = ["uhu"]
        obj.repairanda = ["uhh", "errm", "err", "er", "uh", "erm", "uhm", "um", "oh"]
        obj.forced_repairanda = ["sorry", "oops", "wait", "erm"]
        obj.restarters = ["yeah"]
        return obj

    def get_name(self) -> str:
        return self.context.get_name()

    def _adjust_once(self, goal: Formula | None) -> bool:
        dag = self.get_state()
        if self.context.repair_initiated():
            logger.info("Repair initiated (not fully ported — failing adjust step)")
            return False
        if dag.out_degree(dag.get_current_tuple()) == 0:
            self._apply_all_permutations(goal)
        result: GroundableEdge | None = None
        while True:
            result = dag.go_first()
            if result is not None:
                break
            if not dag.attempt_backtrack():
                break
        return result is not None

    def parse_goal(self, goal: Formula | None) -> bool:
        dag = self.get_state()
        if dag.is_exhausted():
            logger.debug("state exhausted")
            return False
        while True:
            if not self._adjust_once(goal):
                logger.debug("wordstack: %s", dag.word_stack_ref())
                logger.debug("depth: %s", dag.get_depth())
                dag.set_exhausted(True)
                return False
            if not dag.word_stack:
                break
        return True

    def _apply_all_permutations(self, goal: Formula | None) -> None:
        dag = self.get_state()
        if not dag.word_stack:
            return
        word = dag.word_stack[-1]
        if word.word in self.right_edge_indicators:
            return
        if word.word in self.acks:
            return

        all_actions = self.lexicon.lookup(word.word)
        left_adjust: list[LexicalAction] = []
        current_tree = dag.get_current_tuple().get_tree().clone()
        for la in all_actions:
            if la.requires_left_adjustment():
                left_adjust.append(la)
                continue
            logger.debug("applying %s without left adjustment", la)
            ct = current_tree.clone()
            _repoint_for_verb_lexical(ct, la)
            res = la.exec(ct, self.context)
            if res is None:
                continue
            tup = dag.get_new_tuple(res)
            head_less = tup.get_semantics(self.context).remove_head()
            if goal is not None and len(dag.word_stack) == 1 and not head_less.subsumes(goal):
                continue
            edge_acts: list[Action] = [la.instantiate()]
            edge = dag.get_new_edge(edge_acts, word)
            dag.add_child(tup, edge)

        if not left_adjust:
            return

        init_pair = self.adjust_with_non_optional_grammar(
            ([], dag.get_current_tuple().get_tree().clone())
        )
        global_pairs: list[tuple[list[Action], Tree]] = [init_pair]
        tried: dict[str, set[tuple]] = {ca.name: set() for ca in self.optional_grammar.values()}
        idx = 0
        while idx < len(global_pairs):
            if len(global_pairs) > _MAX_LEXICAL_ADJUSTMENT_PAIRS:
                logger.warning(
                    "Lexical optional-grammar expansion exceeded %s pairs — stopping (avoid hang)",
                    _MAX_LEXICAL_ADJUSTMENT_PAIRS,
                )
                break
            cur_acts, cur_tree = global_pairs[idx]
            for ca in sorted(self.optional_grammar.values(), key=lambda x: x.name):
                tkey = _tree_structural_key(cur_tree)
                if tkey in tried[ca.name]:
                    continue
                tried[ca.name].add(tkey)
                nxt = ca.exec(cur_tree.clone(), self.context)
                if nxt is None:
                    continue
                new_acts = list(cur_acts)
                new_acts.append(ca.instantiate())
                adj = self.adjust_with_non_optional_grammar((new_acts, nxt))
                global_pairs.append(adj)
            idx += 1

        for pair_acts, pair_tree in global_pairs:
            for la in left_adjust:
                pt = pair_tree.clone()
                _repoint_for_verb_lexical(pt, la)
                res = la.exec(pt, self.context)
                if res is None:
                    continue
                f = res.get_maximal_semantics(self.context)
                head_less = f.remove_head()
                if goal is not None and len(dag.word_stack) == 1 and not head_less.subsumes(goal):
                    continue
                new_acts = list(pair_acts)
                new_acts.append(la.instantiate())
                edge = dag.get_new_edge(new_acts, word)
                child = dag.get_new_tuple(res)
                dag.add_child(child, edge)

    def init(self) -> None:
        self.forced_restart = False
        self.forced_repair = False
        super().init()

    def parse_word(self, w: UtteredWord) -> WordLevelContextDAG | None:
        participants = list(self.context.get_participants())
        if len(participants) == 2 and w.speaker in participants:
            i = participants.index(w.speaker)
            other = participants[1 - i]
            word = UtteredWord(w.word, w.speaker, other)
        else:
            word = UtteredWord(w.word, w.speaker, w.addressee)
        logger.info("Parsing word: %s", word)

        if word.word == self.WAIT:
            return self.get_state()

        if self.forced_restart or self.forced_repair:
            logger.info("restart/repair path not fully ported")
            self.forced_restart = False
            self.forced_repair = False
            return None

        if word.word in self.repairanda:
            self.get_state().this_is_first_tuple_after_last_word()
            self.get_state().set_repair_processing(True)
            return self.get_state()
        if word.word in self.restarters and self.get_state().repair_processing_enabled():
            self.forced_restart = True
            self.get_state().this_is_first_tuple_after_last_word()
            return self.get_state()
        if word.word in self.forced_repairanda:
            self.forced_repair = True
            self.get_state().this_is_first_tuple_after_last_word()
            self.get_state().set_repair_processing(True)
            return self.get_state()

        actions = self.lexicon.lookup(word.word)
        if not actions:
            logger.error("Word not in Lexicon: %s", word)
            return None

        self.get_state().word_stack_ref().append(word)
        if not self.parse_goal(None):
            logger.error("Cannot parse: %s — resetting", word.word)
            self.get_state().reset_to_first_tuple_after_last_word()
            if not self.get_state().repair_processing_enabled():
                return None
            self.get_state().word_stack_ref().append(word)
            self.get_state().initiate_local_repair()
            if not self.parse_goal(None):
                self.get_state().reset_to_first_tuple_after_last_word()
                return None

        if word.word == self.RELEASE_TURN:
            self.context.open_floor()
        else:
            self.context.set_who_has_floor(word.speaker)

        self.get_state().this_is_first_tuple_after_last_word()
        self.get_state().set_repair_processing(False)
        logger.info("Parsed: %s", word)
        return self.get_state()

    def parse_utterance(self, utt: Utterance) -> bool:
        """Parse each word in order (Java `DAGParser.parseUtterance`)."""
        ok = True
        for uw in utt.words:
            if self.parse_word(uw) is None:
                logger.error("Failed to parse %s", uw)
                ok = False
        return ok

    def get_best_tuple(self) -> DAGTuple:
        """Current DAG tuple (Java `getBestTuple`)."""
        return self.get_state().get_current_tuple()
