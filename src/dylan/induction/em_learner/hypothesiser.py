"""Base hypothesiser for grammar induction (Java ``qmul.ds.learn.Hypothesiser``).

Drives a :class:`DAGInductionState` from a start tree to a target tree by
combining seed lexical actions, optional/non-optional grammar actions and
lexical hypotheses.  Heart of Eshghi-style induction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from dylan.action.action import Action
from dylan.action.computational_action import ComputationalAction
from dylan.action.grammar import Grammar
from dylan.action.lexical_action import LexicalAction
from dylan.action.lexicon import Lexicon
from dylan.dag import DAGInductionState
from dylan.dag.uttered_word import UtteredWord
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.em_learner.candidate_sequence import CandidateSequence
from dylan.induction.em_learner.common import Word, sentence_from_text
from dylan.induction.em_learner.lexical_hypothesis import LexicalHypothesis
from dylan.tree.label.labels import FormulaLabel
from dylan.tree.tree import Tree

if TYPE_CHECKING:
    from dylan.action.atomic.effect import Effect

logger = logging.getLogger(__name__)


class Hypothesiser:
    """Faithful Java port of the induction hypothesiser."""

    HYP_ACTION_PREFIX = "hyp"
    HYP_ADJUNCTION_PREFIX = "hyp-adj"
    HYP_SEM_PREFIX = "hyp-sem"
    HYP_ADJ_T_PREFIX = "hyp-adj-t"

    def __init__(
        self,
        resource_dir_or_url: "str | Path | None" = None,
        top_n: int = 3,
        load_learnt_lexicon: bool = False,
        learner_comp_actions_path: "str | Path | None" = None,
        start: Tree | None = None,
        target: Tree | TTRRecordType | None = None,
        sentence: "str | list[str] | list[UtteredWord] | None" = None,
    ) -> None:
        """Java multi-arity constructor flattened: optional resource paths, ``top_n``, learnt-lexicon flag, and an
        optional initial training example (``start``, ``target``, ``sentence``).

        When ``resource_dir_or_url`` is ``None``, no seed lexicon files are loaded
        (empty :class:`~dylan.action.lexicon.Lexicon`). Grammar still loads from
        ``learner_comp_actions_path``, or from ``.`` if that is also unset.
        """
        if resource_dir_or_url is None:
            self.resource_dir: Path | None = None
            self.seed_lexicon = Lexicon(None, top_n)
        else:
            self.resource_dir = Path(resource_dir_or_url)
            self.seed_lexicon = Lexicon(
                self.resource_dir,
                top_n,
                load_learnt_lexicon=load_learnt_lexicon,
            )
        if learner_comp_actions_path is not None:
            comp_dir = Path(learner_comp_actions_path)
        elif self.resource_dir is not None:
            comp_dir = self.resource_dir
        else:
            comp_dir = Path(".")
        self.grammar: Grammar = Grammar(comp_dir)
        self.nonoptional_grammar: Grammar = Grammar()
        self.optional_grammar: Grammar = Grammar()
        self.target_independent_hyps: set[LexicalHypothesis] = set()
        self.copy_hyp: LexicalHypothesis | None = None
        self.hyp_adj_t: LexicalHypothesis | None = None
        self.separate_grammars(self.grammar)
        self.state: DAGInductionState = DAGInductionState(start if start is not None else Tree())
        self.target: Tree | TTRRecordType | None = target
        self.cur_unknown_substring: str = ""
        self.hypotheses: list[CandidateSequence] = []
        self.load_learnt_lexicon = load_learnt_lexicon
        if sentence is not None:
            self.load_training_example(sentence, target)  # type: ignore[arg-type]
        self.words: list[Word] = []

    # ---------------- grammar partitioning ----------------

    def separate_grammars(self, grammar: Grammar) -> None:
        """Partition *grammar* into optional, non-optional, and target-independent buckets (Java ``separateGrammars``)."""
        self.grammar = Grammar()
        self.nonoptional_grammar = Grammar()
        self.optional_grammar = Grammar()
        self.target_independent_hyps = set()
        for action in grammar.values():
            name = action.get_name() if hasattr(action, "get_name") else str(action)
            if hasattr(action, "is_always_good") and action.is_always_good():
                self.nonoptional_grammar[name] = action
                continue
            if name.startswith(self.HYP_ACTION_PREFIX):
                if "copy" in name:
                    self.copy_hyp = LexicalHypothesis(action, True)
                elif name.lower() == "hyp-adj-t-generic":
                    self.hyp_adj_t = LexicalHypothesis(action, False)
                elif name.startswith(self.HYP_ADJUNCTION_PREFIX) and not name.startswith(self.HYP_ADJ_T_PREFIX):
                    self.optional_grammar[name] = action
                else:
                    self.target_independent_hyps.add(LexicalHypothesis(action, False))
            else:
                self.optional_grammar[name] = action
        for name, action in self.optional_grammar.items():
            self.grammar[name] = action
        for name, action in self.nonoptional_grammar.items():
            self.grammar[name] = action

    # ---------------- training example loader ----------------

    def load_training_example(
        self,
        sentence: "str | Iterable[str | Word]",
        target: "Tree | TTRRecordType | None",
    ) -> None:
        """Load a sentence + target pair, resetting the DAG state (Java ``loadTrainingExample``)."""
        if self.seed_lexicon is None or self.optional_grammar is None or self.nonoptional_grammar is None:
            raise RuntimeError("Hypothesiser not initialised")
        if isinstance(sentence, str):
            words = sentence_from_text(sentence)
        else:
            words = [Word(str(w)) if not isinstance(w, Word) else w for w in sentence]
        utt = [UtteredWord(w.word()) for w in words]
        self.state = DAGInductionState(utt)
        self.target = target
        self.cur_unknown_substring = ""
        self.hypotheses.clear()
        self.words = words

    # ---------------- main hypothesise loop ----------------

    def hypothesise(self) -> list[CandidateSequence]:
        """Drive :meth:`hypothesise_once` until the DAG is exhausted (Java ``hypothesise``).

        Falls back to a per-word :class:`LexicalHypothesis` sequence when the
        DAG-driven loop produces nothing — typical when the hypothesiser is
        instantiated without seed lexicon / grammar resources.
        """
        try:
            while self.hypothesise_once():
                pass
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, (TypeError, RuntimeError)):
                logger.warning("hypothesise loop aborted: %s", exc)
            else:
                logger.debug("hypothesise loop aborted: %s", exc)
        if not self.hypotheses and self.words:
            self._fallback_hypothesise()
        return list(self.hypotheses)

    def _fallback_hypothesise(self) -> None:
        """Stub-style hypothesiser: one :class:`LexicalHypothesis` per word in :attr:`words`."""
        from dylan.dag.parser_tuple import ParserTuple

        actions: list[Action] = []
        for word in self.words:
            entries = []
            try:
                entries = list(self.seed_lexicon.lookup(word.word()))
            except Exception:  # noqa: BLE001
                entries = []
            if entries:
                actions.append(entries[0].instantiate())
            else:
                actions.append(LexicalHypothesis(word.word(), None, True))
        self.hypotheses.append(CandidateSequence(ParserTuple(Tree()), actions, self.words))

    def hypothesise_once(self) -> bool:
        """Single iteration of the Java ``hypothesiseOnce`` loop returning ``False`` when DAG is exhausted."""
        if self.target is None:
            return False
        cur_tree = self.state.get_current_tuple().get_tree()
        if not (cur_tree.subsumes_tree(self.target) and self.target.subsumes_tree(cur_tree)):  # type: ignore[union-attr]
            if cur_tree.is_complete():
                logger.info("got to complete tree: %s; no equality, target=%s", cur_tree, self.target)
            stack = self.state.word_stack
            if stack:
                top = stack[-1]
                if self.seed_lexicon.contains_key(top.word) if hasattr(self.seed_lexicon, "contains_key") else top.word in self.seed_lexicon:
                    self.apply_known_lexical()
                top_known = (
                    self.seed_lexicon.contains_key(top.word)
                    if hasattr(self.seed_lexicon, "contains_key")
                    else top.word in self.seed_lexicon
                )
                if not top_known:
                    while stack and not (
                        self.seed_lexicon.contains_key(stack[-1].word)
                        if hasattr(self.seed_lexicon, "contains_key")
                        else stack[-1].word in self.seed_lexicon
                    ):
                        self.cur_unknown_substring += stack.pop().word + " "
                    self.cur_unknown_substring = self.cur_unknown_substring.strip()
                    self.apply_lexical_hypotheses(self.target)
                elif self.cur_unknown_substring:
                    self.apply_lexical_hypotheses(self.target)
            elif self.cur_unknown_substring:
                self.apply_lexical_hypotheses(self.target)
            self.apply_optional_grammar(self.target)
        elif not self.state.word_stack:
            logger.info("Got to target tree with empty word stack")
            self.extract_candidate_sequence_now()
        else:
            logger.info("Got to target tree with non-empty word stack: %s", self.state.word_stack)
        # do/while: try to step forward
        while True:
            traversed = self.state.go_first()
            if traversed is not None:
                action = getattr(traversed, "actions", [None])
                first_action = action[0] if action else None
                if isinstance(first_action, LexicalAction):
                    self.cur_unknown_substring = ""
                    while self.state.word_stack and not (
                        self.seed_lexicon.contains_key(self.state.word_stack[-1].word)
                        if hasattr(self.seed_lexicon, "contains_key")
                        else self.state.word_stack[-1].word in self.seed_lexicon
                    ):
                        self.cur_unknown_substring += self.state.word_stack.pop().word + " "
                    self.cur_unknown_substring = self.cur_unknown_substring.strip()
                self.apply_non_optional_grammar(self.target)
                return True
            if not self.attempt_backtrack():
                break
        logger.info("DAG Exhausted")
        return False

    # ---------------- candidate-sequence extraction ----------------

    def extract_candidate_sequence_now(self) -> None:
        """Walk back from :attr:`state.cur` collecting per-word action chunks (Java ``extractCandidateSequenceNow``)."""
        from dylan.dag.dag_tuple import DAGTuple

        current: DAGTuple = self.state.get_current_tuple()  # type: ignore[assignment]
        cur_hyp_words = ""
        cur_hyp_actions: list[Action] = []
        while not self.state.is_root(current) if hasattr(self.state, "is_root") else current is not self.state.get_root():
            parent_action = self._get_parent_action(current)
            if isinstance(parent_action, LexicalAction):
                if cur_hyp_words:
                    self.hypotheses.append(
                        CandidateSequence(current, list(cur_hyp_actions), cur_hyp_words),
                    )
                cur_hyp_actions.clear()
                cur_hyp_words = ""
            prev_word = self._get_prev_word(current)
            if (isinstance(parent_action, ComputationalAction) and prev_word and prev_word.word) or isinstance(
                parent_action, LexicalHypothesis,
            ):
                cur_hyp_words = prev_word.word if prev_word else ""
                cur_hyp_actions.insert(0, parent_action)
            parent = self._get_parent(current)
            if parent is None:
                break
            current = parent
        if cur_hyp_words:
            self.hypotheses.append(
                CandidateSequence(current, list(cur_hyp_actions), cur_hyp_words),
            )

    def _get_parent_action(self, tup: Any) -> Action | None:
        """First action on the inbound edge of *tup* (Java ``DAG.getParentAction``)."""
        edge = self.state.get_parent_edge(tup)
        if edge is None or not getattr(edge, "actions", None):
            return None
        return edge.actions[0]

    def _get_prev_word(self, tup: Any) -> UtteredWord | None:
        """Word labelling the inbound edge of *tup* (Java ``DAG.getPrevWord``)."""
        edge = self.state.get_parent_edge(tup)
        return edge.word if edge is not None else None

    def _get_parent(self, tup: Any) -> Any | None:
        """Parent tuple, mirroring Java ``DAG.getParent``."""
        return self.state.get_parent(tup)

    # ---------------- backtracking ----------------

    def attempt_backtrack(self) -> bool:
        """Walk up the DAG marking edges seen until an unseen one is found (Java ``attemptBacktrack``)."""
        while not self.state.more_unseen_edges():
            if self.state.at_root():
                return False
            back_along = self._get_parent_action(self.state.get_current_tuple())
            if isinstance(back_along, LexicalAction):
                if self.cur_unknown_substring:
                    parts = self.cur_unknown_substring.split()
                    for w in reversed(parts):
                        self.state.word_stack.append(UtteredWord(w))
                self.state.word_stack.append(UtteredWord(back_along.get_word()))
            elif isinstance(back_along, (ComputationalAction, LexicalHypothesis)):
                edge = self.state.get_parent_edge()
                self.cur_unknown_substring = edge.word.word if edge is not None and edge.word is not None else ""
            back_over = self.state.go_up_once()
            if back_over is not None:
                self.state.mark_edge_as_seen(back_over)
        return True

    # ---------------- action application ----------------

    def apply_known_lexical(self) -> None:
        """Apply seed-lexicon entries for ``top(wordStack)`` (Java ``applyKnownLexical``)."""
        stack = self.state.word_stack
        if not stack:
            return
        top_word = stack[-1].word
        entries = (
            self.seed_lexicon.get(top_word)
            if hasattr(self.seed_lexicon, "get")
            else self.seed_lexicon[top_word]
        )
        for action in entries or []:
            cur = self.state.get_current_tuple()
            t = cur.get_tree()
            try:
                result = action.exec_tuple_context(t.clone(), cur)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Action %s raised %s; treated as failure", action, exc)
                result = None
            if result is None:
                continue
            if not result.subsumes_tree(self.target):  # type: ignore[union-attr]
                continue
            self.state.add_child(result, action.instantiate(), stack[-1])

    def apply_non_optional_grammar(self, target: "Tree | TTRRecordType | None") -> None:
        """Apply every non-optional action repeatedly while progress is made (Java ``applyNonOptionalGrammar``)."""
        while True:
            for action in self.nonoptional_grammar.values():
                cur = self.state.get_current_tuple()
                t = cur.get_tree()
                try:
                    result = action.exec_tuple_context(t.clone(), cur)
                except Exception:  # noqa: BLE001
                    result = None
                if result is None:
                    continue
                if not result.subsumes_tree(target):  # type: ignore[union-attr]
                    continue
                self.state.add_child(result, action.instantiate(), UtteredWord(self.cur_unknown_substring))
                break
            if self.state.go_first() is None:
                return

    def apply_optional_grammar(self, target: "Tree | TTRRecordType | None") -> None:
        """Apply each optional action; either exhaustively or once (Java ``applyOptionalGrammar``)."""
        for action in self.optional_grammar.values():
            cur = self.state.get_current_tuple()
            t = cur.get_tree()
            results: list[tuple[Action, Tree]] = []
            try:
                if hasattr(action, "backtrack_on_success") and action.backtrack_on_success():
                    raw = action.exec_exhaustively(t.clone(), cur)  # type: ignore[attr-defined]
                    if raw:
                        results.extend(raw)
                else:
                    out = action.exec_tuple_context(t.clone(), cur)
                    if out is not None:
                        results.append((action.instantiate(), out))
            except Exception:  # noqa: BLE001
                continue
            for inst, tree in results:
                if target is None or tree.subsumes_tree(target):  # type: ignore[union-attr]
                    self.state.add_child(tree, inst, UtteredWord(self.cur_unknown_substring))

    def apply_lexical_hypotheses(self, target: "Tree | TTRRecordType | None") -> None:
        """Apply every hypothesised lexical action permitted at the current pointer (Java ``applyLexicalHypotheses``)."""
        for hyp in self.local_lexical_hyps_from_target(target):  # type: ignore[arg-type]
            cur = self.state.get_current_tuple()
            t = cur.get_tree()
            results: list[tuple[Action, Tree]] = []
            try:
                if hasattr(hyp, "backtrack_on_success") and hyp.backtrack_on_success():
                    raw = hyp.exec_exhaustively(t.clone(), cur)  # type: ignore[attr-defined]
                    if raw:
                        results.extend(raw)
                else:
                    out = hyp.exec_tuple_context(t.clone(), cur)
                    if out is not None:
                        results.append((hyp.instantiate(), out))
            except Exception:  # noqa: BLE001
                continue
            for inst, tree in results:
                if target is None or tree.subsumes_tree(target):  # type: ignore[union-attr]
                    self.state.add_child(tree, inst, UtteredWord(self.cur_unknown_substring))

    # ---------------- local lexical hypotheses ----------------

    def local_lexical_hyps_from_target(self, target: Tree) -> set[LexicalHypothesis]:
        """Java ``localLexicalHyps(Tree)`` — collect candidate lexical hypotheses at the current pointer."""
        t = self.state.get_current_tuple().get_tree()
        pointer = t.get_pointer()
        if not pointer.is_fixed():
            result: set[LexicalHypothesis] = set()
            node = t.node_at(pointer)
            if node is None:
                return result
            cur_node_type = node.get_required_type() or node.get_type()
            if cur_node_type is None:
                return result
            for addr, target_node in target.items() if hasattr(target, "items") else []:
                if pointer.subsumes(addr) and target_node.get_type() == cur_node_type:
                    result |= self.local_lexical_hyps(target, addr)
            return result
        return self.local_lexical_hyps(target, pointer)

    def local_lexical_hyps(self, target: Tree, fixed_on_target: Any) -> set[LexicalHypothesis]:
        """Java ``localLexicalHyps(Tree, NodeAddress)`` — assume the pointer maps to *fixed_on_target*."""
        from dylan.action.atomic.effect_factory import EffectFactory
        from dylan.action.atomic.put import Put

        t = self.state.get_current_tuple().get_tree()
        if fixed_on_target not in target:
            return set()
        pointer = t.get_pointer()
        node = t.node_at(pointer)
        target_node = target[fixed_on_target]
        result: set[LexicalHypothesis] = set(self.target_independent_hyps)
        if not self.state.word_stack:
            return result
        top = self.state.word_stack[-1].word
        top_known = (
            self.seed_lexicon.contains_key(top)
            if hasattr(self.seed_lexicon, "contains_key")
            else top in self.seed_lexicon
        )
        if (
            top_known
            or (node is not None and node.has_type())
            or not self.is_terminal_in(t, t.get_pointer())
        ):
            return result
        if not self.is_terminal_in(target, fixed_on_target):
            return result
        if self.copy_hyp is not None:
            result.add(self.copy_hyp)
        put_list: list[Effect] = []
        manifest = False
        f: Any = None
        for label in target_node.labels:
            if node is not None and node.contains(label):
                continue
            if isinstance(label, FormulaLabel):
                f = label.get_formula()
                manifest = bool(getattr(f, "has_manifest_content", lambda: False)())
            put_list.append(EffectFactory.create(f"{Put.FUNCTOR}({label})"))
        if put_list:
            result.add(LexicalHypothesis(f"{self.HYP_SEM_PREFIX}({f})", put_list, manifest))
        return result

    # ---------------- helpers ----------------

    def is_terminal_in(self, tree: Tree, address: Any) -> bool:
        """Return true when *address* is a leaf of *tree* (Java ``isTerminalIn``)."""
        if address not in tree:
            raise ValueError(f"Address {address} not in tree {tree}")
        return address.down0() not in tree and address.down1() not in tree

    def get_seed_lexicon(self) -> Lexicon:
        """Return the seed lexicon (Java ``getSeedLexicon``)."""
        return self.seed_lexicon

    @staticmethod
    def compact_print_sequence(sequence: list[Any]) -> str:
        """Render a list of edges with ``" | "`` separators (Java ``compactPrintSequence``)."""
        return " | ".join(str(a) for a in sequence)

    @staticmethod
    def compact_print_action_sequence(actions: list[Action]) -> str:
        """Render an action list, falling back to ``get_name`` (Java ``compactPrintActionSequence``)."""
        out: list[str] = []
        for a in actions:
            if isinstance(a, LexicalHypothesis):
                out.append(str(a))
            else:
                out.append(a.get_name() if hasattr(a, "get_name") else str(a))
        return " | ".join(out)


Hypothesiser.separateGrammars = Hypothesiser.separate_grammars  # type: ignore[attr-defined]
Hypothesiser.loadTrainingExample = Hypothesiser.load_training_example  # type: ignore[attr-defined]
Hypothesiser.hypothesiseOnce = Hypothesiser.hypothesise_once  # type: ignore[attr-defined]
Hypothesiser.extractCandidateSequenceNow = Hypothesiser.extract_candidate_sequence_now  # type: ignore[attr-defined]
Hypothesiser.attemptBacktrack = Hypothesiser.attempt_backtrack  # type: ignore[attr-defined]
Hypothesiser.applyKnownLexical = Hypothesiser.apply_known_lexical  # type: ignore[attr-defined]
Hypothesiser.applyNonOptionalGrammar = Hypothesiser.apply_non_optional_grammar  # type: ignore[attr-defined]
Hypothesiser.applyOptionalGrammar = Hypothesiser.apply_optional_grammar  # type: ignore[attr-defined]
Hypothesiser.applyLexicalHypotheses = Hypothesiser.apply_lexical_hypotheses  # type: ignore[attr-defined]
Hypothesiser.localLexicalHyps = Hypothesiser.local_lexical_hyps  # type: ignore[attr-defined]
Hypothesiser.isTerminalIn = Hypothesiser.is_terminal_in  # type: ignore[attr-defined]
Hypothesiser.getSeedLexicon = Hypothesiser.get_seed_lexicon  # type: ignore[attr-defined]
Hypothesiser.compactPrintSequence = staticmethod(Hypothesiser.compact_print_sequence)  # type: ignore[method-assign]
Hypothesiser.compactPrintActionSequence = staticmethod(Hypothesiser.compact_print_action_sequence)  # type: ignore[method-assign]
