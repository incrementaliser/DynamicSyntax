"""TTR-specific hypothesiser for EM induction (Java ``qmul.ds.learn.TTRHypothesiser``).

Drives a :class:`DAGInductionState` using a :class:`TypeLattice` of TTR
increments to incrementally hypothesise tree templates plus lexical actions
that lead to ``targetType``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from dylan.action.action import Action
from dylan.action.atomic.effect import Effect
from dylan.action.atomic.effect_factory import EffectFactory
from dylan.action.atomic.put import Put
from dylan.action.atomic.ttr_fresh_put import TTRFreshPut
from dylan.action.lexical_action import LexicalAction
from dylan.dag import DAGInductionState
from dylan.dag.dag_induction_tuple import DAGInductionTuple
from dylan.dag.type_lattice import TypeLattice
from dylan.dag.type_lattice_increment import TypeLatticeIncrement
from dylan.dag.uttered_word import UtteredWord
from dylan.formula.ttr_formula import TTRFormula
from dylan.formula.ttr_record_type import HEAD, TTRRecordType
from dylan.induction.em_learner.candidate_sequence import CandidateSequence
from dylan.induction.em_learner.common import Word, sentence_from_text
from dylan.induction.em_learner.hypothesiser import Hypothesiser
from dylan.induction.em_learner.lexical_hypothesis import LexicalHypothesis
from dylan.induction.em_learner.tree_hypothesis import TreeHypothesis
from dylan.tree.basic_operator import BasicOperator
from dylan.tree.label.labels import FormulaLabel, Requirement, TypeLabel
from dylan.tree.tree import Tree
from dylan.type.dstype import DSType

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Optional structure actions that Java applies even when forward subsumption is partial (open-a-door trace).
_OPTIONAL_WITHOUT_SUBSUMPTION_GATE: frozenset[str] = frozenset(
    {"completion", "intro-pred", "anticipation0", "anticipation1"},
)


class TTRHypothesiser(Hypothesiser):
    """Hypothesise action sequences from axiom tree to target TTR semantics."""

    HYP_SEM_PREFIX = Hypothesiser.HYP_SEM_PREFIX
    HYP_ADJUNCTION_PREFIX = Hypothesiser.HYP_ADJUNCTION_PREFIX
    HYP_ADJ_T_PREFIX = Hypothesiser.HYP_ADJ_T_PREFIX
    def __init__(
        self,
        resource_dir_or_url: "str | Path | None" = None,
        top_n: int = 3,
        load_learnt_lexicon: bool = False,
        learner_comp_actions_path: "str | Path | None" = None,
        rt: TTRRecordType | None = None,
        sent: str | None = None,
    ) -> None:
        """Initialise the TTR hypothesiser.

        Mirrors the Java multi-arity constructors. When ``rt``+``sent`` are
        given, immediately load that training example.
        """
        super().__init__(resource_dir_or_url, top_n, load_learnt_lexicon, learner_comp_actions_path)
        self.target_type: TTRRecordType | None = None
        self.lattice: TypeLattice | None = None
        self.all_words: list[Word] = []
        self.word_index: int = 0
        if rt is not None and sent is not None:
            self.load_training_example(sent, rt)

    # ---------------- training example ----------------

    def load_training_example(
        self,
        sentence: "str | Iterable[str | Word]",
        target: TTRRecordType,
    ) -> None:
        """Load a sentence + TTR target, build the lattice, run :meth:`initialise` (Java ``loadTrainingExample``)."""
        if self.seed_lexicon is None or self.optional_grammar is None or self.nonoptional_grammar is None:
            raise RuntimeError("Hypothesiser not initialised")
        self.all_words = (
            sentence_from_text(sentence)
            if isinstance(sentence, str)
            else [Word(str(w)) if not isinstance(w, Word) else w for w in sentence]
        )
        utt = [UtteredWord(w.word()) for w in self.all_words]
        self.state = DAGInductionState(utt, gold_target=target)
        self.target = target
        self.target_type = target
        self.cur_unknown_substring = ""
        self.lattice = TypeLattice(target)
        self.hypotheses = []
        self.word_index = 0
        self.words = self.all_words
        self.initialise()

    # ---------------- lattice initialisation ----------------

    def initialise(self) -> None:
        """Seed the DAG with tree-hypothesis edges from the lattice's first increments (Java ``initialise``)."""
        if self.target_type is None or self.lattice is None:
            return
        head_field = self.target_type.get_head_field()
        head_label = head_field.label if head_field is not None else HEAD
        try:
            inc_set = self.lattice.get_increments(head_label)
        except Exception as exc:  # noqa: BLE001
            logger.debug("lattice.get_increments raised %s", exc)
            return
        for inc_list in inc_set:
            whole_inc = self.flatten(inc_list)
            head_f = whole_inc.get_head_field()
            filtered = head_f is not None and head_f.ds_type == DSType.es
            try:
                trees = whole_inc.get_maximal_filtered_abstractions(
                    self.state.get_current_tuple().get_tree().get_pointer(), DSType.t, filtered,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("get_maximal_filtered_abstractions raised %s", exc)
                continue
            for tree in trees:
                tree_hyp = TreeHypothesis(list(inc_list), tree)
                child = DAGInductionTuple(self.state.get_current_tuple().get_tree().clone())
                cur_target = getattr(self.state.get_current_tuple(), "get_target_tree", lambda: Tree())()
                cur_nonhead = getattr(self.state.get_current_tuple(), "get_non_head_target", lambda: Tree())()
                merged_inc = cur_target.merge_tree_put_all(tree_hyp.get_tree()) if hasattr(cur_target, "merge_tree_put_all") else tree
                merged_nonhead = cur_nonhead.merge_tree_put_all(tree_hyp.get_tree()) if hasattr(cur_nonhead, "merge_tree_put_all") else tree
                child_added = self.state.add_child(child, tree_hyp, None)
                if hasattr(child_added, "set_target"):
                    child_added.set_target(merged_inc)
                if hasattr(child_added, "set_non_head_target"):
                    child_added.set_non_head_target(merged_nonhead)

    # ---------------- tree hypothesis variants ----------------

    def apply_tree_hypotheses_entity(self) -> None:
        """Branch case: pointer points at an entity node (Java ``applyTreeHypothesesEntity``)."""
        if self.lattice is None:
            return
        cur = self.state.get_current_tuple()
        target_tree = cur.get_target_tree() if hasattr(cur, "get_target_tree") else Tree()
        pointed_addr = target_tree.get_pointer().up() if hasattr(target_tree, "get_pointer") else None
        pointed = target_tree[pointed_addr] if pointed_addr in target_tree else None
        pointed_type: TTRFormula | None = pointed.get_formula() if pointed is not None and hasattr(pointed, "get_formula") else None
        head_label = pointed_type.get_head_field().label if pointed_type is not None and pointed_type.get_head_field() else HEAD
        inc_set = self.lattice.get_increments(head_label)
        for inc in inc_set:
            whole_inc = self.flatten(inc)
            inc_head_label = whole_inc.get_head_field().label if whole_inc.get_head_field() else HEAD
            if inc_head_label == HEAD:
                head_increment = whole_inc
            else:
                whole_inc = whole_inc.clone()
                whole_inc.remove_field_by_label(HEAD) if hasattr(whole_inc, "remove_field_by_label") else None
                head_increment = whole_inc.substitute(inc_head_label, HEAD) if hasattr(whole_inc, "substitute") else whole_inc
            try:
                trees = head_increment.get_maximal_filtered_abstractions(
                    self.state.get_current_tuple().get_tree().get_pointer().up(), DSType.t, False,
                )
            except Exception:  # noqa: BLE001
                continue
            for tree in trees:
                tree_hyp = TreeHypothesis(list(inc), tree)
                child = DAGInductionTuple(self.state.get_current_tuple().get_tree().clone())
                merged_inc = cur.get_target_tree().merge_tree_put_all(tree_hyp.get_tree()) if hasattr(cur.get_target_tree(), "merge_tree_put_all") else tree
                merged_nonhead = cur.get_non_head_target().merge_tree_put_all(tree_hyp.get_tree()) if hasattr(cur.get_non_head_target(), "merge_tree_put_all") else tree
                child_added = self.state.add_child(child, tree_hyp, None)
                if hasattr(child_added, "set_target"):
                    child_added.set_target(merged_inc)
                if hasattr(child_added, "set_non_head_target"):
                    child_added.set_non_head_target(merged_nonhead)

    def apply_tree_hypotheses_event(self) -> None:
        """Branch case: pointer is at an event-type increment (Java ``applyTreeHypothesesEvent``)."""
        if self.lattice is None:
            return
        cur = self.state.get_current_tuple()
        cur_tree = cur.get_tree()
        cur_nonhead = cur.get_non_head_target() if hasattr(cur, "get_non_head_target") else Tree()
        try:
            pointed_on_target = cur_nonhead[cur_tree.get_pointer().up()]
        except Exception:  # noqa: BLE001
            pointed_on_target = None
        pointed_type: TTRFormula | None = pointed_on_target.get_formula() if pointed_on_target is not None and hasattr(pointed_on_target, "get_formula") else None
        head_label = (
            HEAD if pointed_type is None or pointed_type.get_head_field() is None else pointed_type.get_head_field().label
        )
        inc_set = self.lattice.get_increments(head_label)
        for inc in inc_set:
            whole_inc = self.flatten(inc)
            try:
                non_head_trees = whole_inc.get_maximal_filtered_abstractions(
                    cur_tree.get_pointer(), DSType.t, False,
                )
            except Exception:  # noqa: BLE001
                continue
            inc_head = whole_inc.get_head_field()
            inc_head_label = inc_head.label if inc_head is not None else HEAD
            if inc_head_label == HEAD:
                head_increment = whole_inc
            else:
                wcopy = whole_inc.clone()
                wcopy.remove_head() if hasattr(wcopy, "remove_head") else None
                head_increment = wcopy.substitute(inc_head_label, HEAD) if hasattr(wcopy, "substitute") else wcopy
            try:
                trees = head_increment.get_maximal_filtered_abstractions(
                    cur_tree.get_pointer(), DSType.t, False,
                )
            except Exception:  # noqa: BLE001
                continue
            for i, tree in enumerate(trees):
                non_head_tree = non_head_trees[i] if i < len(non_head_trees) else tree
                tree_hyp = TreeHypothesis(list(inc), tree)
                child = DAGInductionTuple(cur_tree.clone())
                merged_inc = cur.get_target_tree().merge_tree_put_all(tree) if hasattr(cur.get_target_tree(), "merge_tree_put_all") else tree
                merged_nonhead = cur_nonhead.merge_tree_put_all(non_head_tree) if hasattr(cur_nonhead, "merge_tree_put_all") else non_head_tree
                child_added = self.state.add_child(child, tree_hyp, None)
                if hasattr(child_added, "set_target"):
                    child_added.set_target(merged_inc)
                if hasattr(child_added, "set_non_head_target"):
                    child_added.set_non_head_target(merged_nonhead)

    def apply_tree_hypotheses_cn(self) -> None:
        """Branch case: pointer's grandparent is an entity, indicating a CN restrictor (Java ``applyTreeHypothesesCN``)."""
        if self.lattice is None:
            return
        cur = self.state.get_current_tuple()
        pointer = cur.get_tree().get_pointer()
        try:
            up_one_up_lup0 = pointer.go(BasicOperator.UP).go(BasicOperator.UP).go(BasicOperator.UP)
        except Exception:  # noqa: BLE001
            return
        node = cur.get_tree()[up_one_up_lup0] if up_one_up_lup0 in cur.get_tree() else None
        if node is None:
            return
        up_type = node.get_type() or node.get_required_type()
        if up_type is None:
            return
        if up_type == DSType.e or up_type == DSType.es:
            target_tree = cur.get_target_tree() if hasattr(cur, "get_target_tree") else Tree()
            mother = target_tree[target_tree.get_pointer().up()] if hasattr(target_tree, "get_pointer") else None
            e_type: TTRRecordType | None = mother.get_formula() if mother is not None else None
            if e_type is None or e_type.get_head_field() is None:
                inc_set = self.lattice.get_head_increments()
            else:
                paths = e_type.get_head_field().ttr_paths if hasattr(e_type.get_head_field(), "ttr_paths") else []
                if not paths:
                    inc_set = self.lattice.get_head_increments()
                else:
                    r_dot_head = paths[0]
                    first_label = r_dot_head.get_first_label() if hasattr(r_dot_head, "get_first_label") else HEAD
                    inc_set = self.lattice.get_increments(first_label)
        else:
            inc_set = self.lattice.get_head_increments()
        for inc in inc_set:
            whole_inc = self.flatten(inc)
            try:
                trees = whole_inc.get_maximal_filtered_abstractions(
                    cur.get_tree().get_pointer().up(), DSType.cn, False,
                )
            except Exception:  # noqa: BLE001
                continue
            for tree in trees:
                tree_hyp = TreeHypothesis(list(inc), tree)
                child = DAGInductionTuple(cur.get_tree().clone())
                merged_inc = cur.get_target_tree().merge_tree_put_all(tree_hyp.get_tree()) if hasattr(cur.get_target_tree(), "merge_tree_put_all") else tree
                merged_nonhead = cur.get_non_head_target().merge_tree_put_all(tree_hyp.get_tree()) if hasattr(cur.get_non_head_target(), "merge_tree_put_all") else tree
                child_added = self.state.add_child(child, tree_hyp, None)
                if hasattr(child_added, "set_target"):
                    child_added.set_target(merged_inc)
                if hasattr(child_added, "set_non_head_target"):
                    child_added.set_non_head_target(merged_nonhead)

    # ---------------- candidate-sequence extraction ----------------

    def _extraction_semantics(self, tup: DAGInductionTuple) -> TTRRecordType:
        """Return induction maximal semantics with metavar domains bound for extraction checks."""
        from dylan.induction.em_learner.induction_semantics import bind_metavar_path_domains

        try:
            sem = tup.get_semantics(tup)
        except Exception:  # noqa: BLE001
            sem = TTRRecordType()
        gold = tup.get_gold_target_type() if hasattr(tup, "get_gold_target_type") else self.target_type
        if not isinstance(sem, TTRRecordType) or gold is None:
            return sem if isinstance(sem, TTRRecordType) else TTRRecordType()
        bound = bind_metavar_path_domains(sem.clone(), gold)
        return bound if isinstance(bound, TTRRecordType) else TTRRecordType()

    def extract_sequence(self) -> CandidateSequence:
        """Walk back to the root collecting every non-tree-hyp action (Java ``extractSequence``)."""
        from dylan.dag.parser_tuple import ParserTuple

        current = self.state.get_current_tuple()
        sequence: list[Action] = []
        while not (self.state.is_root(current) if hasattr(self.state, "is_root") else current is self.state.get_root()):
            behind = self.state.get_parent_edge(current)
            if behind is not None and behind.actions and not isinstance(behind.actions[0], TreeHypothesis):
                sequence.insert(0, behind.actions[0])
            parent = self.state.get_parent(current)
            if parent is None:
                break
            current = parent
        return CandidateSequence(ParserTuple(), sequence, list(self.all_words))

    # ---------------- TTR overrides (Java ``TTRHypothesiser``) ----------------

    def _result_tree_subsumes_target(self, result: Tree, tup: DAGInductionTuple) -> bool:
        """Return whether result maximal semantics subsume ``target_type`` (Java apply_* subsumption gate)."""
        if self.target_type is None:
            return False
        try:
            max_sem = result.get_maximal_semantics(tup)
        except Exception:  # noqa: BLE001
            try:
                max_sem = result.get_maximal_semantics(None)
            except Exception:  # noqa: BLE001
                return False
        return bool(getattr(max_sem, "subsumes", lambda _: False)(self.target_type))

    def apply_known_lexical(self) -> None:
        """Apply seed lexicon for stack top using semantics vs :attr:`target_type` (Java ``applyKnownLexical``)."""
        stack = self.state.word_stack
        if not stack or self.target_type is None:
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
            if not self._result_tree_subsumes_target(result, cur):
                continue
            self.state.add_child(result, action.instantiate(), stack[-1])

    def apply_non_optional_grammar(self, target: Tree | None = None) -> None:
        """Chain non-optional actions while each result's maximal semantics still subsume ``target_type`` (Java)."""
        if self.target_type is None:
            return
        traversed: Any = None
        while True:
            for action in self.nonoptional_grammar.values():
                cur = self.state.get_current_tuple()
                tree = cur.get_tree()
                try:
                    result = action.exec_tuple_context(tree.clone(), cur)
                except Exception:  # noqa: BLE001
                    result = None
                if result is None:
                    continue
                if not self._result_tree_subsumes_target(result, cur):
                    always_good = getattr(action, "is_always_good", lambda: False)()
                    if not (always_good() if callable(always_good) else bool(always_good)):
                        continue
                self.state.add_child(result, action, None)
                break
            traversed = self.state.go_first()
            if traversed is not None:
                logger.info("Gone forward first along: %s", traversed.actions[0] if traversed.actions else traversed)
            if traversed is None:
                return

    def apply_optional_grammar(self, target: Tree | None = None) -> None:
        """Apply optional grammar actions (Java ``applyOptionalGrammar`` with structure-action whitelist)."""
        if self.target_type is None:
            return
        for action in self.optional_grammar.values():
            cur = self.state.get_current_tuple()
            tree = cur.get_tree()
            results: list[tuple[Any, Tree]] = []
            try:
                if hasattr(action, "backtracks_on_success") and action.backtracks_on_success():
                    raw = action.exec_exhaustively(tree.clone(), cur)  # type: ignore[attr-defined]
                    if raw:
                        results.extend(raw)
                else:
                    out = action.exec_tuple_context(tree.clone(), cur)
                    if out is not None:
                        results.append((action.instantiate(), out))
            except Exception:  # noqa: BLE001
                continue
            for inst, res_tree in results:
                if not self._result_tree_subsumes_target(res_tree, cur):
                    inst_name = inst.get_name() if hasattr(inst, "get_name") else ""
                    if inst_name not in _OPTIONAL_WITHOUT_SUBSUMPTION_GATE:
                        continue
                self.state.add_child(res_tree, inst, None)

    def local_lexical_hyps(self, target: Tree, fixed_on_target: Any | None = None) -> set[LexicalHypothesis]:
        """Java ``localLexicalHyps`` — one-arg collection or two-arg at a fixed target address."""
        if fixed_on_target is None:
            return self.local_lexical_hyps_from_target(target)
        return self._local_lexical_hyps_at(target, fixed_on_target)

    def local_lexical_hyps_from_target(self, target: Tree) -> set[LexicalHypothesis]:
        """Collect lexical hyps at the pointer (Java one-arg); also CN terminals when pointer is unfixed."""
        parse_tree = self.state.get_current_tuple().get_tree()
        pointer = parse_tree.get_pointer()
        if pointer.is_fixed():
            return self._local_lexical_hyps_at(target, pointer)
        result = Hypothesiser.local_lexical_hyps_from_target(self, target)
        node = parse_tree.node_at(pointer)
        if node is None:
            return result
        cur_type = node.get_required_type() or node.get_type()
        if cur_type is None:
            return result
        for addr, target_node in target.items():
            if not pointer.subsumes(addr):
                continue
            if target_node.get_type() == cur_type:
                continue
            if self.is_terminal_in(target, addr):
                result |= self._local_lexical_hyps_at(target, addr)
        return result

    def apply_lexical_hypotheses(self, target: Tree | None = None) -> None:
        """Apply local lexical hypotheses with TTR subsumption and adjunction specials (Java ``applyLexicalHypotheses``)."""
        tgt = target if target is not None else Tree()
        if self.target_type is None:
            return
        def _lexical_hyp_apply_order(h: LexicalHypothesis) -> tuple[int, str]:
            """Prefer ``hyp-build-cn-e-1`` over ``hyp-build-cn-e-0`` (Java search order)."""
            name = h.get_name()
            if "hyp-build-cn-e-1" in name:
                return (0, name)
            if "hyp-build-cn-e-0" in name:
                return (1, name)
            return (2, name)

        for hyp in sorted(self.local_lexical_hyps(tgt), key=_lexical_hyp_apply_order):
            cur = self.state.get_current_tuple()
            tree = cur.get_tree()
            results: list[tuple[Any, Tree]] = []
            try:
                if hyp.backtracks_on_success():
                    raw = hyp.exec_exhaustively(tree.clone(), cur)  # type: ignore[attr-defined]
                    if raw:
                        results.extend(raw)
                else:
                    out = hyp.exec_tuple_context(tree.clone(), cur)
                    if out is not None:
                        results.append((hyp.instantiate(), out))
            except Exception:  # noqa: BLE001
                continue
            for inst, res_tree in results:
                nm = inst.get_name() if hasattr(inst, "get_name") else str(inst)
                if nm.startswith(self.HYP_ADJUNCTION_PREFIX):
                    if nm.startswith(self.HYP_ADJ_T_PREFIX) and self.hyp_adj_t is not None:
                        self.state.add_child(res_tree, self.hyp_adj_t, None)
                    else:
                        self.state.add_child(res_tree, inst, None)
                    continue
                if nm.startswith(self.HYP_SEM_PREFIX):
                    stack = self.state.word_stack
                    if not stack:
                        logger.debug("word stack empty; skip hyp-sem %s", inst)
                        continue
                    self.state.add_child(res_tree, inst, stack[-1])
                    continue
                if not self._result_tree_subsumes_target(res_tree, cur):
                    continue
                self.state.add_child(res_tree, inst, None)

    def _local_lexical_hyps_at(self, target: Tree, fixed_on_target: Any) -> set[LexicalHypothesis]:
        """Generate hyp-sem/copy hyps at *fixed_on_target* with freshened Fo puts (Java two-arg ``localLexicalHyps``)."""
        cur_tuple = self.state.get_current_tuple()
        work_target = target
        if fixed_on_target not in work_target:
            # Java TTRHypothesiser.localLexicalHyps(Tree, NodeAddress) returns {} when the
            # fixed address is not yet on the target tree (no parse-tree merge).
            return set()
        tgt_tree = cur_tuple.get_target_tree()
        nh_tree = cur_tuple.get_non_head_target()
        saved_tgt_ptr = tgt_tree.get_pointer()
        saved_nh_ptr = nh_tree.get_pointer()
        tgt_tree.set_pointer(fixed_on_target)
        nh_tree.set_pointer(fixed_on_target)

        parse_tree = cur_tuple.get_tree()
        pointer = parse_tree.get_pointer()
        node = parse_tree.node_at(pointer)

        def _restore_target_pointers() -> None:
            tgt_tree.set_pointer(saved_tgt_ptr)
            nh_tree.set_pointer(saved_nh_ptr)

        target_node = work_target[fixed_on_target]
        result: set[LexicalHypothesis] = set(self.target_independent_hyps)
        stack = self.state.word_stack
        if stack:
            peek = stack[-1].word
            if self.seed_lexicon.contains_key(peek) if hasattr(self.seed_lexicon, "contains_key") else peek in self.seed_lexicon:
                _restore_target_pointers()
                return result
        if node is None:
            _restore_target_pointers()
            return result
        if node.has_type() or not self.is_terminal_in(parse_tree, pointer):
            _restore_target_pointers()
            return result
        if not self.is_terminal_in(work_target, fixed_on_target):
            _restore_target_pointers()
            return result
        if self.copy_hyp is not None:
            result.add(self.copy_hyp)
        requirement = node.get_type_requirement() if hasattr(node, "get_type_requirement") else None
        put_list: list[Effect] = []
        manifest = False
        f: Any = None
        for lab in target_node.labels:
            if lab in node.labels or node.contains(lab):
                continue
            if isinstance(lab, FormulaLabel):
                f = lab.get_formula()
                if not isinstance(f, TTRFormula):
                    continue
                manifest = bool(getattr(f, "has_manifest_content", lambda: False)())
                put_list.append(TTRFreshPut(f.clone() if hasattr(f, "clone") else f))
            elif isinstance(lab, TypeLabel):
                from dylan.tree.label.labels import MetaLabel

                inner = getattr(requirement, "inner", None) if isinstance(requirement, Requirement) else None
                syn_type = None
                if isinstance(inner, TypeLabel):
                    syn_type = inner.type
                elif isinstance(inner, MetaLabel):
                    syn_type = inner.type
                if syn_type is not None and lab.type != syn_type:
                    put_list.append(Put(TypeLabel(syn_type)))
                else:
                    put_list.append(EffectFactory.create(f"{Put.FUNCTOR}({lab})"))
            else:
                put_list.append(EffectFactory.create(f"{Put.FUNCTOR}({lab})"))
        req = node.get_required_type() or node.get_type()
        req_str = str(req) if req is not None else ""
        cn_e_req = req_str == "cn>e" or (
            isinstance(requirement, Requirement)
            and str(getattr(getattr(requirement, "inner", None), "type", "")) == "cn>e"
        )
        if not put_list and cn_e_req:
            from dylan.formula.formula import Formula

            cn_e_fo = Formula.create(
                "R1^[r0 : R1|x1==epsilon(r0.head, r0) : e|head==x1 : e]",
            )
            if isinstance(cn_e_fo, TTRFormula):
                f = cn_e_fo
                manifest = bool(getattr(f, "has_manifest_content", lambda: False)())
                put_list.append(TTRFreshPut(f.clone() if hasattr(f, "clone") else f))
                if isinstance(requirement, Requirement):
                    inner = getattr(requirement, "inner", None)
                    if isinstance(inner, TypeLabel):
                        put_list.insert(0, Put(TypeLabel(inner.type)))
        if not put_list and req_str == "cn":
            gold = cur_tuple.get_gold_target_type() if hasattr(cur_tuple, "get_gold_target_type") else None
            if gold is not None:
                restrictor = self._cn_restrictor_formula_from_gold(gold)
                if restrictor is not None:
                    f = restrictor
                    manifest = bool(getattr(restrictor, "has_manifest_content", lambda: False)())
                    put_list.append(TTRFreshPut(restrictor.clone() if hasattr(restrictor, "clone") else restrictor))
                    if isinstance(requirement, Requirement):
                        inner = getattr(requirement, "inner", None)
                        if isinstance(inner, TypeLabel):
                            put_list.insert(0, Put(TypeLabel(inner.type)))
        if put_list and isinstance(requirement, Requirement):
            inner = getattr(requirement, "inner", None)
            inner_ty = str(getattr(inner, "type", "")) if isinstance(inner, TypeLabel) else ""
            req_ty = str(node.get_required_type() or node.get_type() or "")
            if (inner_ty in ("e>t", "es") or req_ty in ("e>t", "es")) and f is not None and "head==e0" not in str(f):
                gold = cur_tuple.get_gold_target_type() if hasattr(cur_tuple, "get_gold_target_type") else None
                if gold is not None:
                    aug = self._event_hyp_sem_formula_from_gold(gold)
                    if aug is not None:
                        f = aug
                        manifest = True
                        put_list = [TTRFreshPut(aug.clone() if hasattr(aug, "clone") else aug)]
                        if isinstance(inner, TypeLabel):
                            put_list.insert(0, Put(TypeLabel(inner.type)))
        if put_list and isinstance(requirement, Requirement):
            inner = getattr(requirement, "inner", None)
            if isinstance(inner, TypeLabel) and inner.type in ("e>t", "es"):
                has_ty_put = any(
                    isinstance(p, Put)
                    and isinstance(getattr(p, "label", None), TypeLabel)
                    and getattr(p.label, "type", None) == inner.type
                    for p in put_list
                )
                if not has_ty_put:
                    put_list.insert(0, Put(TypeLabel(inner.type)))
        if put_list:
            key = self.HYP_SEM_PREFIX + "(" + str(f if f is not None else "?") + ")"
            if isinstance(requirement, Requirement):
                result.add(LexicalHypothesis(key, requirement, put_list, manifest))
            else:
                result.add(LexicalHypothesis(key, put_list, manifest))
        _restore_target_pointers()
        return result

    def _event_hyp_sem_formula_from_gold(self, gold: TTRRecordType) -> TTRFormula | None:
        """Build Java-style ``R1^(R1 ++ [es|head|obj])`` for opened-event ``hyp-sem`` from corpus gold."""
        from dylan.formula.formula import Formula
        from dylan.formula.ttr_label import TTRLabel

        e0 = gold.get_field(TTRLabel("e0"))
        head = gold.get_field(TTRLabel("head"))
        obj = None
        for field in gold.get_fields():
            mt = str(field.manifest_type) if field.manifest_type is not None else ""
            if "obj(" in mt and field.ds_type is not None and str(field.ds_type) == "t":
                obj = field
                break
        if e0 is None or head is None or obj is None:
            return None
        es_part = str(e0)
        head_part = str(head)
        obj_part = f"{obj.label}==obj(e0, R1.head) : t"
        fo = Formula.create(f"R1^(R1 ++ [{es_part}|{head_part}|{obj_part}])")
        return fo if isinstance(fo, TTRFormula) else None

    def _cn_restrictor_formula_from_gold(self, gold: TTRRecordType) -> TTRFormula | None:
        """Return the embedded CN restrictor entity record from corpus gold (door ``hyp-sem`` Fo)."""
        for field in gold.get_fields():
            inner = field.get_type()
            if field.ds_type is not None or not isinstance(inner, TTRRecordType):
                continue
            label = str(getattr(field, "label", ""))
            if label.startswith("r") and inner.get_head_field() is not None:
                return inner.clone() if hasattr(inner, "clone") else inner
        return None

    def _merge_hyp_sem_into_target(self, result_tree: Tree, fixed_on_target: Any) -> None:
        """After a successful ``hyp-sem``, copy Fo labels from the parse tree into the target (Java parity)."""
        cur = self.state.get_current_tuple()
        tgt = cur.get_target_tree()
        if fixed_on_target not in result_tree or fixed_on_target not in tgt:
            return
        from dylan.tree.label.labels import FormulaLabel

        src = result_tree[fixed_on_target]
        dst = tgt[fixed_on_target]
        for lab in src.labels:
            if isinstance(lab, FormulaLabel) and lab not in dst.labels and not dst.contains(lab):
                dst.add_label(lab)

    def attempt_backtrack(self) -> bool:
        """TTR lattice + word-index backtrack mirroring Java ``attemptBacktrack``."""
        lg = logging.getLogger(__name__)
        lg.info("backtracking...")
        while not self.state.more_unseen_edges():
            if self.state.at_root():
                lg.debug("cannot backtrack from root")
                return False
            back_along = self.state.get_prev_action()
            back_over = self.state.go_up_once()
            if back_over is None:
                return False
            lg.info("Gone back over action: %s", back_along)
            if isinstance(back_along, TreeHypothesis) and self.lattice is not None:
                try:
                    self.lattice.backtrack(back_along.increments)
                except Exception as exc:  # noqa: BLE001
                    lg.debug("lattice.backtrack raised %s", exc)
            elif isinstance(back_along, LexicalHypothesis) and back_along.get_name().startswith(
                self.HYP_SEM_PREFIX,
            ):
                if back_along.has_semantic_content:
                    if back_over.word is not None:
                        self.state.word_stack.append(back_over.word)
                    self.word_index -= 1
            elif isinstance(back_along, LexicalAction):
                if back_over.word is not None:
                    self.state.word_stack.append(back_over.word)
                self.word_index -= 1
            self.state.mark_edge_as_seen_and_below_it_unseen(back_over)
        lg.info("Backtrack succeeded")
        return True

    def hypothesise(self) -> list[CandidateSequence]:
        """Drive :meth:`hypothesise_once` until exhausted; no Java-style fallback sequences."""
        try:
            while self.hypothesise_once():
                pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("hypothesise loop aborted: %s", exc)
            raise
        return list(self.hypotheses)

    # ---------------- main hypothesise loop (override) ----------------

    def hypothesise_once(self) -> bool:
        """Single iteration of the TTR-specific hypothesise loop (Java ``hypothesiseOnce``)."""
        if self.target_type is None:
            return False
        cur = self.state.get_current_tuple()
        prev_action_name = ""
        if not self.state.at_root():
            prev = self.state.get_prev_action()
            prev_action_name = prev.get_name() if prev is not None and hasattr(prev, "get_name") else ""
        try:
            max_sem = self._extraction_semantics(cur)
        except Exception:  # noqa: BLE001
            max_sem = TTRRecordType()
        done_with_branch = False
        if (
            max_sem is not None
            and max_sem.subsumes(self.target_type) if hasattr(max_sem, "subsumes") else False
        ) and self.target_type.subsumes(max_sem):
            if self.state.word_stack:
                logger.warning("word stack is non-empty: %s", self.state.word_stack)
            elif prev_action_name == "thinning" and str(cur.get_tree().get_pointer()) == "01":
                result = self.extract_sequence()
                from dylan.induction.em_learner.hypothesise_parity_util import normalise_sequence_line

                sig = normalise_sequence_line(result.to_short_string())
                known = {normalise_sequence_line(h.to_short_string()) for h in self.hypotheses}
                if sig not in known:
                    self.hypotheses.append(result)
                    self.attempt_backtrack()
                if len(self.hypotheses) > 300:
                    logger.warning("sequences exceeded 300; stopping")
                    return False
                done_with_branch = True
        elif cur.get_tree().is_complete():
            logger.warning("got to complete tree, but no two-way subsumption: %s", cur.get_tree())
        if not self.state.at_root() and not prev_action_name.startswith(self.HYP_ADJUNCTION_PREFIX) and not done_with_branch:
            stack = self.state.word_stack
            top_known = bool(stack) and (
                self.seed_lexicon.contains_key(stack[-1].word)
                if hasattr(self.seed_lexicon, "contains_key")
                else stack[-1].word in self.seed_lexicon
            )
            if top_known:
                self.apply_known_lexical()
                if self.word_index > 0:
                    prev_word = self.all_words[self.word_index - 1].word()
                    prev_known = (
                        self.seed_lexicon.contains_key(prev_word)
                        if hasattr(self.seed_lexicon, "contains_key")
                        else prev_word in self.seed_lexicon
                    )
                    if not prev_known:
                        target_tree = cur.get_target_tree() if hasattr(cur, "get_target_tree") else self.target_type
                        self.apply_lexical_hypotheses(target_tree)
                target_tree = cur.get_target_tree() if hasattr(cur, "get_target_tree") else self.target_type
                self.apply_optional_grammar(target_tree)
            else:
                target_tree = cur.get_target_tree() if hasattr(cur, "get_target_tree") else self.target_type
                self.apply_lexical_hypotheses(target_tree)
                self.apply_optional_grammar(target_tree)
        # do/while DAG search
        while True:
            traversed = self.state.go_first()
            if traversed is not None:
                first_action = traversed.actions[0] if getattr(traversed, "actions", None) else None
                if isinstance(first_action, TreeHypothesis):
                    if self.lattice is not None and not self.lattice.go(first_action.increments):
                        continue
                elif first_action is not None and getattr(first_action, "get_name", lambda: "")().startswith(
                    self.HYP_ADJUNCTION_PREFIX,
                ):
                    prev_tup = self.state.get_parent(self.state.get_current_tuple())
                    pointed = prev_tup.get_tree().get_pointed_node() if prev_tup is not None else None
                    node_type = (pointed.get_type() if pointed is not None else None) or (
                        pointed.get_required_type() if pointed is not None else None
                    )
                    if node_type == DSType.cn:
                        self.apply_tree_hypotheses_cn()
                    elif node_type == DSType.e:
                        self.apply_tree_hypotheses_entity()
                    else:
                        self.apply_tree_hypotheses_event()
                    return True
                elif first_action is not None and getattr(first_action, "get_name", lambda: "")().startswith(
                    self.HYP_SEM_PREFIX,
                ):
                    self.word_index += 1
                elif isinstance(first_action, LexicalAction):
                    self.word_index += 1
                tup = self.state.get_current_tuple()
                target_tree = tup.get_target_tree() if hasattr(tup, "get_target_tree") else Tree()
                self.apply_non_optional_grammar(target_tree)
                return True
            if not self.attempt_backtrack():
                break
        logger.info("DAG Exhausted")
        return False

    # ---------------- helpers ----------------

    @staticmethod
    def flatten(increments: Iterable[TypeLatticeIncrement]) -> TTRRecordType:
        """Combine *increments* via asymmetric merge (Java ``flatten``)."""
        result: TTRRecordType = TTRRecordType()
        for inc in increments:
            if not getattr(inc, "is_positive", lambda: True)():
                continue
            inc_rt = inc.get_increment() if hasattr(inc, "get_increment") else inc
            if inc_rt is None:
                continue
            if hasattr(inc_rt, "asymmetric_merge"):
                merged = inc_rt.asymmetric_merge(result)
                if isinstance(merged, TTRRecordType):
                    result = merged
                    continue
            for field in inc_rt.get_fields() if hasattr(inc_rt, "get_fields") else []:
                result.add_field(field.clone() if hasattr(field, "clone") else field)
        return result


TTRHypothesiser.loadTrainingExample = TTRHypothesiser.load_training_example  # type: ignore[attr-defined]
TTRHypothesiser.applyTreeHypothesesEntity = TTRHypothesiser.apply_tree_hypotheses_entity  # type: ignore[attr-defined]
TTRHypothesiser.applyTreeHypothesesEvent = TTRHypothesiser.apply_tree_hypotheses_event  # type: ignore[attr-defined]
TTRHypothesiser.applyTreeHypothesesCN = TTRHypothesiser.apply_tree_hypotheses_cn  # type: ignore[attr-defined]
TTRHypothesiser.extractSequence = TTRHypothesiser.extract_sequence  # type: ignore[attr-defined]
TTRHypothesiser.applyKnownLexical = TTRHypothesiser.apply_known_lexical  # type: ignore[attr-defined]
TTRHypothesiser.applyNonOptionalGrammar = TTRHypothesiser.apply_non_optional_grammar  # type: ignore[attr-defined]
TTRHypothesiser.applyOptionalGrammar = TTRHypothesiser.apply_optional_grammar  # type: ignore[attr-defined]
TTRHypothesiser.applyLexicalHypotheses = TTRHypothesiser.apply_lexical_hypotheses  # type: ignore[attr-defined]
TTRHypothesiser.localLexicalHyps = TTRHypothesiser.local_lexical_hyps  # type: ignore[attr-defined]
TTRHypothesiser.attemptBacktrack = TTRHypothesiser.attempt_backtrack  # type: ignore[attr-defined]
TTRHypothesiser.hypothesiseOnce = TTRHypothesiser.hypothesise_once  # type: ignore[attr-defined]
