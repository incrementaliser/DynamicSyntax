"""Interactive word-level DAG parser (Java `InteractiveContextParser`)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from loguru import logger as loguru_logger

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
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.nlp.types import DEFAULT_SPEAKER, Dialogue, WAIT_TOKEN, RELEASE_TURN_TOKEN, Utterance
from dylan.parser.dag_parser import DAGParser, _MAX_NONOPTIONAL_ADJUST_PASSES
from dylan.tree.label.labels import Requirement, TypeLabel
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree
from dylan.logging_config import ensure_library_loguru_stderr, sync_dylan_stdlib_level_for_icp
from dylan.logging_context import icp_parser_formula_log_context

if TYPE_CHECKING:
    pass


LogLevel = Literal["off", "error", "warning"]
LogOutput = Literal["terminal", "terminal_and_file", "file"]


def _repoint_for_verb_lexical(tree: Tree, la: LexicalAction) -> None:
    """Move pointer to ``01`` when applying finite/inf verb templates (``v_*``).

    Noun work often leaves the pointer inside an NP; English ``v_tran_fin`` IF clauses
    are stated relative to the predicate functor node ``01``.
    """
    at = la.get_lexical_action_type() or ""
    if not at.startswith("v_"):
        return
    subj = tree.node_at(NodeAddress("00"))
    if subj is None or subj.get_type() != TypeLabel.e.type:
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
MAX_REPAIR_DEPTH = 1

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
    max_repair_depth = MAX_REPAIR_DEPTH

    def __init__(
        self,
        resource_dir: str | Path | None = None,
        *,
        repairing: bool = False,
        top_n: int | tuple[str, ...] = 3,
        participants: tuple[str, ...] = (DEFAULT_NAME,),
        log_level: LogLevel = "off",
        log_output: LogOutput = "terminal",
        log_dir: Path | None = None,
    ) -> None:
        """Construct a parser with optional *resource_dir* (filesystem directory or bundled grammar id).

        Parser-specific logs use *log_level*, *log_output*, and optional *log_dir* (see :meth:`_configure_icp_sinks`).
        ``log_level`` also adjusts stdlib ``logging`` for the ``dylan`` package (lexicon load, DAG trace).
        """
        ensure_library_loguru_stderr()
        participants_resolved = participants if participants else (DEFAULT_NAME,)
        if not isinstance(top_n, int):
            participants_resolved = top_n  # type: ignore[assignment]
            top_n = 3
        self._top_n = top_n
        self._participants = participants_resolved
        self._default_repairing = repairing
        self._init_icp_log_settings(log_level, log_output, log_dir)

        if resource_dir is None:
            self._init_shell_unloaded()
        else:
            p = Path(resource_dir) if isinstance(resource_dir, str) else resource_dir
            if p.is_dir():
                self._apply_resource_dir(p, repairing=repairing)
            else:
                self._init_shell_unloaded()
                self.set_grammar(resource_dir, repairing=repairing)
        self._configure_icp_sinks()

    @classmethod
    def from_resource_dir(
        cls,
        resource_dir: str | Path,
        *,
        top_n: int = 3,
        **kwargs: object,
    ) -> InteractiveContextParser:
        """Load grammars from *resource_dir* (convenience over ``cls(resource_dir, ...)``)."""
        return cls(resource_dir, top_n=top_n, **kwargs)  # type: ignore[arg-type]

    def _init_icp_log_settings(
        self,
        log_level: LogLevel,
        log_output: LogOutput,
        log_dir: Path | None,
    ) -> None:
        """Assign per-parser loguru identity, levels, and output destinations (sinks added in :meth:`_configure_icp_sinks`)."""
        self._icp_id = str(uuid4())
        self._log_level = log_level
        self._log_output = log_output
        self._log_dir = log_dir
        self._icp_log_handler_ids: list[int] = []
        self._log = loguru_logger.bind(icp_id=self._icp_id)
        sync_dylan_stdlib_level_for_icp(log_level)

    def _remove_icp_log_sinks(self) -> None:
        """Remove loguru sinks registered for this parser's ``icp_id``."""
        for hid in self._icp_log_handler_ids:
            try:
                loguru_logger.remove(hid)
            except ValueError:
                pass
        self._icp_log_handler_ids.clear()

    def close(self) -> None:
        """Remove per-parser loguru sinks.

        Call when a parser instance is no longer needed. Do not rely on ``__del__`` for cleanup:
        removing loguru handlers during interpreter shutdown can deadlock on loguru's internal lock
        when several parsers are collected together (e.g. after pytest).
        """
        self._remove_icp_log_sinks()

    def _configure_icp_sinks(self) -> None:
        """Register loguru sinks filtered to this parser's ``icp_id`` (or remove prior registrations)."""
        self._remove_icp_log_sinks()
        if self._log_level == "off":
            return
        lu_level = "ERROR" if self._log_level == "error" else "WARNING"

        def icp_only(record: dict) -> bool:
            return record["extra"].get("icp_id") == self._icp_id

        fmt_stderr = "<level>{level}</level> | {message}\n"
        out = self._log_output
        if out in ("terminal", "terminal_and_file"):
            hid = loguru_logger.add(
                sys.stderr,
                level=lu_level,
                filter=icp_only,
                format=fmt_stderr,
            )
            self._icp_log_handler_ids.append(hid)
        if out in ("terminal_and_file", "file"):
            base = self._log_dir if self._log_dir is not None else Path.cwd() / "logs"
            base.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = base / f"icp_{ts}_{self._icp_id[:8]}.log"
            hid = loguru_logger.add(
                str(path),
                level=lu_level,
                filter=icp_only,
                format="{time} | <level>{level}</level> | {message}\n",
                rotation="10 MB",
            )
            self._icp_log_handler_ids.append(hid)

    def _log_nonoptional_adjust_limit_exceeded(self) -> None:
        """Emit error when the non-optional adjustment loop exceeds its pass bound."""
        self._log.error(
            "adjust_with_non_optional_grammar exceeded {} passes — possible runaway rule loop",
            _MAX_NONOPTIONAL_ADJUST_PASSES,
        )

    def _init_shell_unloaded(self) -> None:
        """Initialise placeholder lexicon/grammar with no dialogue ``context`` until :meth:`set_grammar`."""
        DAGParser.__init__(
            self,
            Lexicon(None, self._top_n),
            Grammar(None),
            SpeechActInferenceGrammar(Path(".")),
        )
        self.context = None
        self.forced_restart = False
        self.forced_repair = False
        self.right_edge_indicators = []
        self.acks = ["uhu"]
        self.repairanda = ["uhh", "errm", "err", "er", "uh", "erm", "uhm", "um", "oh"]
        self.forced_repairanda = ["sorry", "oops", "wait", "erm"]
        self.restarters = ["yeah"]

    def _apply_resource_dir(self, path: Path, *, repairing: bool) -> None:
        """Load lexicon and grammars from *path* and wire a fresh :class:`Context`."""
        parts = self._participants if self._participants else (DEFAULT_NAME,)
        DAGParser.__init__(
            self,
            Lexicon(path, self._top_n),
            Grammar(path),
            SpeechActInferenceGrammar(path),
        )
        dag = WordLevelContextDAG()
        self.context = Context(dag, self.sa_grammar, *parts)
        self.context.set_repair_processing(repairing)
        self.forced_restart = False
        self.forced_repair = False
        self.right_edge_indicators = []
        self.acks = ["uhu"]
        self.repairanda = ["uhh", "errm", "err", "er", "uh", "erm", "uhm", "um", "oh"]
        self.forced_repairanda = ["sorry", "oops", "wait", "erm"]
        self.restarters = ["yeah"]
        self.init()

    def set_grammar(self, grammar: str | Path, *, repairing: bool | None = None) -> None:
        """Load grammar files from a directory path or bundled grammar id/alias into this parser."""
        from dynamicsyntax._session import resolved_grammar_path

        rep = self._default_repairing if repairing is None else repairing
        with resolved_grammar_path(grammar) as path:
            self._apply_resource_dir(path, repairing=rep)

    def parse(
        self,
        sentence_or_goal: str | Formula | None = None,
        /,
        *,
        speaker: str = DEFAULT_SPEAKER,
        trace: bool = False,
    ) -> object:
        """Parse a surface string into a :class:`~dynamicsyntax.parse_result.ParseResult`, or run ``parse_goal``.

        Pass a ``str`` for high-level sentence parsing (requires :meth:`set_grammar` first unless
        constructed with a grammar). Pass ``None`` or a :class:`~dylan.formula.formula.Formula`
        for the internal ``parse_goal`` path (Java ``DAGParser.parse``).
        """
        if isinstance(sentence_or_goal, str):
            return self._parse_surface(sentence_or_goal, speaker=speaker, trace=trace)
        return self.parse_goal(sentence_or_goal)

    def _parse_surface(self, sentence: str, *, speaker: str, trace: bool) -> object:
        """Run :func:`~dynamicsyntax._parse._run_parse_core` for a non-goal surface string."""
        if self.context is None:
            raise ValueError("grammar not loaded; call set_grammar(...) first")
        from dynamicsyntax._parse import _run_parse_core
        from dynamicsyntax.parse_result import ParseResult

        stripped = sentence.strip()
        if not stripped:
            return ParseResult(ok=False, semantics=None, tree=None, sentence="", parser=self)
        return _run_parse_core(self, stripped, speaker=speaker, trace=trace)

    @classmethod
    def from_loaded(
        cls,
        lexicon: Lexicon,
        grammar: Grammar,
        *,
        sa: SpeechActInferenceGrammar | None = None,
        participants: tuple[str, ...] = (DEFAULT_NAME,),
        log_level: LogLevel = "off",
        log_output: LogOutput = "terminal",
        log_dir: Path | None = None,
    ) -> InteractiveContextParser:
        """Build parser from in-memory `Lexicon` / `Grammar` (Java `Lexicon, Grammar` ctor).

        Parser-bound logging uses *log_level*, *log_output*, and *log_dir* like :meth:`__init__`.
        """
        ensure_library_loguru_stderr()
        obj = cls.__new__(cls)
        DAGParser.__init__(obj, lexicon, grammar, sa or SpeechActInferenceGrammar(Path(".")))
        dag = WordLevelContextDAG()
        parts = participants if participants else (DEFAULT_NAME,)
        obj.context = Context(dag, obj.sa_grammar, *parts)
        obj.context.set_repair_processing(False)
        obj._top_n = lexicon.top_n
        obj._participants = parts
        obj._default_repairing = False
        obj.forced_restart = False
        obj.forced_repair = False
        obj.right_edge_indicators = []
        obj.acks = ["uhu"]
        obj.repairanda = ["uhh", "errm", "err", "er", "uh", "erm", "uhm", "um", "oh"]
        obj.forced_repairanda = ["sorry", "oops", "wait", "erm"]
        obj.restarters = ["yeah"]
        obj._init_icp_log_settings(log_level, log_output, log_dir)
        obj._configure_icp_sinks()
        return obj

    def get_name(self) -> str:
        """Return this parser's context name."""
        if self.context is None:
            return DEFAULT_NAME
        return self.context.get_name()

    def repair_initiated(self) -> bool:
        """Return whether local repair has been initiated."""
        if self.context is None:
            return False
        return self.context.repair_initiated()

    def _adjust_once(self, goal: Formula | None) -> bool:
        dag = self.get_state()
        if self.context.repair_initiated():
            self._log.info("Repair initiated")
            repair_word = dag.word_stack_ref().pop()
            if not dag.word_stack:
                return False
            target = dag.word_stack[-1]
            if self.forced_restart:
                self.restart(target)
            else:
                self.backtrack_and_parse(target)
            if repair_word.word != word_level_repair_marker():
                dag.word_stack_ref().append(repair_word)
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
        """Parse until the DAG word stack is empty, optionally enforcing *goal*."""
        with icp_parser_formula_log_context(self._log_level):
            dag = self.get_state()
            if dag.is_exhausted():
                self._log.debug("state exhausted")
                return False
            while True:
                if not self._adjust_once(goal):
                    self._log.debug("wordstack: {}", dag.word_stack_ref())
                    self._log.debug("depth: {}", dag.get_depth())
                    dag.set_exhausted(True)
                    return False
                if not dag.word_stack:
                    break
            return True

    def complete_tree(self, res: Tree) -> tuple[list[Action], Tree]:
        """Run ``complete_once`` until quiescence or ``Tree.is_complete`` (Java ``DAGParser.complete``)."""
        with icp_parser_formula_log_context(self._log_level):
            return super().complete_tree(res)

    def get_final_semantics(self) -> TTRRecordType:
        """Semantics at the current tuple after ``evaluate`` (Java ``getFinalSemantics``)."""
        with icp_parser_formula_log_context(self._log_level):
            return super().get_final_semantics()

    def _apply_all_permutations(self, goal: Formula | None) -> None:
        """Apply every compatible lexical/grammar permutation for the top stack word."""
        dag = self.get_state()
        if not dag.word_stack:
            return
        word = dag.word_stack[-1]
        if word.word in self.right_edge_indicators:
            self.replay_backtracked_actions(word)
            return
        if word.word in self.acks:
            completed = self.complete(word)
            parent_edge = dag.get_parent_edge(completed)
            if parent_edge is not None:
                parent_edge.ground_for(word.speaker)
            return

        all_actions = self.lexicon.lookup(word.word)
        left_adjust: list[LexicalAction] = []
        current_tree = dag.get_current_tuple().get_tree().clone()
        for la in all_actions:
            if la.requires_left_adjustment():
                left_adjust.append(la)
                continue
            self._log.debug("applying {} without left adjustment", la)
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
            self._add_permutation_child(dag.get_current_tuple(), tup, edge_acts, word, la)

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
                self._log.warning(
                    "Lexical optional-grammar expansion exceeded {} pairs — stopping (avoid hang)",
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
                if pair_acts:
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
                child = dag.get_new_tuple(res)
                self._add_permutation_child(dag.get_current_tuple(), child, new_acts, word, la)

    def _add_permutation_child(
        self,
        parent: DAGTuple,
        child: DAGTuple,
        actions: list[Action],
        word: UtteredWord,
        lexical_action: LexicalAction,
    ) -> None:
        """Add a word child, splitting TRP/completion actions when present."""
        dag = self.get_state()
        split_idx = self._index_of_trp(actions)
        repairable = (lexical_action.get_lexical_action_type() or "") not in NON_REPAIRING_ACTION_TYPES
        if split_idx is None:
            edge = dag.get_new_edge(actions, word)
            edge.set_repairable(repairable)
            dag.add_child_from(parent, child, edge)
            return
        completion_actions = actions[: split_idx + 1]
        word_actions = actions[split_idx + 1 :]
        middle_tree = parent.get_tree().clone()
        applied = self.apply_actions(middle_tree, completion_actions)
        if applied is None:
            applied = parent.get_tree().clone()
        middle = dag.get_new_tuple(applied)
        completion_edge = dag.get_new_completion_edge(completion_actions, None)
        completion_edge.set_repairable(False)
        dag.add_child_from(parent, middle, completion_edge)
        edge = dag.get_new_edge(word_actions or [lexical_action.instantiate()], word)
        edge.set_repairable(repairable)
        dag.add_child_from(middle, child, edge)

    def _index_of_trp(self, actions: list[Action]) -> int | None:
        """Return index of first completion/TRP action in *actions*."""
        completion_names = {name.lower() for name in self.completion_grammar.keys()}
        for i, action in enumerate(actions):
            name = action.get_name().lower()
            if name in completion_names or name in {"trp", "completion", "merge"}:
                return i
        return None

    def init(self) -> None:
        """Reset parser flags and context."""
        self.forced_restart = False
        self.forced_repair = False
        super().init()

    def init_participants(self, participants: list[str]) -> None:
        """Reset parser with a new participant list."""
        self.forced_restart = False
        self.forced_repair = False
        self.context.init_participants(participants)

    def new_sentence(self) -> None:
        """Start a new sentence by adding a fresh axiom."""
        self.get_state().add_axiom()

    def parse_word(self, w: UtteredWord) -> WordLevelContextDAG | None:
        """Parse one uttered word."""
        participants = list(self.context.get_participants())
        if len(participants) == 2 and w.speaker in participants:
            i = participants.index(w.speaker)
            other = participants[1 - i]
            word = UtteredWord(w.word, w.speaker, other)
        else:
            word = UtteredWord(w.word, w.speaker, w.addressee)
        self._log.info("Parsing word: {}", word)

        if word.word == self.WAIT:
            return self.get_state()

        if self.forced_restart or self.forced_repair:
            self._log.info("restart/repair path")
            self.get_state().word_stack_ref().append(word)
            self.get_state().initiate_local_repair()
            ok = self.parse_goal(None)
            self.forced_restart = False
            self.forced_repair = False
            if not ok:
                self.get_state().reset_to_first_tuple_after_last_word()
                return None
            self.get_state().this_is_first_tuple_after_last_word()
            return self.get_state()

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
            self._log.error("Word not in Lexicon: {}", word)
            return None

        self.get_state().word_stack_ref().append(word)
        if not self.parse_goal(None):
            self._log.error("Cannot parse: {} — resetting", word.word)
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
        self.context.append_word(word)
        self._log.info("Parsed: {}", word)
        return self.get_state()

    def parse_utterance(self, utt: Utterance) -> bool:
        """Parse each word in order (Java `DAGParser.parseUtterance`)."""
        ok = True
        for uw in utt.words:
            if self.parse_word(uw) is None:
                self._log.error("Failed to parse {}", uw)
                ok = False
        return ok

    def generate_word(self, word: UtteredWord | str, goal: Formula | None = None) -> WordLevelContextDAG | None:
        """Generate/parse one word under an optional semantic goal."""
        uw = word if isinstance(word, UtteredWord) else UtteredWord(word, self.get_name())
        self.get_state().word_stack_ref().append(uw)
        if not self.parse_goal(goal):
            self.get_state().reset_to_first_tuple_after_last_word()
            return None
        if uw.word == self.RELEASE_TURN:
            self.context.open_floor()
        else:
            self.context.set_who_has_floor(uw.speaker)
        self.context.append_word(uw)
        self.get_state().this_is_first_tuple_after_last_word()
        return self.get_state()

    def parse_dialogue(self, dialogue: Dialogue) -> Context[DAGTuple, GroundableEdge]:
        """Parse a dialogue utterance by utterance."""
        participants = dialogue.get_participants()
        if participants:
            self.context.init_participants(participants)
        for utterance in dialogue:
            self.parse_utterance(utterance)
        return self.context

    def get_top_n_pending(self, n: int) -> list[TTRRecordType]:
        """Return up to *n* best final semantics candidates."""
        return self.get_n_best_final_semantics(n)

    def derive_language(
        self,
        *,
        max_len: int,
        min_len: int = 1,
        max_candidates: int | None = None,
        max_successful: int | None = None,
        out_dir: str | Path | None = None,
        grammar_name: str | None = None,
        max_workers: int | None = None,
        speaker: str = DEFAULT_SPEAKER,
        addressee: str = "you",
    ) -> tuple[Path, Path]:
        """Derive bounded language files; delegates to :class:`~dylan.parser.language_derivation.LanguageDerivation`."""
        from dylan.parser.language_derivation import DEFAULT_LANGUAGE_OUTPUT_DIR, LanguageDerivation

        return LanguageDerivation(self).run(
            max_len=max_len,
            min_len=min_len,
            max_candidates=max_candidates,
            max_successful=max_successful,
            out_dir=out_dir if out_dir is not None else DEFAULT_LANGUAGE_OUTPUT_DIR,
            grammar_name=grammar_name,
            max_workers=max_workers,
            speaker=speaker,
            addressee=addressee,
        )

    def derive_language_layered(
        self,
        *,
        max_len: int,
        min_len: int = 1,
        max_successful: int | None = None,
        out_dir: str | Path | None = None,
        grammar_name: str | None = None,
        max_workers: int | None = None,
        speaker: str = DEFAULT_SPEAKER,
        addressee: str = "you",
    ) -> dict[int, tuple[Path, Path, Path]]:
        """Layered prefix derivation with per-depth ``layer_i`` output files (forward-only, ``top_n=1``).

        Uses multi-process BFS when ``max_workers`` is not ``1`` (default: one fewer than logical CPUs);
        pass ``max_workers=1`` for a single-process run.
        Each layer also writes ``*_fringe.txt`` (one feasible prefix per line after that layer's fringe is fixed).
        """
        from dylan.parser.language_derivation import DEFAULT_LANGUAGE_OUTPUT_DIR, LanguageDerivation

        return LanguageDerivation(self).run_layered(
            max_len=max_len,
            min_len=min_len,
            max_successful=max_successful,
            out_dir=out_dir if out_dir is not None else DEFAULT_LANGUAGE_OUTPUT_DIR,
            grammar_name=grammar_name,
            max_workers=max_workers,
            speaker=speaker,
            addressee=addressee,
        )

    def derive_language_layered_category(
        self,
        *,
        max_len: int,
        min_len: int = 1,
        max_successful: int | None = None,
        out_dir: str | Path | None = None,
        grammar_name: str | None = None,
        max_workers: int | None = None,
        speaker: str = DEFAULT_SPEAKER,
        addressee: str = "you",
    ) -> dict[int, tuple[Path, Path, Path]]:
        """Layered derivation using one representative word per lexical template.

        Parallelism and fringe output files match :meth:`derive_language_layered` (``max_workers``).
        """
        from dylan.parser.language_derivation import DEFAULT_LANGUAGE_OUTPUT_DIR, LanguageDerivation

        return LanguageDerivation(self).run_layered_category(
            max_len=max_len,
            min_len=min_len,
            max_successful=max_successful,
            out_dir=out_dir if out_dir is not None else DEFAULT_LANGUAGE_OUTPUT_DIR,
            grammar_name=grammar_name,
            max_workers=max_workers,
            speaker=speaker,
            addressee=addressee,
        )

    def derive_language_layered_random(
        self,
        *,
        max_len: int,
        max_paths: int,
        max_steps: int | None = None,
        seed: int | None = None,
        min_len: int = 1,
        max_successful: int | None = None,
        out_dir: str | Path | None = None,
        grammar_name: str | None = None,
        speaker: str = DEFAULT_SPEAKER,
        addressee: str = "you",
        use_category_vocab: bool = False,
    ) -> dict[int, tuple[Path, Path, Path]]:
        """Random-walk layered derivation with the same completion rules as :meth:`derive_language_layered`.

        Fringe paths are included in the returned map; those files stay empty for this mode.
        """
        from dylan.parser.language_derivation import DEFAULT_LANGUAGE_OUTPUT_DIR, LanguageDerivation

        return LanguageDerivation(self).run_layered_random(
            max_len=max_len,
            max_paths=max_paths,
            max_steps=max_steps,
            seed=seed,
            min_len=min_len,
            max_successful=max_successful,
            out_dir=out_dir if out_dir is not None else DEFAULT_LANGUAGE_OUTPUT_DIR,
            grammar_name=grammar_name,
            speaker=speaker,
            addressee=addressee,
            use_category_vocab=use_category_vocab,
        )

    def replay_backtracked_actions(self, word: UtteredWord) -> bool:
        """Replay the current edge action sequence at a right-edge indicator."""
        dag = self.get_state()
        parent_edge = dag.get_parent_edge()
        if parent_edge is None:
            return False
        tree = dag.get_current_tuple().get_tree().clone()
        res = self.apply_actions(tree, parent_edge.get_actions())
        if res is None:
            return False
        child = dag.get_new_tuple(res)
        edge = dag.get_new_action_replay_edge(parent_edge.get_actions(), word)
        edge.set_repairable(False)
        dag.add_child(child, edge)
        return True

    def restart(self, word: UtteredWord) -> None:
        """Restart from the last post-word anchor and parse *word* again."""
        dag = self.get_state()
        dag.reset_to_first_tuple_after_last_word()
        dag.word_stack_ref().append(word)
        self._apply_all_permutations(None)

    def backtrack_and_parse(self, word: UtteredWord) -> None:
        """Backtrack locally then parse *word* again."""
        dag = self.get_state()
        depth = 0
        while depth < self.max_repair_depth and dag.get_parent(dag.get_current_tuple()) is not None:
            edge = dag.get_parent_edge()
            if edge is None:
                break
            if edge.is_repairable() and not edge.is_grounded_for(word.speaker):
                edge.backtrack(dag)
                break
            edge.backtrack(dag)
            depth += 1
        dag.word_stack_ref().append(word)
        self._apply_all_permutations(None)

    def left_adjust_and_apply(self, lexical_action: LexicalAction) -> bool:
        """Probe whether a lexical action can apply after left adjustment."""
        dag = self.get_state()
        before = dag.get_current_tuple()
        fake_word = UtteredWord(lexical_action.word, self.get_name())
        self._add_permutation_child(before, before, [lexical_action.instantiate()], fake_word, lexical_action)
        return dag.out_degree(before) > 0

    def get_local_generation_options(self) -> set[str]:
        """Return lexicon words with at least one locally applicable action."""
        options: set[str] = set()
        tree = self.get_state().get_current_tuple().get_tree()
        for word, actions in self.lexicon.items():
            for action in actions:
                if action.exec(tree.clone(), self.context) is not None:
                    options.add(word)
                    break
        return options

    def roll_back(self, n: int) -> bool:
        """Roll parser context back by *n* word edges."""
        return self.get_state().roll_back(n)

    def get_dialogue_history(self) -> list[UtteredWord]:
        """Return parsed dialogue word history."""
        return self.context.get_dialogue_history()

    def is_exhausted(self) -> bool:
        """Return whether the DAG state is exhausted."""
        return self.get_state().is_exhausted()

    def get_best_tuple(self) -> DAGTuple:
        """Current DAG tuple (Java `getBestTuple`)."""
        return self.get_state().get_current_tuple()


def word_level_repair_marker() -> str:
    """Return the repair-init marker used by the word-level DAG."""
    from dylan.dag.word_level_context_dag import REPAIR_INIT_PREFIX

    return REPAIR_INIT_PREFIX


InteractiveContextParser.getName = InteractiveContextParser.get_name  # type: ignore[attr-defined]
InteractiveContextParser.repairInitiated = InteractiveContextParser.repair_initiated  # type: ignore[attr-defined]
InteractiveContextParser.adjustOnce = InteractiveContextParser._adjust_once  # type: ignore[attr-defined]
InteractiveContextParser.applyAllPermutations = InteractiveContextParser._apply_all_permutations  # type: ignore[attr-defined]
InteractiveContextParser.parseGoal = InteractiveContextParser.parse_goal  # type: ignore[attr-defined]
InteractiveContextParser.initParticipants = InteractiveContextParser.init_participants  # type: ignore[attr-defined]
InteractiveContextParser.newSentence = InteractiveContextParser.new_sentence  # type: ignore[attr-defined]
InteractiveContextParser.parseWord = InteractiveContextParser.parse_word  # type: ignore[attr-defined]
InteractiveContextParser.parseUtterance = InteractiveContextParser.parse_utterance  # type: ignore[attr-defined]
InteractiveContextParser.generateWord = InteractiveContextParser.generate_word  # type: ignore[attr-defined]
InteractiveContextParser.parseDialogue = InteractiveContextParser.parse_dialogue  # type: ignore[attr-defined]
InteractiveContextParser.getTopNPending = InteractiveContextParser.get_top_n_pending  # type: ignore[attr-defined]
InteractiveContextParser.deriveLanguage = InteractiveContextParser.derive_language  # type: ignore[attr-defined]
InteractiveContextParser.deriveLanguageLayered = InteractiveContextParser.derive_language_layered  # type: ignore[attr-defined]
InteractiveContextParser.deriveLanguageLayeredCategory = (  # type: ignore[attr-defined]
    InteractiveContextParser.derive_language_layered_category
)
InteractiveContextParser.deriveLanguageLayeredRandom = InteractiveContextParser.derive_language_layered_random  # type: ignore[attr-defined]
InteractiveContextParser.replayBacktrackedActions = InteractiveContextParser.replay_backtracked_actions  # type: ignore[attr-defined]
InteractiveContextParser.backtrackAndParse = InteractiveContextParser.backtrack_and_parse  # type: ignore[attr-defined]
InteractiveContextParser.leftAdjustAndApply = InteractiveContextParser.left_adjust_and_apply  # type: ignore[attr-defined]
InteractiveContextParser.getLocalGenerationOptions = InteractiveContextParser.get_local_generation_options  # type: ignore[attr-defined]
InteractiveContextParser.rollBack = InteractiveContextParser.roll_back  # type: ignore[attr-defined]
InteractiveContextParser.getDialogueHistory = InteractiveContextParser.get_dialogue_history  # type: ignore[attr-defined]
InteractiveContextParser.isExhausted = InteractiveContextParser.is_exhausted  # type: ignore[attr-defined]
InteractiveContextParser.getBestTuple = InteractiveContextParser.get_best_tuple  # type: ignore[attr-defined]
