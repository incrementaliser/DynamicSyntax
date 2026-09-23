"""Headless grammar load and parse session shared by the Flet GUI and the browser UI.

The static site calls into Python via :mod:`dylan.pyodide_api` (JSON); that module
delegates to :class:`ParseSession` here. There is no separate ``web_api`` under
``dylan.gui`` — only this session layer plus the Pyodide façade at package root.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from dylan.action.computational_action import ComputationalAction
from dylan.dag.groundable_edge import CompletionEdge
from dylan.gui.formatting import format_dag_overview, format_ds_tree, format_semantics_display
from dylan.gui.tree_viz import format_ds_tree_ascii
from dylan.nlp.types import DEFAULT_SPEAKER, utterance_from_text
from dylan.parser.interactive_context_parser import InteractiveContextParser
from dylan.tree.tree import Tree

logger = logging.getLogger(__name__)

GUI_INFO_HELP_TEXT = (
    "Load grammar by selecting the grammar folder (it must contain lexicon and action files). "
    "Type a sentence, then Parse. "
    "The arrows beside #interpretations move between readings of that sentence. "
    "Ctrl+Plus and Ctrl+Minus zoom the tree; Ctrl+0 returns to 100%. "
    "Fit scrolls so the middle is in the pane."
)

FLET_INFO_HELP_LINES: tuple[str, ...] = (
    "Load grammar by selecting the grammar folder (it must contain lexicon and action files).",
    "Type a sentence, then Parse.",
    "The arrows beside #interpretations move between readings of that sentence.",
    "Ctrl+Plus and Ctrl+Minus zoom the tree; Ctrl+0 returns to 100%.",
    "Fit scrolls so the middle is in the pane.",
    "Reset clears the derivation back to the empty axiom. The grammar stays loaded.",
    "Press Reset before parsing a sentence that does not continue the previous one.",
)
FLET_INFO_HELP_TEXT = "\n".join(FLET_INFO_HELP_LINES)

NO_NEW_WORDS_LOG = "No new words to parse."

INTERPRETATION_CAP: int = 30


def resolve_grammar_directory(path_str: str) -> Path:
    """Return the grammar folder: the directory itself, or the parent of a picked file."""
    p = Path(path_str.strip())
    if p.is_file():
        return p.parent
    return p


class GrammarLoadLogFilter(logging.Filter):
    """Keep INFO+ from all ``dylan`` loggers; DEBUG only from lexicon (skipped lines, etc.)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.INFO:
            return True
        return record.name.startswith("dylan.action.lexicon")


class DylanCaptureHandler(logging.Handler):
    """Collect log records under the ``dylan`` namespace while a grammar is loading."""

    def __init__(self, lines: list[str]) -> None:
        super().__init__(level=logging.DEBUG)
        self._lines = lines
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        self.addFilter(GrammarLoadLogFilter())

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._lines.append(self.format(record))
        except Exception:
            self.handleError(record)


def format_grammar_load_report(
    path: Path,
    captured: list[str],
    *,
    ok: bool,
    error: str | None = None,
) -> str:
    """Build an event-history grammar-load block (warnings kept; no success fluff)."""
    resolved = path.resolve() if path.exists() else path
    if ok:
        lines = [f"Grammar loaded: {resolved}"]
    else:
        lines = [f"Grammar load failed: {resolved}"]
        if error:
            lines.append(error)
    if captured:
        lines.append("")
        lines.extend(captured)
    warn_count = sum(1 for s in captured if s.startswith("WARNING "))
    if ok and warn_count:
        lines.append("")
        lines.append(
            f"Note: {warn_count} warning(s) above — review lexicon / templates; "
            "the grammar may be incomplete.",
        )
    return "\n".join(lines)


def format_event_log(message: str) -> str:
    """Return a short Logs-panel event line (no section banners)."""
    return message.strip()


def format_parse_state_log(msg: str) -> str:
    """Backward-compatible alias for :func:`format_event_log`."""
    return format_event_log(msg)


def format_interpretation_readout(index: int, count: int, *, capped: bool) -> str:
    """Bordered-box text ``#interpretations: 1 / N``, or ``#interpretations: 0``."""
    if count <= 0:
        return "#interpretations: 0"
    total = f"{count}+" if capped else str(count)
    return f"#interpretations: {index} / {total}"


def format_interpretation_log(index: int, count: int, *, capped: bool) -> str:
    """Logs line when the user moves to interpretation *index*."""
    total = f"{count}+" if capped else str(count)
    return f"Interpretation {index} / {total}"


def format_word_event(*, word: str, ok: bool, reason: str | None = None) -> str:
    """One Logs line for a single word of an incremental parse."""
    token = word.strip() or "—"
    if ok:
        return f"{token} parsed"
    if reason == "lexicon":
        return f"{token} failed (not in lexicon)"
    return f"{token} failed (no derivation)"


def format_parse_event(
    *,
    sentence: str,
    ok: bool,
    missing_words: list[str] | None = None,
) -> str:
    """One-line parse result for the Logs event history.

    On failure, name lexicon misses when possible; otherwise report that no derivation applied.
    """
    quoted = sentence.strip()
    if ok:
        return f"Parsed “{quoted}” — OK"
    missing = [w for w in (missing_words or []) if w]
    if missing:
        listed = ", ".join(missing)
        return f"Parsed “{quoted}” — failed ({listed} not in lexicon)"
    return f"Parsed “{quoted}” — failed (no derivation for this sentence)"


def format_session_info(
    *,
    grammar_path: str | None,
    repairing: bool,
    last_event: str,
    load_warning_count: int,
    pointer: str | None,
    tuple_id: int | None,
) -> str:
    """Live Info-card text: short how-to plus current session facts."""
    grammar = grammar_path if grammar_path else "(none)"
    ptr = pointer if pointer else "—"
    tup = f"#{tuple_id}" if tuple_id is not None else "—"
    return "\n".join(
        [
            GUI_INFO_HELP_TEXT,
            "",
            f"Grammar: {grammar}",
            f"Repair: {'on' if repairing else 'off'}",
            f"Last: {last_event}",
            f"Load warnings: {load_warning_count}",
            f"Pointer: {ptr}",
            f"Current DAG tuple: {tup}",
        ],
    )


def last_event_tone(last_event: str) -> str:
    """Return ``error``, ``ok``, or ``body`` for the Status Last row."""
    low = last_event.lower()
    if "failed" in low or "does not continue" in low or "load a grammar" in low:
        return "error"
    if " — ok" in low or "loaded:" in low or "axiom" in low:
        return "ok"
    return "body"


def format_noncontinuation(sentence: str) -> str:
    """Log line when *sentence* does not extend the words already in the derivation."""
    return f"Parse: “{sentence.strip()}” does not continue the current derivation."


def format_action_log_lines(sequences: list[tuple[str, ...]]) -> list[str]:
    """Format computational-action lines for one word across interpretations.

    *sequences* is in 1-based interpretation order. One sequence shared by every
    interpretation has no prefix. Distinct sequences get one line each; identical
    sequences share a line. Empty sequences produce no line.
    """
    grouped: dict[tuple[str, ...], list[int]] = {}
    order: list[tuple[str, ...]] = []
    for index, seq in enumerate(sequences, start=1):
        if seq not in grouped:
            grouped[seq] = []
            order.append(seq)
        grouped[seq].append(index)
    if len(order) == 1:
        only = order[0]
        if not only:
            return []
        return [" ".join(only)]
    lines: list[str] = []
    for seq in order:
        if not seq:
            continue
        indexes = ", ".join(str(number) for number in grouped[seq])
        lines.append(f"(interp {indexes}) {' '.join(seq)}")
    return lines


def _is_action_token(token: str) -> bool:
    """True if *token* looks like a computational-action name."""
    body = token
    while body.startswith("*") or body.startswith("+"):
        body = body[1:]
    if not body or not body[0].isalpha():
        return False
    return all(ch.isalnum() or ch in "_+-" for ch in body)


def is_action_log_line(line: str) -> bool:
    """True when *line* is a computational-action log line, with or without ``(interp N)``."""
    text = line.strip()
    if text.endswith(" parsed") or " failed" in text:
        return False
    body = text
    if text.startswith("(interp "):
        close = text.find(") ")
        if close < 0:
            return False
        head = text[len("(interp ") : close]
        if not head or any(not part.strip().isdigit() for part in head.split(",")):
            return False
        body = text[close + 2 :]
    if not body:
        return False
    return all(_is_action_token(tok) for tok in body.split())


def _word_actions_from_edges(edges: list[object]) -> list[tuple[str, tuple[str, ...]]]:
    """Group computational-action names by word along a root-to-cursor edge path.

    Completion edges count toward the following word. Names stay in path order.
    Lexical actions are omitted. Trailing completions attach to the previous word.
    """
    pending: list[str] = []
    rows: list[tuple[str, tuple[str, ...]]] = []
    for edge in edges:
        actions = getattr(edge, "actions", []) or []
        names = [action.get_name() for action in actions if isinstance(action, ComputationalAction)]
        if isinstance(edge, CompletionEdge):
            pending.extend(names)
            continue
        uttered = getattr(edge, "word", None)
        word = uttered.word if uttered is not None else None
        if not word:
            pending.extend(names)
            continue
        rows.append((str(word), tuple([*pending, *names])))
        pending = []
    if pending and rows:
        word, acts = rows[-1]
        rows[-1] = (word, acts + tuple(pending))
    return rows


@dataclass(frozen=True)
class StatusField:
    """One Status-card value and the tone used to colour it."""

    value: str
    tone: str


@dataclass(frozen=True)
class SessionStatus:
    """Live Status card: grammar, repair, last event, warnings, pointer, and DAG tuple."""

    grammar: StatusField
    repair: StatusField
    last: StatusField
    warnings: StatusField
    pointer: StatusField
    dag_tuple: StatusField


def format_session_status(
    *,
    grammar_path: str | None,
    repairing: bool,
    last_event: str,
    load_warning_count: int,
    pointer: str | None,
    tuple_id: int | None,
) -> SessionStatus:
    """Build the Status-card rows for the live session facts."""
    warning_tone = "amber" if load_warning_count > 0 else "muted"
    return SessionStatus(
        grammar=StatusField(grammar_path if grammar_path else "(none)", "muted"),
        repair=StatusField("on" if repairing else "off", "amber" if repairing else "muted"),
        last=StatusField(last_event, last_event_tone(last_event)),
        warnings=StatusField(str(load_warning_count), warning_tone),
        pointer=StatusField(pointer if pointer else "—", "mono"),
        dag_tuple=StatusField(f"#{tuple_id}" if tuple_id is not None else "—", "mono"),
    )


@dataclass(frozen=True)
class TreePanelState:
    """Strings for the parse-tree / address-order views (graph is drawn in the Flet canvas)."""

    address_order: str
    parse_tree_ascii: str


@dataclass(frozen=True)
class ViewStrings:
    """All tab panels plus tree text views."""

    address_order: str
    parse_tree_ascii: str
    semantics: str
    dag: str


class ParseSession:
    """Holds ``InteractiveContextParser`` state and produces view strings for UIs."""

    def __init__(self) -> None:
        self.parser: InteractiveContextParser | None = None
        self.last_tree: Tree | None = None
        self.grammar_path: Path | None = None
        self.repairing: bool = False
        self.load_warning_count: int = 0
        self.last_event: str = "No grammar loaded."
        self.interpretation_index: int = 0
        self.interpretation_count: int = 0
        self.interpretation_capped: bool = False
        self._consumed_words: list[str] = []
        self._interpretation_word_actions: list[list[tuple[str, tuple[str, ...]]]] = []

    def _clear_interpretations(self) -> None:
        """Forget the interpretation sequence (no parser, or no tree)."""
        self.interpretation_index = 0
        self.interpretation_count = 0
        self.interpretation_capped = False
        self._interpretation_word_actions = []

    def _clear_consumed_words(self) -> None:
        """Forget words that have entered the derivation."""
        self._consumed_words.clear()

    def _snapshot_word_actions(self) -> list[tuple[str, tuple[str, ...]]]:
        """Computational actions per word on the path to the current tuple.

        Returns an empty list when the parser DAG cannot be walked (test doubles).
        """
        if self.parser is None:
            return []
        dag = self.parser.get_state()
        get_parent_edge = getattr(dag, "get_parent_edge", None)
        get_parent = getattr(dag, "get_parent", None)
        get_current = getattr(dag, "get_current_tuple", None)
        if not callable(get_parent_edge) or not callable(get_parent) or not callable(get_current):
            return []
        cursor = get_current()
        edges: list[object] = []
        seen: set[int] = set()
        while cursor is not None and id(cursor) not in seen:
            seen.add(id(cursor))
            edge = get_parent_edge(cursor)
            if edge is None:
                break
            edges.append(edge)
            cursor = get_parent(cursor)
        edges.reverse()
        return _word_actions_from_edges(edges)

    def refresh_interpretations(self) -> None:
        """Count interpretations from the post-parse anchor, then show the first.

        The walk stops at :data:`INTERPRETATION_CAP`. One further successful step
        marks the count as capped (``30+``). The parser is left on interpretation 1.
        Each visited interpretation records the computational actions on its path.
        """
        if self.parser is None:
            self._clear_interpretations()
            return
        dag = self.parser.get_state()
        capped = False
        count = 1
        recorded: list[list[tuple[str, tuple[str, ...]]]] = []
        try:
            dag.reset_to_first_tuple_after_last_word()
            recorded.append(self._snapshot_word_actions())
            while count < INTERPRETATION_CAP:
                if not self.parser.parse_goal(None):
                    break
                count += 1
                recorded.append(self._snapshot_word_actions())
            else:
                if self.parser.parse_goal(None):
                    capped = True
        finally:
            dag.reset_to_first_tuple_after_last_word()
        self._interpretation_word_actions = recorded
        self.interpretation_index = 1
        self.interpretation_count = count
        self.interpretation_capped = capped

    def select_interpretation(self, index: int) -> tuple[str | None, str | None]:
        """Show the 1-based interpretation *index*.

        Returns ``(error, log_line)``. An index outside ``1 .. count``, or the
        index already shown, does nothing and returns ``(None, None)``.
        """
        if self.parser is None:
            self.last_event = "Interpretations: load a grammar first."
            return ("Interpretations: load a grammar first.", None)
        if (
            self.interpretation_count <= 0
            or index < 1
            or index > self.interpretation_count
            or index == self.interpretation_index
        ):
            return (None, None)
        dag = self.parser.get_state()
        dag.reset_to_first_tuple_after_last_word()
        reached = 1
        for _ in range(index - 1):
            if not self.parser.parse_goal(None):
                break
            reached += 1
        self.interpretation_index = reached
        log = format_interpretation_log(
            reached,
            self.interpretation_count,
            capped=self.interpretation_capped,
        )
        self.last_event = log
        return (None, log)

    def _pointer_and_tuple(self) -> tuple[str | None, int | None]:
        """Return the live pointer address and current DAG tuple id."""
        pointer: str | None = None
        tuple_id: int | None = None
        if self.last_tree is not None:
            pointer = str(self.last_tree.pointer.address)
        if self.parser is not None:
            current = self.parser.get_state().get_current_tuple()
            tuple_id = int(current.tuple_id)
            pointer = str(current.get_tree().pointer.address)
        return pointer, tuple_id

    def session_info_text(self) -> str:
        """Return the browser Info card: how-to plus current session facts."""
        pointer, tuple_id = self._pointer_and_tuple()
        grammar = str(self.grammar_path) if self.grammar_path is not None else None
        return format_session_info(
            grammar_path=grammar,
            repairing=self.repairing,
            last_event=self.last_event,
            load_warning_count=self.load_warning_count,
            pointer=pointer,
            tuple_id=tuple_id,
        )

    def session_status(self) -> SessionStatus:
        """Return the Flet Status card rows for the live session."""
        pointer, tuple_id = self._pointer_and_tuple()
        grammar = str(self.grammar_path) if self.grammar_path is not None else None
        return format_session_status(
            grammar_path=grammar,
            repairing=self.repairing,
            last_event=self.last_event,
            load_warning_count=self.load_warning_count,
            pointer=pointer,
            tuple_id=tuple_id,
        )

    def set_grammar(self, path_str: str, *, repairing: bool) -> str:
        """Load grammar from a filesystem directory *path_str*; return log text and set ``self.parser``.

        Bundled grammar nicknames (e.g. ``\"ttr\"``) are only accepted on a parser from
        :func:`dynamicsyntax.icp` via :meth:`InteractiveContextParser.set_grammar`;
        this GUI/session path expects a real directory from the file picker.
        """
        p = Path(path_str.strip())
        self.repairing = repairing
        self._clear_consumed_words()
        if not p.is_dir():
            self.parser = None
            self.last_tree = None
            self.grammar_path = None
            self.load_warning_count = 0
            self._clear_interpretations()
            self.last_event = f"Grammar load failed: {p}"
            return format_grammar_load_report(p, [], ok=False, error=f"Not a directory: {p}")
        captured: list[str] = []
        cap = DylanCaptureHandler(captured)
        dylan_log = logging.getLogger("dylan")
        saved_level = dylan_log.level
        dylan_log.setLevel(logging.DEBUG)
        dylan_log.addHandler(cap)
        try:
            try:
                self.parser = InteractiveContextParser(p, repairing=repairing)
                self.parser.init()
                self.refresh_interpretations()
                self.last_tree = self.parser.get_best_tuple().get_tree()
                self.grammar_path = p.resolve()
                self.load_warning_count = sum(1 for s in captured if s.startswith("WARNING "))
                self.last_event = f"Grammar loaded: {self.grammar_path}"
                return format_grammar_load_report(p, captured, ok=True)
            except OSError as ex:
                self.parser = None
                self.last_tree = None
                self.grammar_path = None
                self.load_warning_count = 0
                self._clear_interpretations()
                self.last_event = f"Grammar load failed: {p}"
                return format_grammar_load_report(p, captured, ok=False, error=str(ex))
            except Exception as ex:  # noqa: BLE001
                self.parser = None
                self.last_tree = None
                self.grammar_path = None
                self.load_warning_count = 0
                self._clear_interpretations()
                self.last_event = f"Grammar load failed: {type(ex).__name__}: {ex}"
                logger.exception("Grammar load failed")
                return format_grammar_load_report(
                    p,
                    captured,
                    ok=False,
                    error=f"{type(ex).__name__}: {ex}",
                )
        finally:
            dylan_log.removeHandler(cap)
            dylan_log.setLevel(saved_level)

    def tree_panel_state(self, ds_tree: Tree) -> TreePanelState:
        """Build address-order text and ASCII tree for *ds_tree*; store as ``last_tree``."""
        self.last_tree = ds_tree
        address = format_ds_tree(ds_tree)
        ascii_art = format_ds_tree_ascii(ds_tree)
        return TreePanelState(address_order=address, parse_tree_ascii=ascii_art)

    def current_view_strings(self) -> ViewStrings | None:
        """Return strings for all tabs from current parser state, or ``None`` if no parser."""
        if self.parser is None:
            return None
        ds_tree = self.parser.get_best_tuple().get_tree()
        tree_state = self.tree_panel_state(ds_tree)
        dag = format_dag_overview(self.parser.get_state())
        try:
            raw = str(self.parser.get_final_semantics())
            sem = format_semantics_display(raw) if raw.strip() else "(no semantics yet)"
        except (TypeError, ValueError) as ex:
            sem = f"(could not read semantics: {ex})"
        return ViewStrings(
            address_order=tree_state.address_order,
            parse_tree_ascii=tree_state.parse_tree_ascii,
            semantics=sem,
            dag=dag,
        )

    def run_init(self, *, success_event: str = "Init — axiom state.") -> str | None:
        """Re-init the parser to the axiom state.

        *success_event* is the Logs line on success. The browser keeps the default
        ``Init — axiom state.`` The Flet Reset button passes its own line.
        Returns an error log line, or ``None`` on success.
        """
        if self.parser is None:
            verb = "Reset" if success_event.startswith("Reset") else "Init"
            self.last_event = f"{verb}: load a grammar first."
            return self.last_event
        self.parser.init()
        self._clear_consumed_words()
        self.refresh_interpretations()
        self.last_event = success_event
        return None

    def run_new_sentence(self) -> str | None:
        """Reset DAG for a new sentence; returns an error log block or ``None`` on success."""
        if self.parser is None:
            self.last_event = "New sentence: load a grammar first."
            return "New sentence: load a grammar first."
        self.parser.new_sentence()
        self._clear_consumed_words()
        self.refresh_interpretations()
        self.last_event = "New sentence — DAG reset to axiom."
        return None

    def _actions_for_consumed_index(self, index: int, word: str) -> list[str]:
        """Action-log lines for consumed word *index* across the recorded interpretations."""
        sequences: list[tuple[str, ...]] = []
        for path in self._interpretation_word_actions:
            if index < len(path) and path[index][0] == word:
                sequences.append(path[index][1])
            else:
                sequences.append(())
        return format_action_log_lines(sequences)

    def run_parse(
        self,
        sentence: str,
        *,
        reset_before: bool,
        speaker: str = DEFAULT_SPEAKER,
    ) -> tuple[str | None, bool | None, list[str]]:
        """Parse *sentence*, logging only words that are new to the derivation.

        Returns ``(error_or_none, parse_ok_or_none, log_lines)``.
        A failed word does not stop later words, matching ``parse_utterance``.
        When *reset_before* is true the derivation is cleared first and every word is logged.
        A sentence that does not extend the words already parsed is refused.
        """
        if self.parser is None:
            self.last_event = "Parse: load a grammar first."
            return ("Parse: load a grammar first.", None, [])
        text = sentence.strip()
        if not text:
            self.last_event = "Parse: enter a sentence."
            return ("Parse: enter a sentence.", None, [])
        if reset_before:
            self.parser.init()
            self._clear_consumed_words()
        utt = utterance_from_text(speaker, text)
        tokens = [uw.word or "" for uw in utt.words]
        prior = list(self._consumed_words)
        if tokens[: len(prior)] != prior:
            message = format_noncontinuation(text)
            self.last_event = message
            return (message, None, [])
        if len(tokens) == len(prior):
            self.last_event = NO_NEW_WORDS_LOG
            return (None, None, [NO_NEW_WORDS_LOG])
        suffix = utt.words[len(prior) :]
        prior_len = len(prior)
        outcomes: list[tuple[str, bool, str | None]] = []
        ok = True
        for uw in suffix:
            word = uw.word or ""
            if self.parser.parse_word(uw) is None:
                ok = False
                in_lexicon = bool(self.parser.lexicon.lookup(word))
                outcomes.append((word, False, None if in_lexicon else "lexicon"))
            else:
                self._consumed_words.append(word)
                outcomes.append((word, True, None))
        self.refresh_interpretations()
        events: list[str] = []
        success_index = prior_len
        for word, word_ok, reason in outcomes:
            events.append(format_word_event(word=word, ok=word_ok, reason=reason))
            if not word_ok:
                continue
            events.extend(self._actions_for_consumed_index(success_index, word))
            success_index += 1
        self.last_event = format_parse_event(
            sentence=text,
            ok=ok,
            missing_words=[
                (uw.word or "")
                for uw in utt.words
                if uw.word and not self.parser.lexicon.lookup(uw.word)
            ]
            if not ok
            else None,
        )
        return (None, ok, events)

    def run_step_through(self) -> tuple[str | None, bool | None]:
        """Advance to the next interpretation. Prefer :meth:`select_interpretation`."""
        if self.parser is None:
            self.last_event = "Interpretations: load a grammar first."
            return ("Interpretations: load a grammar first.", None)
        _err, log = self.select_interpretation(self.interpretation_index + 1)
        if log is None:
            self.last_event = "No further interpretation available."
            return (None, False)
        return (None, True)
