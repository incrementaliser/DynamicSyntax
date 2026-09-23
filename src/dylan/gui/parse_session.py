"""Headless grammar load and parse session shared by the Flet GUI and the browser UI.

The static site calls into Python via :mod:`dylan.pyodide_api` (JSON); that module
delegates to :class:`ParseSession` here. There is no separate ``web_api`` under
``dylan.gui`` — only this session layer plus the Pyodide façade at package root.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

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

    def _clear_interpretations(self) -> None:
        """Forget the interpretation sequence (no parser, or no tree)."""
        self.interpretation_index = 0
        self.interpretation_count = 0
        self.interpretation_capped = False

    def refresh_interpretations(self) -> None:
        """Count interpretations from the post-parse anchor, then show the first.

        The walk stops at :data:`INTERPRETATION_CAP`. One further successful step
        marks the count as capped (``30+``). The parser is left on interpretation 1.
        """
        if self.parser is None:
            self._clear_interpretations()
            return
        dag = self.parser.get_state()
        capped = False
        count = 1
        try:
            dag.reset_to_first_tuple_after_last_word()
            while count < INTERPRETATION_CAP:
                if not self.parser.parse_goal(None):
                    break
                count += 1
            else:
                if self.parser.parse_goal(None):
                    capped = True
        finally:
            dag.reset_to_first_tuple_after_last_word()
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

    def session_info_text(self) -> str:
        """Return the live Info card for the current session fields and parser pointer."""
        pointer: str | None = None
        tuple_id: int | None = None
        if self.last_tree is not None:
            pointer = str(self.last_tree.pointer.address)
        if self.parser is not None:
            current = self.parser.get_state().get_current_tuple()
            tuple_id = int(current.tuple_id)
            pointer = str(current.get_tree().pointer.address)
        grammar = str(self.grammar_path) if self.grammar_path is not None else None
        return format_session_info(
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

    def run_init(self) -> str | None:
        """Re-init parser; returns an error log block or ``None`` on success."""
        if self.parser is None:
            self.last_event = "Init: load a grammar first."
            return "Init: load a grammar first."
        self.parser.init()
        self.refresh_interpretations()
        self.last_event = "Init — axiom state."
        return None

    def run_new_sentence(self) -> str | None:
        """Reset DAG for a new sentence; returns an error log block or ``None`` on success."""
        if self.parser is None:
            self.last_event = "New sentence: load a grammar first."
            return "New sentence: load a grammar first."
        self.parser.new_sentence()
        self.refresh_interpretations()
        self.last_event = "New sentence — DAG reset to axiom."
        return None

    def run_parse(
        self,
        sentence: str,
        *,
        reset_before: bool,
        speaker: str = DEFAULT_SPEAKER,
    ) -> tuple[str | None, bool | None, list[str]]:
        """Parse *sentence* word by word.

        Returns ``(error_or_none, parse_ok_or_none, per_word_log_lines)``.
        A failed word does not stop later words, matching ``parse_utterance``.
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
        utt = utterance_from_text(speaker, text)
        events: list[str] = []
        ok = True
        for uw in utt.words:
            word = uw.word or ""
            if self.parser.parse_word(uw) is None:
                ok = False
                in_lexicon = bool(self.parser.lexicon.lookup(word))
                events.append(
                    format_word_event(
                        word=word,
                        ok=False,
                        reason=None if in_lexicon else "lexicon",
                    ),
                )
            else:
                events.append(format_word_event(word=word, ok=True))
        self.refresh_interpretations()
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
