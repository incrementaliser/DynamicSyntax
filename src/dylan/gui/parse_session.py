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
    "Click Load grammar to pick a folder; the grammar loads as soon as you confirm.\n\n"
    "Pick again after changing Repair processing. "
    "The loaded path is listed under each grammar load.\n\n"
    "Grammar load details and parse output appear in Logs below."
)


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
    """Build user-visible grammar load text including captured ``dylan`` log lines."""
    lines = [
        "=== Grammar load ===",
        f"Directory: {path.resolve()}",
    ]
    if ok:
        lines.append("Parser object created and init() completed.")
    if error:
        lines.append(f"Error: {error}")
    if captured:
        lines.append("")
        lines.append("Messages from loaders (warnings often mean skipped lexicon lines):")
        lines.extend(captured)
    elif ok:
        lines.append("")
        lines.append(
            "(No loader messages matched the log capture filter during load — "
            "if entries are missing, they may be below INFO level outside lexicon.)",
        )
    warn_count = sum(1 for s in captured if s.startswith("WARNING "))
    if ok and warn_count:
        lines.append("")
        lines.append(
            f"Note: {warn_count} warning(s) above — review lexicon / templates; "
            "the grammar may be incomplete.",
        )
    return "\n".join(lines)


def format_parse_state_log(msg: str) -> str:
    """Standard log block after parse or state change."""
    return f"=== Parse / state ===\n{msg}"


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

    def set_grammar(self, path_str: str, *, repairing: bool) -> str:
        """Load grammar from a filesystem directory *path_str*; return log text and set ``self.parser``.

        Bundled grammar nicknames (e.g. ``\"ttr\"``) are only accepted on a parser from
        :func:`dynamicsyntax.icp` via :meth:`InteractiveContextParser.set_grammar`;
        this GUI/session path expects a real directory from the file picker.
        """
        p = Path(path_str.strip())
        if not p.is_dir():
            self.parser = None
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
                self.last_tree = self.parser.get_best_tuple().get_tree()
                return format_grammar_load_report(p, captured, ok=True)
            except OSError as ex:
                self.parser = None
                self.last_tree = None
                return format_grammar_load_report(p, captured, ok=False, error=str(ex))
            except Exception as ex:  # noqa: BLE001
                self.parser = None
                self.last_tree = None
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
            sem = format_semantics_display(str(self.parser.get_final_semantics()))
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
            return "=== Init ===\nLoad grammar first."
        self.parser.init()
        return None

    def run_new_sentence(self) -> str | None:
        """Reset DAG for a new sentence; returns an error log block or ``None`` on success."""
        if self.parser is None:
            return "=== New sentence ===\nLoad grammar first."
        self.parser.new_sentence()
        return None

    def run_parse(
        self,
        sentence: str,
        *,
        reset_before: bool,
        speaker: str = DEFAULT_SPEAKER,
    ) -> tuple[str | None, bool | None]:
        """Parse *sentence*; returns ``(error_log_or_none, parse_ok_or_none)``."""
        if self.parser is None:
            return ("=== Parse ===\nLoad grammar first.", None)
        text = sentence.strip()
        if not text:
            return ("=== Parse ===\nEnter a sentence to parse.", None)
        if reset_before:
            self.parser.init()
        utt = utterance_from_text(speaker, text)
        ok = self.parser.parse_utterance(utt)
        return (None, ok)
