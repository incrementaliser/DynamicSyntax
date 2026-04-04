"""Flet desktop GUI for loading grammars and parsing (Java ``ParserGUI`` / ``ParserPanel`` subset).

Targets Flet >= 0.80 (async ``FilePicker`` methods, ``Tabs`` uses
``content`` + ``length``, ``Button.content`` keyword).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from dylan.gui.formatting import format_dag_overview, format_ds_tree
from dylan.nlp.types import DEFAULT_SPEAKER, utterance_from_text
from dylan.parser.interactive_context_parser import InteractiveContextParser

logger = logging.getLogger(__name__)


class _GrammarLoadLogFilter(logging.Filter):
    """Keep INFO+ from all ``dylan`` loggers; DEBUG only from lexicon (skipped lines, etc.)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.INFO:
            return True
        return record.name.startswith("dylan.action.lexicon")


class _DylanCaptureHandler(logging.Handler):
    """Collect log records under the ``dylan`` namespace while a grammar is loading."""

    def __init__(self, lines: list[str]) -> None:
        super().__init__(level=logging.DEBUG)
        self._lines = lines
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        self.addFilter(_GrammarLoadLogFilter())

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._lines.append(self.format(record))
        except Exception:
            self.handleError(record)


def _ensure_dylan_stderr_logging() -> None:
    """Attach a stderr handler on ``dylan`` so loader/parser logs appear in the terminal."""
    lg = logging.getLogger("dylan")
    for h in lg.handlers:
        if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stderr:
            return
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    lg.addHandler(sh)
    lg.propagate = False


def main() -> None:
    """Entry point for ``dylan-gui`` console script; opens the Flet desktop app."""
    import flet as ft

    def build(page: ft.Page) -> None:
        """Lay out controls mirroring the Java parser frame."""
        if not logging.root.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(levelname)s %(name)s: %(message)s",
                stream=sys.stderr,
                force=False,
            )
        _ensure_dylan_stderr_logging()

        page.title = "DyLan - The Dynamic Syntax Parser"
        page.window.width = 1100
        page.window.height = 780
        page.window.min_width = 800
        page.window.min_height = 560

        parser_holder: list[InteractiveContextParser | None] = [None]

        grammar_browse_btn = ft.IconButton(
            icon=ft.Icons.FOLDER_OPEN,
            tooltip="Browse grammar folder",
            icon_size=22,
            style=ft.ButtonStyle(padding=4),
        )
        grammar_field = ft.TextField(
            label="Grammar folder",
            hint_text="Directory with lexicon.txt, lexical-actions.txt, computational-actions.txt",
            expand=True,
            tooltip="Same layout as Java resource folder",
            suffix=grammar_browse_btn,
        )
        sentence_corpus_field = ft.TextField(
            label="Sentence/corpus",
            hint_text="Whitespace-separated tokens; multiline scratch pad",
            multiline=True,
            min_lines=4,
            max_lines=12,
            expand=True,
        )
        log_mono = ft.TextStyle(
            font_family="Consolas, monospace",
            size=12,
            color="#eceff1",
        )
        log_view = ft.TextField(
            read_only=True,
            multiline=True,
            min_lines=8,
            max_lines=32,
            expand=True,
            text_style=log_mono,
            bgcolor="#1c262b",
            border_color="#37474f",
            hint_text="Grammar load messages, warnings, and recent actions",
        )

        mono = ft.TextStyle(font_family="Consolas, monospace", size=13)
        tree_view = ft.TextField(
            multiline=True, read_only=True, min_lines=18, expand=True, text_style=mono,
        )
        sem_view = ft.TextField(
            multiline=True, read_only=True, min_lines=18, expand=True, text_style=mono,
        )
        dag_view = ft.TextField(
            multiline=True, read_only=True, min_lines=18, expand=True, text_style=mono,
        )
        reset_before = ft.Checkbox(label="Reset state before parse", value=True)
        repair_cb = ft.Checkbox(
            label="Repair processing",
            value=False,
            tooltip="NOTE: repair path only partially ported; may fail on repairs.",
        )

        # --- helpers ----------------------------------------------------------

        def set_log(text: str) -> None:
            """Replace the log panel content, mirror it to stderr, and refresh the page."""
            log_view.value = text
            print(text, file=sys.stderr, flush=True)
            page.update()

        def append_log(text: str) -> None:
            """Append a block to the log panel, mirror that block to stderr, and refresh."""
            cur = (log_view.value or "").rstrip()
            log_view.value = f"{cur}\n\n{text}" if cur else text
            print(text, file=sys.stderr, flush=True)
            page.update()

        def _format_grammar_load_report(
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

        def apply_grammar(path_str: str) -> None:
            """Load a grammar directory into a fresh parser."""
            p = Path(path_str.strip())
            if not p.is_dir():
                set_log(_format_grammar_load_report(p, [], ok=False, error=f"Not a directory: {p}"))
                return
            captured: list[str] = []
            cap = _DylanCaptureHandler(captured)
            dylan_log = logging.getLogger("dylan")
            saved_level = dylan_log.level
            dylan_log.setLevel(logging.DEBUG)
            dylan_log.addHandler(cap)
            try:
                try:
                    parser_holder[0] = InteractiveContextParser(
                        p, repairing=bool(repair_cb.value),
                    )
                    parser_holder[0].init()
                    set_log(_format_grammar_load_report(p, captured, ok=True))
                except OSError as ex:
                    parser_holder[0] = None
                    set_log(
                        _format_grammar_load_report(
                            p, captured, ok=False, error=str(ex),
                        ),
                    )
                except Exception as ex:  # noqa: BLE001
                    parser_holder[0] = None
                    logger.exception("Grammar load failed")
                    set_log(
                        _format_grammar_load_report(
                            p,
                            captured,
                            ok=False,
                            error=f"{type(ex).__name__}: {ex}",
                        ),
                    )
            finally:
                dylan_log.removeHandler(cap)
                dylan_log.setLevel(saved_level)

        def _refresh_views(par: InteractiveContextParser, msg: str) -> None:
            """Populate the tree / semantics / DAG text fields from current state."""
            dag = par.get_state()
            tree_view.value = format_ds_tree(par.get_best_tuple().get_tree())
            dag_view.value = format_dag_overview(dag)
            try:
                sem_view.value = str(par.get_final_semantics())
            except (TypeError, ValueError) as ex:
                sem_view.value = f"(could not read semantics: {ex})"
            append_log(f"=== Parse / state ===\n{msg}")

        # --- event handlers ---------------------------------------------------

        async def pick_grammar_dir(_: ft.ControlEvent | None = None) -> None:
            """Open a native directory picker for the grammar folder."""
            path = await ft.FilePicker().get_directory_path(
                dialog_title="Select grammar folder",
            )
            if path:
                grammar_field.value = path
                apply_grammar(path)

        def do_load_path(_: ft.ControlEvent | None = None) -> None:
            """Load the grammar from the path typed in the text field."""
            apply_grammar(grammar_field.value or "")

        def do_init(_: ft.ControlEvent | None = None) -> None:
            """Re-initialise the parser to the axiom state."""
            par = parser_holder[0]
            if par is None:
                append_log("=== Init ===\nLoad grammar first.")
                return
            par.init()
            _refresh_views(par, "Parser re-initialised (axiom state).")

        def do_new_sentence(_: ft.ControlEvent | None = None) -> None:
            """Reset the DAG for a fresh sentence."""
            par = parser_holder[0]
            if par is None:
                append_log("=== New sentence ===\nLoad grammar first.")
                return
            par.new_sentence()
            _refresh_views(par, "New sentence — DAG reset to axiom.")

        def do_parse(_: ft.ControlEvent | None = None) -> None:
            """Parse the sentence in the text field."""
            par = parser_holder[0]
            if par is None:
                append_log("=== Parse ===\nLoad grammar first.")
                return
            if reset_before.value:
                par.init()
            sp = DEFAULT_SPEAKER
            sentence = (sentence_corpus_field.value or "").strip()
            if not sentence:
                append_log("=== Parse ===\nEnter a sentence to parse.")
                return
            utt = utterance_from_text(sp, sentence)
            ok = par.parse_utterance(utt)
            msg = (
                "Parse OK."
                if ok
                else "Parse finished with failures (check sentence / lexicon)."
            )
            _refresh_views(par, msg)

        sentence_corpus_field.on_submit = do_parse
        grammar_browse_btn.on_click = pick_grammar_dir

        # --- app bar ----------------------------------------------------------

        page.appbar = ft.AppBar(
            title=ft.Text("DyLan - The Dynamic Syntax Parser"),
            center_title=True,
            automatically_imply_leading=False,
            bgcolor="#455a64",
            color="white",
        )

        # --- layout -----------------------------------------------------------

        tabs_widget = ft.Tabs(
            length=3,
            selected_index=0,
            expand=True,
            content=ft.Column(
                expand=True,
                controls=[
                    ft.TabBar(
                        tabs=[
                            ft.Tab(label="Parse tree", icon=ft.Icons.ACCOUNT_TREE),
                            ft.Tab(label="Semantics", icon=ft.Icons.DATA_OBJECT),
                            ft.Tab(label="DAG", icon=ft.Icons.HUB),
                        ],
                    ),
                    ft.TabBarView(
                        expand=True,
                        controls=[
                            ft.Container(content=tree_view, padding=12, expand=True),
                            ft.Container(content=sem_view, padding=12, expand=True),
                            ft.Container(content=dag_view, padding=12, expand=True),
                        ],
                    ),
                ],
            ),
        )

        page.add(
            ft.Column(
                [
                    ft.Row(
                        [
                            grammar_field,
                            ft.ElevatedButton(
                                content="Load",
                                on_click=do_load_path,
                            ),
                        ],
                        spacing=8,
                    ),
                    ft.Row(
                        [
                            repair_cb,
                            reset_before,
                            ft.ElevatedButton(
                                content="Init",
                                on_click=do_init,
                                tooltip="context.init()",
                            ),
                            ft.ElevatedButton(
                                content="New sentence",
                                on_click=do_new_sentence,
                                tooltip="Reset DAG to axiom (Java newSentence)",
                            ),
                        ],
                        spacing=12,
                    ),
                    ft.Container(
                        content=ft.Row(
                            [
                                sentence_corpus_field,
                                ft.FilledButton(
                                    content="Parse",
                                    icon=ft.Icons.PLAY_ARROW,
                                    on_click=do_parse,
                                ),
                            ],
                            spacing=8,
                        ),
                        padding=ft.padding.only(top=24),
                    ),
                    tabs_widget,
                    ft.Container(
                        content=ft.Row(
                            [log_view],
                            expand=True,
                            spacing=0,
                        ),
                        bgcolor="#263238",
                        padding=10,
                        border_radius=6,
                        expand=True,
                    ),
                ],
                expand=True,
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            )
        )
        set_log(
            "Load a grammar folder with Load or the folder icon, then enter a sentence and Parse.\n"
            "Loader warnings (e.g. skipped lexicon lines) appear below after each grammar load.",
        )

    ft.app(target=build)


if __name__ == "__main__":
    main()
