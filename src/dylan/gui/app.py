"""Flet desktop GUI for loading grammars and parsing (Java ``ParserGUI`` / ``ParserPanel`` subset).

Targets Flet >= 0.80 (async ``FilePicker`` methods, ``Tabs`` uses
``content`` + ``length``, ``Button.content`` keyword).
"""

from __future__ import annotations

import logging
from typing import Any
import sys
from pathlib import Path

from dylan.gui.formatting import format_dag_overview, format_ds_tree
from dylan.nlp.types import DEFAULT_SPEAKER, utterance_from_text
from dylan.parser.interactive_context_parser import InteractiveContextParser

logger = logging.getLogger(__name__)

_INFO_HELP_TEXT = (
    "Click Load grammar to pick a folder; the grammar loads as soon as you confirm.\n\n"
    "Pick again after changing Repair processing. "
    "The loaded path is listed under each grammar load.\n\n"
    "Grammar load details and parse output appear in Logs below."
)


def _estimate_wrapped_line_count(text: str, chars_per_line: int = 44) -> int:
    """Approximate how many display lines *text* needs when wrapped in a narrow panel."""
    total = 0
    for block in text.split("\n"):
        s = block.strip()
        if not s:
            total += 1
            continue
        total += max(1, (len(s) + chars_per_line - 1) // chars_per_line)
    return max(total, 2)


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
        page.window.width = 1400
        page.window.height = 900
        page.window.min_width = 1000
        page.window.min_height = 640

        parser_holder: list[InteractiveContextParser | None] = [None]
        grammar_path_holder: list[str | None] = [None]

        # --- GUI theme (single source for the four main panels + tabs) ----------------
        PANEL_BACKGROUND = "#263238"
        BOX_BACKGROUND_COLOR = "#1c262b"
        BOX_BORDER_COLOR = "#37474f"
        BODY_TEXT_COLOR = "#eceff1"
        MUTED_TEXT_COLOR = "#b0bec5"
        HINT_TEXT_COLOR = "#90a4ae"
        BODY_FONT_SIZE = 12
        MONO_FONT_SIZE = 13
        CAPTION_FONT_SIZE = 12
        MONO_FONT_FAMILY = "Consolas, monospace"
        PROSE_LINE_HEIGHT = 1.35

        body_text_style = ft.TextStyle(
            size=BODY_FONT_SIZE,
            color=BODY_TEXT_COLOR,
            height=PROSE_LINE_HEIGHT,
        )
        mono_text_style = ft.TextStyle(
            font_family=MONO_FONT_FAMILY,
            size=MONO_FONT_SIZE,
            color=BODY_TEXT_COLOR,
        )
        field_label_style = ft.TextStyle(color=MUTED_TEXT_COLOR)
        hint_text_style = ft.TextStyle(color=HINT_TEXT_COLOR)
        toolbar_label_style = ft.TextStyle(color=BODY_TEXT_COLOR)

        _outline = ft.InputBorder.OUTLINE
        _primary_btn_shape = ft.RoundedRectangleBorder(radius=6)
        _primary_btn_padding = ft.padding.symmetric(horizontal=16, vertical=12)
        page.bgcolor = PANEL_BACKGROUND

        def _dark_outlined_textfield(**kwargs: Any) -> ft.TextField:
            """TextField with shared dark filled outline styling; callers pass field-specific kwargs."""
            defaults: dict[str, Any] = {
                "border": _outline,
                "filled": True,
                "fill_color": BOX_BACKGROUND_COLOR,
                "hover_color": BOX_BACKGROUND_COLOR,
                "focused_bgcolor": BOX_BACKGROUND_COLOR,
                "border_color": BOX_BORDER_COLOR,
            }
            defaults.update(kwargs)
            return ft.TextField(**defaults)

        def _dark_borderless_textfield(**kwargs: Any) -> ft.TextField:
            """TextField with shared dark fill and no inner outline (used inside captioned boxes)."""
            defaults: dict[str, Any] = {
                "border": ft.InputBorder.NONE,
                "filled": True,
                "fill_color": BOX_BACKGROUND_COLOR,
                "hover_color": BOX_BACKGROUND_COLOR,
                "focused_bgcolor": BOX_BACKGROUND_COLOR,
            }
            defaults.update(kwargs)
            return ft.TextField(**defaults)

        def _border_caption_box(
            title: str,
            content: ft.Control,
            *,
            caption_bg: str,
            fill_color: str | None,
            border_color: str | None = None,
            content_padding: ft.PaddingValue | None = None,
            expand: bool = False,
            title_color: str | None = None,
            fill_vertical: bool = False,
        ) -> ft.Container:
            """Outlined box with *title* on the top border; use *fill_vertical* only inside bounded flex areas (tabs/logs)."""
            bc = border_color if border_color is not None else BOX_BORDER_COLOR
            tc = title_color if title_color is not None else MUTED_TEXT_COLOR
            pad = content_padding if content_padding is not None else ft.padding.all(10)
            caption_text = ft.Text(
                title,
                size=CAPTION_FONT_SIZE,
                color=tc,
            )
            # Row + Column gives unbounded max height to non-flex children; an expanding Stack
            # then claims infinite height and hides siblings. Use loose Stack + intrinsic height
            # unless *fill_vertical* (tabs / logs) where the parent supplies a bounded flex area.
            if fill_vertical:
                bordered = ft.Container(
                    margin=ft.margin.only(top=8),
                    expand=True,
                    border=ft.border.all(1, bc),
                    border_radius=4,
                    bgcolor=fill_color,
                    padding=pad,
                    content=content,
                )
                stack = ft.Stack(
                    fit=ft.StackFit.EXPAND,
                    expand=True,
                    controls=[
                        bordered,
                        ft.Container(
                            left=12,
                            top=0,
                            bgcolor=caption_bg,
                            padding=ft.padding.symmetric(horizontal=4),
                            content=caption_text,
                        ),
                    ],
                )
            else:
                bordered = ft.Container(
                    margin=ft.margin.only(top=8),
                    border=ft.border.all(1, bc),
                    border_radius=4,
                    bgcolor=fill_color,
                    padding=pad,
                    content=content,
                )
                stack = ft.Stack(
                    fit=ft.StackFit.LOOSE,
                    controls=[
                        bordered,
                        ft.Container(
                            left=12,
                            top=0,
                            bgcolor=caption_bg,
                            padding=ft.padding.symmetric(horizontal=4),
                            content=caption_text,
                        ),
                    ],
                )
            return ft.Container(expand=expand, content=stack)
        load_grammar_btn = ft.FilledButton(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.FOLDER_OPEN, size=20),
                    ft.Text("Load grammar"),
                ],
                tight=True,
                spacing=8,
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            tooltip="Open a folder dialog; the selected directory is loaded as the grammar",
            style=ft.ButtonStyle(
                padding=_primary_btn_padding,
                shape=_primary_btn_shape,
            ),
        )
        sentence_field = _dark_borderless_textfield(
            hint_text="Enter your sentence here...",
            multiline=True,
            min_lines=3,
            max_lines=6,
            expand=True,
            color=BODY_TEXT_COLOR,
            cursor_color=BODY_TEXT_COLOR,
            hint_style=hint_text_style,
        )
        sentence_box = _border_caption_box(
            "Sentence",
            sentence_field,
            caption_bg=BOX_BACKGROUND_COLOR,
            fill_color=BOX_BACKGROUND_COLOR,
            expand=True,
        )
        _info_lines = _estimate_wrapped_line_count(_INFO_HELP_TEXT)
        info_field = _dark_borderless_textfield(
            value=_INFO_HELP_TEXT,
            read_only=True,
            multiline=True,
            min_lines=_info_lines,
            max_lines=_info_lines,
            expand=True,
            text_style=body_text_style,
            dense=True,
        )
        info_box = _border_caption_box(
            "Info",
            info_field,
            caption_bg=BOX_BACKGROUND_COLOR,
            fill_color=BOX_BACKGROUND_COLOR,
            expand=True,
        )
        log_text = ft.Text(
            value="",
            style=mono_text_style,
            selectable=True,
        )
        log_scroll = ft.Column(
            controls=[log_text],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )
        log_box = _border_caption_box(
            "Logs",
            log_scroll,
            caption_bg=BOX_BACKGROUND_COLOR,
            fill_color=BOX_BACKGROUND_COLOR,
            expand=True,
            fill_vertical=True,
        )

        tree_view = _dark_borderless_textfield(
            multiline=True,
            read_only=True,
            min_lines=6,
            max_lines=None,
            expand=True,
            text_style=mono_text_style,
        )
        output_box = _border_caption_box(
            "Output",
            tree_view,
            caption_bg=BOX_BACKGROUND_COLOR,
            fill_color=BOX_BACKGROUND_COLOR,
            expand=True,
            fill_vertical=True,
        )
        sem_view = _dark_outlined_textfield(
            label="Semantics",
            label_style=field_label_style,
            multiline=True,
            read_only=True,
            min_lines=6,
            max_lines=None,
            expand=True,
            text_style=mono_text_style,
        )
        dag_view = _dark_outlined_textfield(
            label="DAG",
            label_style=field_label_style,
            multiline=True,
            read_only=True,
            min_lines=6,
            max_lines=None,
            expand=True,
            text_style=mono_text_style,
        )
        reset_before = ft.Checkbox(
            label="Reset state before parse",
            value=True,
            label_style=toolbar_label_style,
        )
        repair_cb = ft.Checkbox(
            label="Repair processing",
            value=False,
            tooltip="NOTE: repair path only partially ported; may fail on repairs.",
            label_style=toolbar_label_style,
        )

        # --- helpers ----------------------------------------------------------

        def set_log(text: str) -> None:
            """Replace the log panel content, mirror it to stderr, and refresh the page."""
            log_text.value = text
            print(text, file=sys.stderr, flush=True)
            page.update()

        def append_log(text: str) -> None:
            """Append a block to the log panel, mirror that block to stderr, and refresh."""
            cur = (log_text.value or "").rstrip()
            log_text.value = f"{cur}\n\n{text}" if cur else text
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
            """Open a native directory picker for the grammar folder and load it."""
            path = await ft.FilePicker().get_directory_path(
                dialog_title="Select grammar folder",
            )
            if path:
                grammar_path_holder[0] = path
                apply_grammar(path)

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
            sentence = (sentence_field.value or "").strip()
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

        sentence_field.on_submit = do_parse
        parse_btn = ft.FilledButton(
            content="Parse",
            icon=ft.Icons.PLAY_ARROW,
            on_click=do_parse,
            style=ft.ButtonStyle(
                padding=_primary_btn_padding,
                shape=_primary_btn_shape,
            ),
        )
        load_grammar_btn.on_click = pick_grammar_dir

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
                            ft.Container(content=output_box, padding=6, expand=True),
                            ft.Container(content=sem_view, padding=6, expand=True),
                            ft.Container(content=dag_view, padding=6, expand=True),
                        ],
                    ),
                ],
            ),
        )

        init_btn = ft.ElevatedButton(
            content="Init",
            on_click=do_init,
            tooltip="context.init()",
        )
        new_sentence_btn = ft.ElevatedButton(
            content="New sentence",
            on_click=do_new_sentence,
            tooltip="Reset DAG to axiom (Java newSentence)",
        )
        grammar_toolbar = ft.Row(
            [
                load_grammar_btn,
                ft.Container(expand=True),
                repair_cb,
                reset_before,
                init_btn,
                new_sentence_btn,
            ],
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        sentence_row = ft.Container(
            content=ft.Row(
                [
                    sentence_box,
                    parse_btn,
                ],
                spacing=8,
            ),
            padding=ft.padding.only(top=8),
        )
        left_column = ft.Column(
            [
                grammar_toolbar,
                sentence_row,
                tabs_widget,
            ],
            expand=True,
            spacing=4,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        logs_panel = ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        controls=[info_box],
                        tight=True,
                        vertical_alignment=ft.CrossAxisAlignment.START,
                    ),
                    ft.Container(content=log_box, expand=True),
                ],
                expand=True,
                spacing=10,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
            bgcolor=PANEL_BACKGROUND,
            padding=12,
            border_radius=8,
            expand=35,
        )
        page.add(
            ft.Row(
                [
                    ft.Container(
                        content=left_column,
                        expand=65,
                        padding=ft.padding.only(right=8),
                        bgcolor=PANEL_BACKGROUND,
                    ),
                    logs_panel,
                ],
                expand=True,
                spacing=0,
                vertical_alignment=ft.CrossAxisAlignment.STRETCH,
            )
        )

        async def _center_window() -> None:
            """Place the window in the middle of the screen after size is applied."""
            await page.window.center()

        page.run_task(_center_window)

    ft.app(target=build)


if __name__ == "__main__":
    main()
