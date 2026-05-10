"""Flet desktop GUI for loading grammars and parsing (Java ``ParserGUI`` / ``ParserPanel`` subset).

Targets Flet >= 0.80 (async ``FilePicker`` methods, ``Tabs`` uses
``content`` + ``length``, four main tabs: parse graph, address list, semantics, DAG).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

# Running ``python .../src/dylan/gui/app.py`` only puts ``gui/`` on sys.path; add the
# directory that contains the ``dylan`` package (checkout ``src/``).
_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from dylan.gui.parse_session import (
    GUI_INFO_HELP_TEXT,
    ParseSession,
    format_parse_state_log,
)
from dylan.gui.tree_viz import build_canvas_shapes, compute_tree_layout
from dylan.nlp.types import DEFAULT_SPEAKER
from dylan.tree.tree import Tree

logger = logging.getLogger(__name__)

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
    try:
        import flet as ft
        import flet.canvas as cv
    except ImportError as exc:
        raise SystemExit(
            "The DyLan GUI needs Flet (optional dependency). From the repo root run:\n"
            "  uv sync --group dev\n"
            "or: uv pip install -e \".[gui]\"\n"
            "PyPI installs: pip install \"dynamicsyntax[gui]\""
        ) from exc

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

        def on_page_error(e: ft.ControlEvent) -> None:
            """Log Flet client errors (red banner text is often mirrored here)."""
            logger.warning("Flet page error: %s", getattr(e, "data", e))

        page.on_error = on_page_error

        page.window.width = 1400
        page.window.height = 900
        page.window.min_width = 1000
        page.window.min_height = 640

        session = ParseSession()

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
        _primary_btn_padding = ft.Padding.symmetric(horizontal=16, vertical=12)
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
            pad = content_padding if content_padding is not None else ft.Padding.all(10)
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
                    margin=ft.Margin.only(top=8),
                    expand=True,
                    border=ft.Border.all(width=1, color=bc),
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
                            padding=ft.Padding.symmetric(horizontal=4),
                            content=caption_text,
                        ),
                    ],
                )
            else:
                bordered = ft.Container(
                    margin=ft.Margin.only(top=8),
                    border=ft.Border.all(width=1, color=bc),
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
                            padding=ft.Padding.symmetric(horizontal=4),
                            content=caption_text,
                        ),
                    ],
                )
            return ft.Container(expand=expand, content=stack)
        set_grammar_btn = ft.FilledButton(
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
        _info_lines = _estimate_wrapped_line_count(GUI_INFO_HELP_TEXT)
        info_field = _dark_borderless_textfield(
            value=GUI_INFO_HELP_TEXT,
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
        tree_zoom_label = ft.Text(
            "Zoom: 100%",
            size=11,
            color=MUTED_TEXT_COLOR,
        )
        tree_canvas = cv.Canvas(
            expand=True,
            shapes=[],
        )
        tree_canvas_viewer = ft.InteractiveViewer(
            content=tree_canvas,
            expand=True,
            pan_enabled=True,
            scale_enabled=True,
            trackpad_scroll_causes_scale=True,
            min_scale=0.5,
            max_scale=4.0,
            boundary_margin=200,
            constrained=False,
            scale_factor=200,
        )
        parse_tree_graph_column = ft.Column(
            [
                ft.Row(
                    [tree_zoom_label],
                    alignment=ft.MainAxisAlignment.END,
                ),
                ft.Container(content=tree_canvas_viewer, expand=True),
            ],
            expand=True,
        )
        output_box = _border_caption_box(
            "Output",
            parse_tree_graph_column,
            caption_bg=BOX_BACKGROUND_COLOR,
            fill_color=BOX_BACKGROUND_COLOR,
            expand=True,
            fill_vertical=True,
        )
        address_order_box = _border_caption_box(
            "Address order",
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
        show_logs_toggle = ft.Checkbox(
            label="Logs",
            value=True,
            label_style=ft.TextStyle(color=BODY_TEXT_COLOR, size=11),
            tooltip="Show or hide Info and Logs (right panel)",
            visual_density=ft.VisualDensity.COMPACT,
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

        def apply_grammar(path_str: str) -> None:
            """Load a grammar directory into a fresh parser."""
            log_text = session.set_grammar(path_str, repairing=bool(repair_cb.value))
            set_log(log_text)

        def _parse_tree_render_px(*, logs_column_visible: bool) -> tuple[int, int]:
            """Approximate pixels available for the parse-tree canvas inside the Output tab."""
            w = float(page.window.width or page.width or 1400)
            h = float(page.window.height or page.height or 900)
            left_ratio = 0.65 if logs_column_visible else 1.0
            chrome_x = 80.0
            left_w = max(400.0, w * left_ratio - chrome_x)
            chrome_y = 268.0
            tree_h = max(320.0, h - chrome_y)
            return int(left_w), int(tree_h)

        def _paint_parse_tree_canvas(w: float, h: float) -> None:
            """Re-draw ``tree_canvas`` from ``session.last_tree`` at logical size *(w, h)*."""
            wf = max(40.0, float(w))
            hf = max(40.0, float(h))
            tree_canvas.width = wf
            tree_canvas.height = hf
            ds_tree = session.last_tree
            if ds_tree is None or not ds_tree:
                tree_canvas.shapes = [
                    cv.Rect(
                        x=0,
                        y=0,
                        width=wf,
                        height=hf,
                        paint=ft.Paint(style=ft.PaintingStyle.FILL, color=PANEL_BACKGROUND),
                    ),
                ]
                return
            layout = compute_tree_layout(
                ds_tree,
                wf,
                hf,
                font_size=float(MONO_FONT_SIZE),
            )
            tree_canvas.shapes = build_canvas_shapes(
                layout,
                ds_tree.pointer,
                font_size=float(MONO_FONT_SIZE),
            )

        def on_tree_zoom_update(e: ft.ScaleUpdateEvent) -> None:
            """Show the current interactive-viewer zoom factor while the user zooms/pans."""
            tree_zoom_label.value = f"Zoom: {int(round(float(e.scale) * 100.0))}%"
            tree_zoom_label.update()

        def _refresh_parse_tree_visual(ds_tree: Tree) -> None:
            """Fill address-order text and re-paint the parse-tree canvas from *ds_tree*."""
            tw, th = _parse_tree_render_px(logs_column_visible=bool(show_logs_toggle.value))
            st = session.tree_panel_state(ds_tree)
            tree_view.value = st.address_order
            _paint_parse_tree_canvas(float(tw), float(th))

        def _refresh_views(msg: str) -> None:
            """Populate the tree / semantics / DAG text fields from current parser state."""
            tw, th = _parse_tree_render_px(logs_column_visible=bool(show_logs_toggle.value))
            vs = session.current_view_strings()
            if vs is None:
                append_log(format_parse_state_log(msg))
                return
            tree_view.value = vs.address_order
            _paint_parse_tree_canvas(float(tw), float(th))
            dag_view.value = vs.dag
            sem_view.value = vs.semantics
            append_log(format_parse_state_log(msg))

        # --- event handlers ---------------------------------------------------

        async def pick_grammar_dir(_: ft.ControlEvent | None = None) -> None:
            """Open a native directory picker for the grammar folder and load it."""
            path = await ft.FilePicker().get_directory_path(
                dialog_title="Select grammar folder",
            )
            if path:
                apply_grammar(path)

        def do_init(_: ft.ControlEvent | None = None) -> None:
            """Re-initialise the parser to the axiom state."""
            err = session.run_init()
            if err is not None:
                append_log(err)
                return
            _refresh_views("Parser re-initialised (axiom state).")

        def do_new_sentence(_: ft.ControlEvent | None = None) -> None:
            """Reset the DAG for a fresh sentence."""
            err = session.run_new_sentence()
            if err is not None:
                append_log(err)
                return
            _refresh_views("New sentence — DAG reset to axiom.")

        def do_parse(_: ft.ControlEvent | None = None) -> None:
            """Parse the sentence in the text field."""
            err, ok = session.run_parse(
                sentence_field.value or "",
                reset_before=bool(reset_before.value),
                speaker=DEFAULT_SPEAKER,
            )
            if err is not None:
                append_log(err)
                return
            msg = (
                "Parse OK."
                if ok
                else "Parse finished with failures (check sentence / lexicon)."
            )
            _refresh_views(msg)

        def do_step_through(_: ft.ControlEvent | None = None) -> None:
            """Advance to the next interpretation from the current parser state."""
            err, ok = session.run_step_through()
            if err is not None:
                append_log(err)
                return
            msg = "Stepped to next interpretation." if ok else "No further interpretation available."
            _refresh_views(msg)

        def on_tree_canvas_resize(e: cv.CanvasResizeEvent) -> None:
            """Re-layout the parse tree when the canvas control receives its real size."""
            aw = float(e.width)
            ah = float(e.height)
            if aw < 32.0 or ah < 32.0:
                return
            if session.last_tree is not None:
                _paint_parse_tree_canvas(aw, ah)

        tree_canvas.on_resize = on_tree_canvas_resize
        tree_canvas_viewer.on_interaction_update = on_tree_zoom_update

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
        step_through_btn = ft.Button(
            content="Step Through",
            on_click=do_step_through,
            tooltip="Advance to the next parse interpretation (Java Step Through)",
        )
        set_grammar_btn.on_click = pick_grammar_dir

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
            length=4,
            selected_index=0,
            expand=True,
            content=ft.Column(
                expand=True,
                controls=[
                    ft.TabBar(
                        scrollable=False,
                        tab_alignment=ft.TabAlignment.FILL,
                        tabs=[
                            ft.Tab(label="Parse tree", icon=ft.Icons.ACCOUNT_TREE),
                            ft.Tab(
                                label="Address order",
                                icon=ft.Icons.FORMAT_LIST_BULLETED,
                            ),
                            ft.Tab(label="Semantics", icon=ft.Icons.DATA_OBJECT),
                            ft.Tab(label="DAG", icon=ft.Icons.HUB),
                        ],
                    ),
                    ft.TabBarView(
                        expand=True,
                        controls=[
                            ft.Container(content=output_box, padding=4, expand=True),
                            ft.Container(content=address_order_box, padding=6, expand=True),
                            ft.Container(content=sem_view, padding=6, expand=True),
                            ft.Container(content=dag_view, padding=6, expand=True),
                        ],
                    ),
                ],
            ),
        )

        init_btn = ft.Button(
            content="Init",
            on_click=do_init,
            tooltip="context.init()",
        )
        new_sentence_btn = ft.Button(
            content="New sentence",
            on_click=do_new_sentence,
            tooltip="Reset DAG to axiom (Java newSentence)",
        )
        grammar_toolbar = ft.Row(
            [
                set_grammar_btn,
                show_logs_toggle,
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
                    step_through_btn,
                ],
                spacing=8,
            ),
            padding=ft.Padding.only(top=8),
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
        left_wrap = ft.Container(
            content=left_column,
            expand=65,
            padding=ft.Padding.only(right=8),
            bgcolor=PANEL_BACKGROUND,
        )

        def on_show_logs_change(e: ft.ControlEvent) -> None:
            """Show or hide the Info/Logs column; when off, the parse UI uses the full width."""
            show = bool(e.control.value)
            logs_panel.visible = show
            left_wrap.expand = 65 if show else True
            if session.parser is not None and session.last_tree is not None:
                _refresh_parse_tree_visual(session.last_tree)
            page.update()

        show_logs_toggle.on_change = on_show_logs_change

        def on_window_resize(_: ft.ControlEvent | None = None) -> None:
            """Re-layout the parse-tree canvas when the window grows or shrinks."""
            if session.parser is None or session.last_tree is None:
                return
            _refresh_parse_tree_visual(session.last_tree)
            page.update()

        page.on_resize = on_window_resize

        page.add(
            ft.Row(
                [
                    left_wrap,
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

    use_web = (
        os.environ.get("DYLAN_FLET_WEB", "").strip().lower() in ("1", "true", "yes", "on")
        or os.environ.get("CODESPACES", "").strip().lower() == "true"
    )
    if use_web:
        ft.run(
            main=build,
            view=ft.AppView.WEB_BROWSER,
            port=int(os.environ.get("DYLAN_FLET_PORT", "8550")),
        )
    else:
        ft.run(main=build)


if __name__ == "__main__":
    main()
