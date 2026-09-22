"""Flet desktop GUI for loading grammars and parsing (Java ``ParserGUI`` / ``ParserPanel`` subset).

Targets Flet >= 0.80 (async ``FilePicker`` methods, ``Tabs`` uses
``content`` + ``length``, four main tabs: parse graph, address list, semantics, DAG).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Running ``python .../src/dylan/gui/app.py`` only puts ``gui/`` on sys.path; add the
# directory that contains the ``dylan`` package (checkout ``src/``).
_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from loguru import logger

from dylan.logging_config import configure_logging

from dylan.gui.parse_session import (
    ParseSession,
    format_event_log,
    format_interpretation_readout,
    resolve_grammar_directory,
)
from dylan.gui.tree_viz import build_canvas_shapes, compute_tree_layout
from dylan.nlp.types import DEFAULT_SPEAKER
from dylan.tree.tree import Tree


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
        configure_logging("INFO")

        page.title = "DyLan - The Dynamic Syntax Parser"

        def on_page_error(e: ft.ControlEvent) -> None:
            """Log Flet client errors (red banner text is often mirrored here)."""
            logger.warning("Flet page error: {}", getattr(e, "data", e))

        page.on_error = on_page_error

        page.window.width = 1400
        page.window.height = 900
        page.window.min_width = 800
        page.window.min_height = 560

        session = ParseSession()
        grammar_picker = ft.FilePicker()
        page.services = [*page.services, grammar_picker]
        _repo_resources = Path(__file__).resolve().parents[3] / "resources"

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
            tooltip="Open a folder dialog and select the grammar directory (all files in that folder are loaded).",
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
        info_field = _dark_borderless_textfield(
            value=session.session_info_text(),
            read_only=True,
            multiline=True,
            min_lines=8,
            max_lines=14,
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
        LOG_ERROR_COLOR = "#ef5350"
        LOG_PARSED_COLOR = "#66bb6a"
        log_scroll = ft.Column(
            controls=[],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=10,
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
            tooltip="Node boxes stay at natural size. Scroll the pane to see a larger tree.",
        )
        fit_btn = ft.Button(
            content="Fit",
            tooltip="Scroll so the middle of the natural-size tree is in this pane.",
        )
        tree_canvas = cv.Canvas(
            shapes=[],
            width=400,
            height=320,
            expand=False,
        )
        tree_host = ft.Container(
            content=tree_canvas,
            alignment=ft.Alignment.CENTER,
            width=400,
            height=320,
        )
        tree_h_scroll = ft.Row(
            controls=[tree_host],
            scroll=ft.ScrollMode.AUTO,
        )
        tree_v_scroll = ft.Column(
            controls=[tree_h_scroll],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )
        tree_stage = ft.Container(
            content=tree_v_scroll,
            expand=True,
            height=320,
        )
        parse_tree_graph_column = ft.Column(
            [
                ft.Row(
                    [
                        fit_btn,
                        ft.Container(expand=True),
                        tree_zoom_label,
                    ],
                    alignment=ft.MainAxisAlignment.START,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                tree_stage,
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
            tooltip="If on, Init runs before each Parse so the sentence is not appended to the previous derivation.",
            label_style=toolbar_label_style,
        )
        repair_cb = ft.Checkbox(
            label="Repair processing (partial — may fail)",
            value=False,
            tooltip="Repair path is only partially ported; leave off unless you are testing repairs.",
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

        last_tree_bbox: list[tuple[float, float]] = [(40.0, 40.0)]
        measured_viewport: list[tuple[float, float]] = [(0.0, 0.0)]
        centering_guard: list[bool] = [False]

        def on_tree_stage_size(e: ft.LayoutSizeChangeEvent) -> None:
            """Remember the Output pane size and re-centre a painted tree in that pane."""
            w = float(getattr(e, "width", 0.0) or 0.0)
            h = float(getattr(e, "height", 0.0) or 0.0)
            if w < 32.0 or h < 32.0:
                return
            prev_w, prev_h = measured_viewport[0]
            measured_viewport[0] = (w, h)
            if abs(prev_w - w) < 1.0 and abs(prev_h - h) < 1.0:
                return
            if session.last_tree is None or centering_guard[0]:
                return
            bw, bh = last_tree_bbox[0]
            centering_guard[0] = True
            try:
                _place_tree_host(bw, bh)
                page.update()
            except Exception as ex:
                logger.debug("Tree recentre skipped: {}", ex)
            finally:
                centering_guard[0] = False

        tree_stage.on_size_change = on_tree_stage_size

        def _sync_info() -> None:
            """Refresh the Info card from the live session."""
            info_field.value = session.session_info_text()

        def _sync_interpretation_controls() -> None:
            """Show ``#interpretations: index / N`` and disable arrows at the ends."""
            interp_readout.value = format_interpretation_readout(
                session.interpretation_index,
                session.interpretation_count,
                capped=session.interpretation_capped,
            )
            prev_interp_btn.disabled = session.interpretation_index <= 1
            next_interp_btn.disabled = (
                session.interpretation_count <= 0
                or session.interpretation_index >= session.interpretation_count
            )

        def _log_block_control(block: str) -> ft.Text:
            """Build a selectable log line.

            Failure lines are red. On a successful word, only the word ``parsed`` is green.
            """
            mono = ft.TextStyle(
                font_family=MONO_FONT_FAMILY,
                size=MONO_FONT_SIZE,
                color=BODY_TEXT_COLOR,
            )
            is_need_grammar = "load a grammar first" in block.lower()
            is_word_fail = " failed" in block.lower() or block.lower().endswith("failed")
            if is_need_grammar or is_word_fail:
                return ft.Text(
                    value=block,
                    style=ft.TextStyle(
                        font_family=MONO_FONT_FAMILY,
                        size=MONO_FONT_SIZE,
                        color=LOG_ERROR_COLOR,
                    ),
                    selectable=True,
                )
            if "\n" not in block and block.endswith(" parsed"):
                token = block[: -len(" parsed")]
                return ft.Text(
                    spans=[
                        ft.TextSpan(text=f"{token} ", style=mono),
                        ft.TextSpan(
                            text="parsed",
                            style=ft.TextStyle(
                                font_family=MONO_FONT_FAMILY,
                                size=MONO_FONT_SIZE,
                                color=LOG_PARSED_COLOR,
                            ),
                        ),
                    ],
                    selectable=True,
                )
            return ft.Text(value=block, style=mono, selectable=True)

        def set_log(text: str) -> None:
            """Replace the log panel content, mirror it to stderr, and refresh the page."""
            block = format_event_log(text) if text else ""
            log_scroll.controls = [_log_block_control(block)] if block else []
            print(text, file=sys.stderr, flush=True)
            page.update()

        def append_log(text: str) -> None:
            """Append a block to the log panel, mirror that block to stderr, and refresh."""
            block = format_event_log(text) if text else ""
            if block:
                log_scroll.controls.append(_log_block_control(block))
            print(block, file=sys.stderr, flush=True)
            page.update()

        def _viewport_px() -> tuple[float, float]:
            """Return the measured Output pane, or a window-chrome estimate until the first layout."""
            w, h = measured_viewport[0]
            if w >= 32.0 and h >= 32.0:
                return w, h
            win_w = float(page.window.width or page.width or 1400)
            win_h = float(page.window.height or page.height or 900)
            left_ratio = 0.65 if bool(show_logs_toggle.value) else 1.0
            vw = max(320.0, win_w * left_ratio - 80.0)
            vh = max(240.0, win_h - 320.0)
            return vw, vh

        def _place_tree_host(bbox_w: float, bbox_h: float) -> None:
            """Size the host to the pane when the drawing fits, otherwise to the drawing.

            The canvas stays at *bbox_w* by *bbox_h* and is centred in that host.
            """
            vw, vh = _viewport_px()
            tree_host.width = max(bbox_w, vw)
            tree_host.height = max(bbox_h, vh)
            tree_h_scroll.width = vw
            tree_zoom_label.value = "Zoom: 100%"

        def _schedule_tree_scroll(dx: float, dy: float) -> None:
            """Move both scroll axes so the requested offset is in view."""

            async def _run() -> None:
                try:
                    await tree_h_scroll.scroll_to(offset=dx, duration=0)
                    await tree_v_scroll.scroll_to(offset=dy, duration=0)
                except Exception as ex:
                    logger.debug("Tree scroll skipped: {}", ex)

            page.run_task(_run)

        def _paint_parse_tree_canvas() -> tuple[float, float]:
            """Draw ``session.last_tree`` at natural size and centre it when it fits the pane."""
            vw, vh = _viewport_px()
            ds_tree = session.last_tree
            tree_canvas.expand = False
            if ds_tree is None or not ds_tree:
                tree_canvas.width = vw
                tree_canvas.height = vh
                tree_canvas.shapes = [
                    cv.Rect(
                        x=0,
                        y=0,
                        width=vw,
                        height=vh,
                        paint=ft.Paint(style=ft.PaintingStyle.FILL, color=PANEL_BACKGROUND),
                    ),
                ]
                last_tree_bbox[0] = (vw, vh)
                _place_tree_host(vw, vh)
                _schedule_tree_scroll(0.0, 0.0)
                page.update()
                return vw, vh
            layout = compute_tree_layout(
                ds_tree,
                font_size=float(MONO_FONT_SIZE),
                label_density="full",
            )
            bw = float(layout.canvas_w)
            bh = float(layout.canvas_h)
            tree_canvas.width = bw
            tree_canvas.height = bh
            tree_canvas.shapes = build_canvas_shapes(
                layout,
                ds_tree.pointer,
                font_size=float(MONO_FONT_SIZE),
            )
            last_tree_bbox[0] = (bw, bh)
            _place_tree_host(bw, bh)
            _schedule_tree_scroll(0.0, 0.0)
            page.update()
            return last_tree_bbox[0]

        def apply_grammar(path_str: str) -> None:
            """Load a grammar directory into a fresh parser."""
            report = session.set_grammar(path_str, repairing=bool(repair_cb.value))
            _sync_info()
            _sync_interpretation_controls()
            set_log(report)
            if session.last_tree is not None:
                _refresh_parse_tree_visual(session.last_tree)

        def _refresh_parse_tree_visual(ds_tree: Tree) -> None:
            """Fill address-order text and paint the canvas from *ds_tree*."""
            st = session.tree_panel_state(ds_tree)
            tree_view.value = st.address_order
            _paint_parse_tree_canvas()

        def _refresh_views(msg: str | None) -> None:
            """Populate the tree / semantics / DAG fields and Info card from current parser state."""
            vs = session.current_view_strings()
            _sync_info()
            if vs is None:
                if msg:
                    append_log(msg)
                return
            tree_view.value = vs.address_order
            _paint_parse_tree_canvas()
            dag_view.value = vs.dag
            sem_view.value = vs.semantics if vs.semantics.strip() else "(no semantics yet)"
            _sync_interpretation_controls()
            if msg:
                append_log(msg)
            else:
                page.update()

        # --- event handlers ---------------------------------------------------

        async def pick_grammar_dir(_: ft.ControlEvent | None = None) -> None:
            """Open a native folder picker and load the selected grammar directory."""
            initial: str | None = None
            if session.grammar_path is not None:
                initial = str(session.grammar_path)
            elif _repo_resources.is_dir():
                initial = str(_repo_resources)
            path = await grammar_picker.get_directory_path(
                dialog_title="Select the grammar folder",
                initial_directory=initial,
            )
            if not path:
                return
            apply_grammar(str(resolve_grammar_directory(path)))

        def do_init(_: ft.ControlEvent | None = None) -> None:
            """Re-initialise the parser to the axiom state."""
            err = session.run_init()
            if err is not None:
                _sync_info()
                append_log(err)
                return
            _refresh_views(session.last_event)

        def do_new_sentence(_: ft.ControlEvent | None = None) -> None:
            """Reset the DAG for a fresh sentence."""
            err = session.run_new_sentence()
            if err is not None:
                _sync_info()
                append_log(err)
                return
            _refresh_views(session.last_event)

        def do_parse(_: ft.ControlEvent | None = None) -> None:
            """Parse the sentence in the text field, logging one line per word."""
            err, _ok, events = session.run_parse(
                sentence_field.value or "",
                reset_before=bool(reset_before.value),
                speaker=DEFAULT_SPEAKER,
            )
            if err is not None:
                _sync_info()
                append_log(err)
                return
            _refresh_views(None)
            for line in events:
                append_log(line)

        def do_select_interpretation(index: int) -> None:
            """Show interpretation *index* and append a log line when it changes."""
            err, log = session.select_interpretation(index)
            if err is not None:
                _sync_info()
                _sync_interpretation_controls()
                append_log(err)
                return
            if log is None:
                _sync_interpretation_controls()
                page.update()
                return
            _refresh_views(log)

        def do_prev_interpretation(_: ft.ControlEvent | None = None) -> None:
            """Move to the previous interpretation."""
            do_select_interpretation(session.interpretation_index - 1)

        def do_next_interpretation(_: ft.ControlEvent | None = None) -> None:
            """Move to the next interpretation."""
            do_select_interpretation(session.interpretation_index + 1)

        async def do_fit(_: ft.ControlEvent | None = None) -> None:
            """Scroll so the middle of the natural-size tree is in the pane."""
            bw, bh = last_tree_bbox[0]
            _place_tree_host(bw, bh)
            vw, vh = _viewport_px()
            hw = float(tree_host.width or bw)
            hh = float(tree_host.height or bh)
            dx = max(0.0, (hw - vw) * 0.5)
            dy = max(0.0, (hh - vh) * 0.5)
            try:
                await tree_h_scroll.scroll_to(offset=dx, duration=0)
                await tree_v_scroll.scroll_to(offset=dy, duration=0)
            except Exception as ex:
                logger.debug("Tree scroll reset skipped: {}", ex)
            page.update()

        fit_btn.on_click = do_fit

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
        interp_readout = ft.Text(
            format_interpretation_readout(0, 0, capped=False),
            size=12,
            color=BODY_TEXT_COLOR,
        )
        interp_box = ft.Container(
            content=interp_readout,
            border=ft.Border.all(1, BOX_BORDER_COLOR),
            border_radius=4,
            padding=ft.Padding.symmetric(horizontal=12, vertical=8),
            bgcolor=BOX_BACKGROUND_COLOR,
        )
        _arrow_style = ft.ButtonStyle(padding=ft.Padding.all(2))
        prev_interp_btn = ft.IconButton(
            icon=ft.Icons.CHEVRON_LEFT,
            icon_color=BODY_TEXT_COLOR,
            icon_size=20,
            tooltip="Previous interpretation",
            disabled=True,
            style=_arrow_style,
            on_click=do_prev_interpretation,
        )
        next_interp_btn = ft.IconButton(
            icon=ft.Icons.CHEVRON_RIGHT,
            icon_color=BODY_TEXT_COLOR,
            icon_size=20,
            tooltip="Next interpretation",
            disabled=True,
            style=_arrow_style,
            on_click=do_next_interpretation,
        )
        parse_cluster = ft.Column(
            [
                parse_btn,
                ft.Row(
                    [prev_interp_btn, interp_box, next_interp_btn],
                    spacing=2,
                    tight=True,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ],
            spacing=6,
            tight=True,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
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
            tooltip="Reset the parser to the axiom DS Tree (clears the current derivation).",
        )
        new_sentence_btn = ft.Button(
            content="New sentence",
            on_click=do_new_sentence,
            tooltip="Start a new sentence: reset the DAG to the axiom without unloading the grammar.",
        )
        grammar_toolbar = ft.Column(
            [
                ft.Row(
                    [
                        set_grammar_btn,
                        show_logs_toggle,
                        ft.Container(expand=True),
                        init_btn,
                        new_sentence_btn,
                    ],
                    spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Row(
                    [
                        repair_cb,
                        reset_before,
                    ],
                    spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ],
            spacing=4,
        )
        sentence_row = ft.Container(
            content=ft.Row(
                [
                    ft.Container(content=sentence_box, expand=True),
                    parse_cluster,
                ],
                spacing=28,
                vertical_alignment=ft.CrossAxisAlignment.START,
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
            """Re-centre the natural-size tree when the window size changes."""
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
