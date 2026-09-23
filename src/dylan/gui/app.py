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
from dylan.gui.tree_viz import (
    DrawingPlacement,
    anchor_viewport_position,
    build_canvas_shapes,
    compute_tree_layout,
    format_zoom_percent,
    place_drawing_in_viewport,
    scale_tree_layout,
    step_zoom,
    theme_for_zoom,
    zoom_action_for_key,
)
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
            'or: uv pip install -e ".[gui]"\n'
            'PyPI installs: pip install "dynamicsyntax[gui]"'
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
            tooltip="Ctrl+Plus and Ctrl+Minus zoom. Ctrl+0 returns to 100%. The root node stays put.",
        )
        fit_btn = ft.Button(
            content="Fit",
            tooltip="Scroll so the middle of the tree is in this pane. Does not change zoom.",
        )
        tree_canvas = cv.Canvas(
            shapes=[],
            width=400,
            height=320,
            expand=False,
        )
        tree_canvas_slot = ft.Container(
            content=tree_canvas,
            left=0,
            top=0,
            width=400,
            height=320,
        )
        tree_host = ft.Stack(
            controls=[tree_canvas_slot],
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
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
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

        tree_zoom: list[float] = [1.0]
        scroll_xy: list[float] = [0.0, 0.0]
        last_placement: list[DrawingPlacement | None] = [None]
        last_anchor: list[tuple[float, float] | None] = [None]
        measured_viewport: list[tuple[float, float]] = [(0.0, 0.0)]
        centering_guard: list[bool] = [False]

        def _on_h_scroll(e: ft.OnScrollEvent) -> None:
            """Remember the horizontal Camera offset."""
            scroll_xy[0] = float(getattr(e, "pixels", 0.0) or 0.0)

        def _on_v_scroll(e: ft.OnScrollEvent) -> None:
            """Remember the vertical Camera offset."""
            scroll_xy[1] = float(getattr(e, "pixels", 0.0) or 0.0)

        tree_h_scroll.on_scroll = _on_h_scroll
        tree_v_scroll.on_scroll = _on_v_scroll

        def on_tree_stage_size(e: ft.LayoutSizeChangeEvent) -> None:
            """Remember the Output pane size and keep the root node where it is."""
            w = float(getattr(e, "width", 0.0) or 0.0)
            h = float(getattr(e, "height", 0.0) or 0.0)
            if w < 32.0 or h < 32.0:
                return
            prev_w, prev_h = measured_viewport[0]
            measured_viewport[0] = (w, h)
            if abs(prev_w - w) < 1.0 and abs(prev_h - h) < 1.0:
                return
            if centering_guard[0]:
                return
            centering_guard[0] = True
            try:
                hold = _live_root_hold() if session.last_tree is not None else None
                _paint_parse_tree_canvas(hold=hold)
            except Exception as ex:
                logger.debug("Tree resize skipped: {}", ex)
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

        def _reset_tree_camera() -> None:
            """Return Zoom to 100% and drop the root hold so the next paint uses the default Camera."""
            tree_zoom[0] = 1.0
            last_placement[0] = None
            last_anchor[0] = None

        def _live_root_hold() -> tuple[float, float] | None:
            """Return the root node's current viewport position, if a drawing is on screen."""
            placement = last_placement[0]
            anchor = last_anchor[0]
            if placement is None or anchor is None:
                return None
            return anchor_viewport_position(placement, anchor, scroll_xy[0], scroll_xy[1])

        def _apply_placement(placement: DrawingPlacement, canvas_w: float, canvas_h: float) -> None:
            """Size the scroll host and pin the canvas at *placement*'s offset."""
            vw, _vh = _viewport_px()
            tree_canvas.width = canvas_w
            tree_canvas.height = canvas_h
            tree_canvas_slot.left = placement.canvas_x
            tree_canvas_slot.top = placement.canvas_y
            tree_canvas_slot.width = canvas_w
            tree_canvas_slot.height = canvas_h
            tree_host.width = placement.host_w
            tree_host.height = placement.host_h
            tree_h_scroll.width = vw
            tree_zoom_label.value = format_zoom_percent(tree_zoom[0])
            last_placement[0] = placement

        def _schedule_tree_scroll(dx: float, dy: float) -> None:
            """Move both scroll axes so the requested offset is in view."""
            scroll_xy[0] = dx
            scroll_xy[1] = dy

            async def _run() -> None:
                try:
                    await tree_h_scroll.scroll_to(offset=dx, duration=0)
                    await tree_v_scroll.scroll_to(offset=dy, duration=0)
                except Exception as ex:
                    logger.debug("Tree scroll skipped: {}", ex)

            page.run_task(_run)

        def _render_tree(
            *,
            hold: tuple[float, float] | None,
            fit_scroll: bool = False,
        ) -> tuple[float, float]:
            """Draw ``session.last_tree`` at the current Zoom and return the scroll offset.

            *hold* keeps the root node at that viewport position. Without it, a drawing
            that fits is centred. *fit_scroll* then scrolls so the middle of that
            drawing is in the pane. Zoom is unchanged.
            """
            vw, vh = _viewport_px()
            ds_tree = session.last_tree
            tree_canvas.expand = False
            zoom = tree_zoom[0]
            if ds_tree is None or not ds_tree:
                tree_canvas.shapes = [
                    cv.Rect(
                        x=0,
                        y=0,
                        width=vw,
                        height=vh,
                        paint=ft.Paint(style=ft.PaintingStyle.FILL, color=PANEL_BACKGROUND),
                    ),
                ]
                placement = place_drawing_in_viewport(vw, vh, vw, vh)
                _apply_placement(placement, vw, vh)
                last_anchor[0] = None
                return 0.0, 0.0
            natural = compute_tree_layout(
                ds_tree,
                font_size=float(MONO_FONT_SIZE),
                label_density="full",
            )
            layout = scale_tree_layout(natural, zoom)
            anchor: tuple[float, float] | None = None
            for node in layout.nodes:
                if node.addr == ds_tree.root_addr:
                    anchor = (node.cx, node.cy)
                    break
            last_anchor[0] = anchor
            tree_canvas.shapes = build_canvas_shapes(
                layout,
                ds_tree.pointer,
                theme_for_zoom(zoom),
                font_size=float(MONO_FONT_SIZE) * zoom,
                text_padding=4.0 * zoom,
            )
            if hold is not None and anchor is not None:
                placement = place_drawing_in_viewport(
                    layout.canvas_w,
                    layout.canvas_h,
                    vw,
                    vh,
                    anchor=anchor,
                    hold=hold,
                )
                scroll = (placement.scroll_x, placement.scroll_y)
            else:
                placement = place_drawing_in_viewport(
                    layout.canvas_w,
                    layout.canvas_h,
                    vw,
                    vh,
                )
                if fit_scroll:
                    scroll = (
                        max(0.0, (placement.host_w - vw) * 0.5),
                        max(0.0, (placement.host_h - vh) * 0.5),
                    )
                else:
                    scroll = (0.0, 0.0)
            _apply_placement(placement, float(layout.canvas_w), float(layout.canvas_h))
            return scroll

        def _paint_parse_tree_canvas(
            *,
            hold: tuple[float, float] | None = None,
            fit_scroll: bool = False,
        ) -> None:
            """Paint the DS Tree and scroll to the Camera offset for this paint."""
            dx, dy = _render_tree(hold=hold, fit_scroll=fit_scroll)
            scroll_xy[0] = dx
            scroll_xy[1] = dy
            page.update()
            _schedule_tree_scroll(dx, dy)

        def apply_grammar(path_str: str) -> None:
            """Load a grammar directory into a fresh parser."""
            report = session.set_grammar(path_str, repairing=bool(repair_cb.value))
            _sync_info()
            _sync_interpretation_controls()
            set_log(report)
            if session.last_tree is not None:
                _refresh_parse_tree_visual(session.last_tree, reset_zoom=True)

        def _refresh_parse_tree_visual(ds_tree: Tree, *, reset_zoom: bool = False) -> None:
            """Fill address-order text and paint the canvas from *ds_tree*.

            *reset_zoom* returns to 100% and the default Camera. Otherwise the
            current Zoom is kept and the root node stays where it is.
            """
            st = session.tree_panel_state(ds_tree)
            tree_view.value = st.address_order
            if reset_zoom:
                _reset_tree_camera()
            hold = None if reset_zoom else _live_root_hold()
            _paint_parse_tree_canvas(hold=hold)

        def _refresh_views(msg: str | None) -> None:
            """Populate the tree / semantics / DAG fields and Info card from current parser state."""
            vs = session.current_view_strings()
            _sync_info()
            if vs is None:
                if msg:
                    append_log(msg)
                return
            _reset_tree_camera()
            tree_view.value = vs.address_order
            _paint_parse_tree_canvas(hold=None)
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
            """Show interpretation *index* without adding a Logs line."""
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
            _refresh_views(None)

        def do_prev_interpretation(_: ft.ControlEvent | None = None) -> None:
            """Move to the previous interpretation."""
            do_select_interpretation(session.interpretation_index - 1)

        def do_next_interpretation(_: ft.ControlEvent | None = None) -> None:
            """Move to the next interpretation."""
            do_select_interpretation(session.interpretation_index + 1)

        async def do_fit(_: ft.ControlEvent | None = None) -> None:
            """Scroll so the middle of the tree at the current Zoom is in the pane."""
            dx, dy = _render_tree(hold=None, fit_scroll=True)
            scroll_xy[0] = dx
            scroll_xy[1] = dy
            page.update()
            try:
                await tree_h_scroll.scroll_to(offset=dx, duration=0)
                await tree_v_scroll.scroll_to(offset=dy, duration=0)
            except Exception as ex:
                logger.debug("Tree scroll reset skipped: {}", ex)

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
            """Keep Zoom and the root node's place when the window size changes."""
            if session.parser is None or session.last_tree is None:
                return
            _refresh_parse_tree_visual(session.last_tree)

        def on_keyboard(e: ft.KeyboardEvent) -> None:
            """Zoom when Ctrl+Plus, Ctrl+Minus, or Ctrl+0 is pressed on the Parse tree tab."""
            if not e.ctrl or e.alt or e.meta:
                return
            if int(tabs_widget.selected_index or 0) != 0:
                return
            action = zoom_action_for_key(e.key)
            if action is None or not session.last_tree:
                return
            if action == "in":
                new_zoom = step_zoom(tree_zoom[0], 1)
            elif action == "out":
                new_zoom = step_zoom(tree_zoom[0], -1)
            else:
                new_zoom = 1.0
            if abs(new_zoom - tree_zoom[0]) < 1e-9:
                return
            hold = _live_root_hold()
            tree_zoom[0] = new_zoom
            _paint_parse_tree_canvas(hold=hold)

        page.on_keyboard_event = on_keyboard

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
