"""Flet desktop GUI for loading grammars and parsing (Java ``ParserGUI`` / ``ParserPanel`` subset).

Targets Flet >= 0.80 (async ``FilePicker`` methods, ``Tabs`` uses
``content`` + ``length``, four main tabs: DS Tree, address list, semantics, DAG).
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

from dylan.gui.paint import (
    BODY_FONT_FAMILY,
    DEFAULT_PAINT,
    PAINTS,
    PAINT_BY_KEY,
    GuiPaint,
)
from dylan.gui.parse_session import (
    FLET_INFO_HELP_LINES,
    ParseSession,
    SessionStatus,
    StatusField,
    format_event_log,
    format_interpretation_readout,
    is_action_log_line,
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

        # --- coat of paint (colours, type, radius). Layout stays put. ----------------
        active: list[GuiPaint] = [DEFAULT_PAINT]
        caption_chrome: list[tuple[ft.Container, ft.Container, ft.Text]] = []

        def coat() -> GuiPaint:
            """Return the coat currently painted on the window."""
            return active[0]

        def _filled_button_style(paint: GuiPaint) -> ft.ButtonStyle:
            """Filled style for Load grammar and Parse."""
            return ft.ButtonStyle(
                bgcolor=paint.commit,
                color=paint.commit_text,
                icon_color=paint.commit_text,
                padding=_primary_btn_padding,
                shape=ft.RoundedRectangleBorder(radius=paint.radius),
            )

        def _outline_button_style(paint: GuiPaint) -> ft.ButtonStyle:
            """Outline style for Reset and Fit."""
            return ft.ButtonStyle(
                bgcolor=ft.Colors.TRANSPARENT,
                color=paint.signal,
                icon_color=paint.signal,
                side=ft.BorderSide(width=1, color=paint.rule),
                shape=ft.RoundedRectangleBorder(radius=paint.radius),
            )

        BODY_FONT_SIZE = 12
        MONO_FONT_SIZE = 13
        CAPTION_FONT_SIZE = 12
        MONO_FONT_FAMILY = "Consolas"
        LEFT_COLUMN_EXPAND = 80
        LOGS_COLUMN_EXPAND = 20

        def _mono_style(color: str, *, weight: ft.FontWeight | None = None) -> ft.TextStyle:
            """Consolas style at the log size, in *color*."""
            return ft.TextStyle(
                font_family=MONO_FONT_FAMILY,
                size=MONO_FONT_SIZE,
                color=color,
                weight=weight,
            )

        mono_text_style = _mono_style(coat().ink)
        field_label_style = ft.TextStyle(color=coat().quiet)
        hint_text_style = ft.TextStyle(color=coat().hint)
        toolbar_label_style = ft.TextStyle(color=coat().ink, font_family=BODY_FONT_FAMILY)

        _outline = ft.InputBorder.OUTLINE
        _primary_btn_padding = ft.Padding.symmetric(horizontal=16, vertical=12)
        page.bgcolor = coat().ground
        page.theme_mode = ft.ThemeMode.DARK

        def _dark_outlined_textfield(**kwargs: Any) -> ft.TextField:
            """TextField with the active coat's filled outline; callers pass field-specific kwargs."""
            stage = coat().stage
            defaults: dict[str, Any] = {
                "border": _outline,
                "filled": True,
                "fill_color": stage,
                "hover_color": stage,
                "focused_bgcolor": stage,
                "border_color": coat().rule,
                "focused_border_color": coat().signal,
                "color": coat().ink,
                "cursor_color": coat().ink,
                "text_style": _mono_style(coat().ink),
                "label_style": ft.TextStyle(color=coat().quiet, font_family=BODY_FONT_FAMILY),
            }
            defaults.update(kwargs)
            return ft.TextField(**defaults)

        def _dark_borderless_textfield(**kwargs: Any) -> ft.TextField:
            """TextField with the active coat's fill and no inner outline (captioned boxes)."""
            stage = coat().stage
            defaults: dict[str, Any] = {
                "border": ft.InputBorder.NONE,
                "filled": True,
                "fill_color": stage,
                "hover_color": stage,
                "focused_bgcolor": stage,
                "color": coat().ink,
                "cursor_color": coat().ink,
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
            content_height: int | None = None,
        ) -> ft.Container:
            """Outlined box with *title* on the top border.

            Use *fill_vertical* only inside a bounded flex area (tabs, logs).
            *content_height* fixes the border height for a loose stack so sibling
            cards stay the same height without an expanding stack.
            """
            bc = border_color if border_color is not None else coat().rule
            tc = title_color if title_color is not None else coat().quiet
            pad = content_padding if content_padding is not None else ft.Padding.all(10)
            caption_text = ft.Text(
                title,
                size=CAPTION_FONT_SIZE,
                color=tc,
                font_family=coat().caption_font(),
            )
            # Row + Column gives unbounded max height to non-flex children; an expanding Stack
            # then claims infinite height and hides siblings. Use loose Stack + intrinsic height
            # unless *fill_vertical* (tabs / logs) where the parent supplies a bounded flex area.
            if fill_vertical:
                bordered = ft.Container(
                    margin=ft.Margin.only(top=8),
                    expand=True,
                    border=ft.Border.all(width=1, color=bc),
                    border_radius=coat().radius,
                    bgcolor=fill_color,
                    padding=pad,
                    content=content,
                )
                chip = ft.Container(
                    left=12,
                    top=0,
                    bgcolor=caption_bg,
                    padding=ft.Padding.symmetric(horizontal=4),
                    content=caption_text,
                )
                stack = ft.Stack(
                    fit=ft.StackFit.EXPAND,
                    expand=True,
                    controls=[
                        bordered,
                        chip,
                    ],
                )
            else:
                bordered = ft.Container(
                    margin=ft.Margin.only(top=8),
                    height=content_height,
                    border=ft.Border.all(width=1, color=bc),
                    border_radius=coat().radius,
                    bgcolor=fill_color,
                    padding=pad,
                    content=content,
                )
                chip = ft.Container(
                    left=12,
                    top=0,
                    bgcolor=caption_bg,
                    padding=ft.Padding.symmetric(horizontal=4),
                    content=caption_text,
                )
                stack = ft.Stack(
                    fit=ft.StackFit.LOOSE,
                    controls=[
                        bordered,
                        chip,
                    ],
                )
            caption_chrome.append((bordered, chip, caption_text))
            return ft.Container(expand=expand, content=stack)

        grammar_icon = ft.Icon(ft.Icons.FOLDER_OPEN, size=20, color=coat().commit_text)
        grammar_label = ft.Text("Load grammar", color=coat().commit_text)
        set_grammar_btn = ft.FilledButton(
            content=ft.Row(
                [
                    grammar_icon,
                    grammar_label,
                ],
                tight=True,
                spacing=8,
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            tooltip="Open a folder dialog and select the grammar directory (all files in that folder are loaded).",
            style=_filled_button_style(coat()),
        )
        sentence_field = _dark_borderless_textfield(
            hint_text="Enter your sentence here...",
            multiline=True,
            min_lines=3,
            max_lines=6,
            expand=True,
            color=coat().ink,
            cursor_color=coat().ink,
            hint_style=hint_text_style,
        )
        sentence_box = _border_caption_box(
            "Sentence",
            sentence_field,
            caption_bg=coat().ground,
            fill_color=coat().stage,
        )

        def _status_value_text() -> ft.Text:
            """One Status value, updated in place when the session changes."""
            return ft.Text(
                "—",
                size=BODY_FONT_SIZE,
                color=coat().quiet,
                expand=True,
                selectable=True,
            )

        status_values: dict[str, ft.Text] = {
            "grammar": _status_value_text(),
            "repair": _status_value_text(),
            "last": _status_value_text(),
            "warnings": _status_value_text(),
            "pointer": _status_value_text(),
            "dag_tuple": _status_value_text(),
        }

        status_labels: list[ft.Text] = []

        def _status_row(key: str, label: str) -> ft.Row:
            """A quiet label beside a coloured Status value."""
            label_text = ft.Text(
                label,
                width=78,
                size=11,
                color=coat().quiet,
                weight=ft.FontWeight.W_600,
                font_family=coat().caption_font(),
            )
            status_labels.append(label_text)
            return ft.Row(
                [
                    label_text,
                    status_values[key],
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.START,
            )

        status_column = ft.Column(
            [
                _status_row("grammar", "Grammar"),
                _status_row("repair", "Repair"),
                _status_row("last", "Last"),
                _status_row("warnings", "Load warnings"),
                _status_row("pointer", "Pointer"),
                _status_row("dag_tuple", "DAG tuple"),
            ],
            spacing=4,
            tight=True,
        )
        status_box = _border_caption_box(
            "Status",
            status_column,
            caption_bg=coat().ground,
            fill_color=coat().stage,
        )

        def _tone_color(tone: str) -> str:
            """Map a Status tone onto the active coat. Meanings stay fixed."""
            paint = coat()
            colors = {
                "muted": paint.quiet,
                "amber": paint.warning,
                "ok": paint.ok,
                "error": paint.error,
                "body": paint.ink,
                "mono": paint.ink,
            }
            return colors[tone]

        log_blocks: list[str] = []
        log_scroll = ft.Column(
            controls=[],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=10,
        )
        log_box = _border_caption_box(
            "Logs",
            log_scroll,
            caption_bg=coat().ground,
            fill_color=coat().stage,
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
            color=coat().quiet,
            tooltip="Ctrl+Plus and Ctrl+Minus zoom. Ctrl+0 returns to 100%. The root node stays put.",
        )
        fit_btn = ft.Button(
            content="Fit",
            style=_outline_button_style(coat()),
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
            caption_bg=coat().ground,
            fill_color=coat().stage,
            expand=True,
            fill_vertical=True,
        )
        address_order_box = _border_caption_box(
            "Address order",
            tree_view,
            caption_bg=coat().ground,
            fill_color=coat().stage,
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
        repair_cb = ft.Checkbox(
            label="Repair",
            value=False,
            tooltip="Repair path is only partially ported; leave off unless you are testing repairs.",
            label_style=toolbar_label_style,
            visual_density=ft.VisualDensity.COMPACT,
        )
        show_logs_toggle = ft.Checkbox(
            label="Logs",
            value=True,
            label_style=ft.TextStyle(color=coat().ink, size=11),
            tooltip="Show or hide Status and Logs (right panel)",
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

        def _apply_status_field(key: str, field: StatusField) -> None:
            """Paint one Status value in the colour for its tone."""
            text = status_values[key]
            text.value = field.value
            text.color = _tone_color(field.tone)
            text.font_family = MONO_FONT_FAMILY if field.tone == "mono" else BODY_FONT_FAMILY
            text.weight = (
                ft.FontWeight.W_600
                if field.tone in {"ok", "error", "amber"}
                else ft.FontWeight.W_400
            )

        def _sync_status() -> None:
            """Refresh the Status card from the live session. Info stays the how-to."""
            status: SessionStatus = session.session_status()
            _apply_status_field("grammar", status.grammar)
            _apply_status_field("repair", status.repair)
            _apply_status_field("last", status.last)
            _apply_status_field("warnings", status.warnings)
            _apply_status_field("pointer", status.pointer)
            _apply_status_field("dag_tuple", status.dag_tuple)

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

            Failure lines use the coat's error red. On a successful word, only
            the word ``parsed`` uses the coat's ok green.
            """
            paint = coat()
            mono = _mono_style(paint.ink)
            is_need_grammar = "load a grammar first" in block.lower()
            is_word_fail = " failed" in block.lower() or block.lower().endswith("failed")
            if is_need_grammar or is_word_fail:
                return ft.Text(
                    value=block,
                    style=_mono_style(paint.error),
                    selectable=True,
                )
            if "\n" not in block and block.endswith(" parsed"):
                token = block[: -len(" parsed")]
                return ft.Text(
                    spans=[
                        ft.TextSpan(text=f"{token} ", style=mono),
                        ft.TextSpan(
                            text="parsed",
                            style=_mono_style(paint.ok, weight=ft.FontWeight.W_600),
                        ),
                    ],
                    selectable=True,
                )
            if "\n" not in block and is_action_log_line(block):
                action_style = _mono_style(paint.signal)
                if block.startswith("(interp ") and ") " in block:
                    prefix, rest = block.split(") ", 1)
                    return ft.Text(
                        spans=[
                            ft.TextSpan(
                                text=f"{prefix}) ",
                                style=_mono_style(paint.interpretation, weight=ft.FontWeight.W_700),
                            ),
                            ft.TextSpan(text=rest, style=action_style),
                        ],
                        selectable=True,
                    )
                return ft.Text(value=block, style=action_style, selectable=True)
            return ft.Text(value=block, style=mono, selectable=True)

        def _rebuild_log_controls() -> None:
            """Recreate log lines so their colours follow the active coat."""
            log_scroll.controls = [_log_block_control(block) for block in log_blocks]

        def set_log(text: str) -> None:
            """Replace the log panel content, mirror it to stderr, and refresh the page."""
            block = format_event_log(text) if text else ""
            log_blocks.clear()
            if block:
                log_blocks.append(block)
            _rebuild_log_controls()
            print(text, file=sys.stderr, flush=True)
            page.update()

        def append_log(text: str) -> None:
            """Append a block to the log panel, mirror that block to stderr, and refresh."""
            block = format_event_log(text) if text else ""
            if block:
                log_blocks.append(block)
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
            left_ratio = 0.80 if bool(show_logs_toggle.value) else 1.0
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
                        paint=ft.Paint(style=ft.PaintingStyle.FILL, color=coat().stage),
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
                theme_for_zoom(zoom, coat().canvas_theme()),
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
            _sync_status()
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
            """Refresh the tree, semantics, DAG, and Status from the current parser."""
            vs = session.current_view_strings()
            _sync_status()
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

        def do_reset(_: ft.ControlEvent | None = None) -> None:
            """Clear the derivation back to the empty axiom. The grammar stays loaded."""
            err = session.run_init(success_event="Reset — axiom state.")
            if err is not None:
                _sync_status()
                append_log(err)
                return
            _refresh_views(session.last_event)

        def do_parse(_: ft.ControlEvent | None = None) -> None:
            """Parse the sentence, logging only words that are new to the derivation."""
            err, _ok, events = session.run_parse(
                sentence_field.value or "",
                reset_before=False,
                speaker=DEFAULT_SPEAKER,
            )
            if err is not None:
                _sync_status()
                append_log(err)
                return
            _refresh_views(None)
            for line in events:
                append_log(line)

        def do_select_interpretation(index: int) -> None:
            """Show interpretation *index* without adding a Logs line."""
            err, log = session.select_interpretation(index)
            if err is not None:
                _sync_status()
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
            style=_filled_button_style(coat()),
        )
        interp_readout = ft.Text(
            format_interpretation_readout(0, 0, capped=False),
            size=12,
            color=coat().ink,
            font_family=BODY_FONT_FAMILY,
        )
        interp_box = ft.Container(
            content=interp_readout,
            border=ft.Border.all(1, coat().rule),
            border_radius=coat().radius,
            padding=ft.Padding.symmetric(horizontal=12, vertical=8),
            bgcolor=coat().stage,
        )
        _arrow_style = ft.ButtonStyle(padding=ft.Padding.all(2))
        prev_interp_btn = ft.IconButton(
            icon=ft.Icons.CHEVRON_LEFT,
            icon_color=coat().ink,
            icon_size=20,
            tooltip="Previous interpretation",
            disabled=True,
            style=_arrow_style,
            on_click=do_prev_interpretation,
        )
        next_interp_btn = ft.IconButton(
            icon=ft.Icons.CHEVRON_RIGHT,
            icon_color=coat().ink,
            icon_size=20,
            tooltip="Next interpretation",
            disabled=True,
            style=_arrow_style,
            on_click=do_next_interpretation,
        )
        set_grammar_btn.on_click = pick_grammar_dir

        # --- layout -----------------------------------------------------------

        tab_bar = ft.TabBar(
            scrollable=False,
            tab_alignment=ft.TabAlignment.FILL,
            indicator_color=coat().signal,
            divider_color=coat().rule,
            label_color=coat().ink,
            unselected_label_color=coat().quiet,
            tabs=[
                ft.Tab(label="DS Tree", icon=ft.Icons.ACCOUNT_TREE),
                ft.Tab(
                    label="Address order",
                    icon=ft.Icons.FORMAT_LIST_BULLETED,
                ),
                ft.Tab(label="Semantics", icon=ft.Icons.DATA_OBJECT),
                ft.Tab(label="DAG", icon=ft.Icons.HUB),
            ],
        )
        tabs_widget = ft.Tabs(
            length=4,
            selected_index=0,
            expand=True,
            content=ft.Column(
                expand=True,
                controls=[
                    tab_bar,
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

        reset_btn = ft.Button(
            content="Reset",
            on_click=do_reset,
            style=_outline_button_style(coat()),
            tooltip="Clear the derivation and return the tree to the empty axiom. The grammar stays loaded.",
        )
        # New sentence is hidden for now. ParseSession.run_new_sentence remains for the browser.
        title_text = ft.Text(
            "DyLan - The Dynamic Syntax Parser",
            size=26,
            weight=ft.FontWeight.W_600,
            font_family=coat().title_font(),
            color=coat().ink,
            text_align=ft.TextAlign.CENTER,
        )
        title_box = ft.Container(
            content=title_text,
            bgcolor=coat().stage,
            border=ft.Border.all(width=1, color=coat().rule),
            border_radius=coat().radius,
            padding=ft.Padding.symmetric(horizontal=22, vertical=12),
        )
        paint_dropdown = ft.Dropdown(
            label="Coat",
            value=coat().key,
            width=180,
            dense=True,
            text_size=12,
            filled=True,
            fill_color=coat().stage,
            bgcolor=coat().stage,
            color=coat().ink,
            border_color=coat().rule,
            border_radius=coat().radius,
            label_style=ft.TextStyle(color=coat().quiet, size=12, font_family=BODY_FONT_FAMILY),
            text_style=ft.TextStyle(color=coat().ink, size=12, font_family=BODY_FONT_FAMILY),
            options=[ft.DropdownOption(key=item.key, text=item.label) for item in PAINTS],
            tooltip="Switch the window's coat of paint. The parser layout stays the same.",
        )
        title_row = ft.Row(
            [
                ft.Container(expand=True),
                title_box,
                ft.Container(
                    content=paint_dropdown,
                    expand=True,
                    alignment=ft.Alignment.CENTER_RIGHT,
                ),
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        def show_help(_: ft.ControlEvent) -> None:
            """Open the how-to. A click outside the card closes it."""

            def close_help(event: ft.ControlEvent) -> None:
                """Remove the how-to when the click lands on the scrim."""
                if event.control is not scrim:
                    return
                if scrim in page.overlay:
                    page.overlay.remove(scrim)
                page.update()

            def keep_open(_: ft.ControlEvent) -> None:
                """Leave the how-to open when the click lands on the card."""
                return

            card = ft.Container(
                width=460,
                bgcolor=coat().stage,
                border=ft.Border.all(width=1, color=coat().rule),
                border_radius=coat().radius,
                padding=16,
                on_click=keep_open,
                content=ft.Column(
                    [
                        ft.Text(
                            "Info",
                            color=coat().ink,
                            size=16,
                            weight=ft.FontWeight.W_600,
                            font_family=coat().title_font(),
                        ),
                        *[
                            ft.Text(
                                line,
                                color=coat().ink,
                                size=BODY_FONT_SIZE,
                                font_family=BODY_FONT_FAMILY,
                            )
                            for line in FLET_INFO_HELP_LINES
                        ],
                    ],
                    tight=True,
                    spacing=8,
                    scroll=ft.ScrollMode.AUTO,
                ),
            )
            scrim = ft.Container(
                expand=True,
                width=page.width or page.window.width,
                height=page.height or page.window.height,
                bgcolor=ft.Colors.with_opacity(0.35, "black"),
                alignment=ft.Alignment.CENTER,
                on_click=close_help,
                content=card,
            )
            page.overlay.append(scrim)
            page.update()

        help_mark = ft.Text("?", size=18, weight=ft.FontWeight.W_600, color=coat().ink)
        help_btn = ft.Container(
            content=help_mark,
            width=36,
            height=36,
            alignment=ft.Alignment.CENTER,
            bgcolor=coat().stage,
            border=ft.Border.all(width=1, color=coat().rule),
            border_radius=coat().radius,
            on_click=show_help,
            tooltip="How to load a grammar, parse, zoom, and reset",
        )
        action_row = ft.Row(
            [
                set_grammar_btn,
                parse_btn,
                ft.Row(
                    [prev_interp_btn, interp_box, next_interp_btn],
                    spacing=2,
                    tight=True,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                reset_btn,
                ft.Container(expand=True),
                show_logs_toggle,
                repair_cb,
                ft.Container(expand=True),
                help_btn,
            ],
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        left_column = ft.Column(
            [
                title_row,
                action_row,
                ft.Container(content=sentence_box, padding=ft.Padding.only(top=8)),
                tabs_widget,
            ],
            expand=True,
            spacing=4,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        logs_panel = ft.Container(
            content=ft.Column(
                [
                    status_box,
                    ft.Container(content=log_box, expand=True),
                ],
                expand=True,
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
            bgcolor=coat().ground,
            padding=8,
            border_radius=coat().radius,
            expand=LOGS_COLUMN_EXPAND,
        )
        left_wrap = ft.Container(
            content=left_column,
            expand=LEFT_COLUMN_EXPAND,
            padding=ft.Padding.only(right=8),
            bgcolor=coat().ground,
        )

        def on_show_logs_change(e: ft.ControlEvent) -> None:
            """Show or hide Status and Logs. When hidden, the parse UI uses the full width."""
            show = bool(e.control.value)
            logs_panel.visible = show
            left_wrap.expand = LEFT_COLUMN_EXPAND if show else True
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
            """Zoom when Ctrl+Plus, Ctrl+Minus, or Ctrl+0 is pressed on the DS Tree tab."""
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

        def _style_text_field(field: ft.TextField, *, mono: bool, outlined: bool) -> None:
            """Repaint one text field from the active coat."""
            paint = coat()
            stage = paint.stage
            field.color = paint.ink
            field.cursor_color = paint.ink
            field.fill_color = stage
            field.hover_color = stage
            field.focused_bgcolor = stage
            if mono:
                field.text_style = _mono_style(paint.ink)
            else:
                field.text_style = ft.TextStyle(
                    font_family=BODY_FONT_FAMILY,
                    size=BODY_FONT_SIZE,
                    color=paint.ink,
                )
            field.hint_style = ft.TextStyle(
                color=paint.hint,
                size=BODY_FONT_SIZE,
                font_family=BODY_FONT_FAMILY,
            )
            field.label_style = ft.TextStyle(
                color=paint.quiet,
                size=12,
                font_family=BODY_FONT_FAMILY,
            )
            if outlined:
                field.border_color = paint.rule
                field.focused_border_color = paint.signal

        def _apply_chrome() -> None:
            """Repaint every chrome control from the active coat. Layout stays put."""
            paint = coat()
            theme = ft.Theme(font_family=BODY_FONT_FAMILY, color_scheme_seed=paint.signal)
            page.theme = theme
            page.dark_theme = theme
            page.theme_mode = ft.ThemeMode.DARK
            page.bgcolor = paint.ground
            left_wrap.bgcolor = paint.ground
            logs_panel.bgcolor = paint.ground
            logs_panel.border_radius = paint.radius
            title_text.font_family = paint.title_font()
            title_text.color = paint.ink
            title_box.bgcolor = paint.stage
            title_box.border = ft.Border.all(width=1, color=paint.rule)
            title_box.border_radius = paint.radius
            help_mark.color = paint.ink
            help_btn.bgcolor = paint.stage
            help_btn.border = ft.Border.all(width=1, color=paint.rule)
            help_btn.border_radius = paint.radius
            caption_font = paint.caption_font()
            for frame, chip, label in caption_chrome:
                frame.bgcolor = paint.stage
                frame.border = ft.Border.all(width=1, color=paint.rule)
                frame.border_radius = paint.radius
                chip.bgcolor = paint.ground
                label.color = paint.quiet
                label.font_family = caption_font
            for label in status_labels:
                label.color = paint.quiet
                label.font_family = caption_font
            _style_text_field(sentence_field, mono=False, outlined=False)
            _style_text_field(tree_view, mono=True, outlined=False)
            _style_text_field(sem_view, mono=True, outlined=True)
            _style_text_field(dag_view, mono=True, outlined=True)
            tree_zoom_label.color = paint.quiet
            tree_zoom_label.font_family = BODY_FONT_FAMILY
            interp_readout.color = paint.ink
            interp_box.bgcolor = paint.stage
            interp_box.border = ft.Border.all(1, paint.rule)
            interp_box.border_radius = paint.radius
            prev_interp_btn.icon_color = paint.ink
            next_interp_btn.icon_color = paint.ink
            filled = _filled_button_style(paint)
            outline = _outline_button_style(paint)
            set_grammar_btn.style = filled
            parse_btn.style = filled
            grammar_icon.color = paint.commit_text
            grammar_label.color = paint.commit_text
            reset_btn.style = outline
            fit_btn.style = outline
            checkbox_ink = ft.TextStyle(color=paint.ink, font_family=BODY_FONT_FAMILY)
            repair_cb.label_style = checkbox_ink
            show_logs_toggle.label_style = ft.TextStyle(
                color=paint.ink,
                size=11,
                font_family=BODY_FONT_FAMILY,
            )
            for checkbox in (repair_cb, show_logs_toggle):
                checkbox.fill_color = paint.ground
                checkbox.active_color = paint.signal
                checkbox.check_color = paint.signal
            tab_bar.indicator_color = paint.signal
            tab_bar.divider_color = paint.rule
            tab_bar.label_color = paint.ink
            tab_bar.unselected_label_color = paint.quiet
            paint_dropdown.fill_color = paint.stage
            paint_dropdown.bgcolor = paint.stage
            paint_dropdown.color = paint.ink
            paint_dropdown.border_color = paint.rule
            paint_dropdown.focused_border_color = paint.signal
            paint_dropdown.border_radius = paint.radius
            paint_dropdown.label_style = ft.TextStyle(
                color=paint.quiet,
                size=12,
                font_family=BODY_FONT_FAMILY,
            )
            paint_dropdown.text_style = ft.TextStyle(
                color=paint.ink,
                size=12,
                font_family=BODY_FONT_FAMILY,
            )

        def on_coat_select(e: ft.ControlEvent) -> None:
            """Paint the window with the coat chosen in the Coat control."""
            chosen = PAINT_BY_KEY.get(str(e.control.value or ""))
            if chosen is None or chosen.key == coat().key:
                return
            active[0] = chosen
            _apply_chrome()
            _rebuild_log_controls()
            _sync_status()
            hold = _live_root_hold() if session.last_tree is not None else None
            _paint_parse_tree_canvas(hold=hold)

        paint_dropdown.on_select = on_coat_select
        _apply_chrome()
        _sync_status()

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
            """Place the desktop window in the middle of the screen after size is applied."""
            web = os.environ.get("DYLAN_FLET_WEB", "").strip().lower() in ("1", "true", "yes", "on")
            codespaces = os.environ.get("CODESPACES", "").strip().lower() == "true"
            if web or codespaces:
                return
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
