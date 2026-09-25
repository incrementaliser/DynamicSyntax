"""Coats of paint for the Flet parser window.

The window opens on the original blue-grey coat. Life Aquatic is a navy
instrument panel with several luminous accents. Asteroid City at dusk is the
ink-teal diorama. The parser arrangement is shared. Only colour, type, and
corner radius change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dylan.gui.tree_viz import CanvasTreeTheme

FontCandidate = tuple[str, tuple[str, ...]]
"""A typeface and the filenames that mean it is installed."""

_SEGOE_UI: FontCandidate = ("Segoe UI", ("segoeui.ttf",))
_LIFE_AQUATIC_TITLE: tuple[FontCandidate, ...] = (
    ("Century Gothic", ("GOTHIC.TTF",)),
    ("Trebuchet MS", ("trebuc.ttf",)),
    _SEGOE_UI,
)
_ORIGINAL_TITLE: tuple[FontCandidate, ...] = (
    ("Garamond", ("GARA.TTF",)),
    ("Palatino Linotype", ("pala.ttf",)),
    ("Georgia", ("georgia.ttf",)),
)
_ASTEROID_TITLE: tuple[FontCandidate, ...] = (
    ("Copperplate Gothic Light", ("COPRGTL.TTF",)),
    ("Copperplate Gothic Bold", ("COPRGTB.TTF",)),
    _SEGOE_UI,
)
_BODY_CAPTION: tuple[FontCandidate, ...] = (_SEGOE_UI,)


def default_font_dirs() -> tuple[Path, ...]:
    """Return existing font directories for this machine."""
    candidates: list[Path] = []
    windir = os.environ.get("WINDIR")
    if windir:
        candidates.append(Path(windir) / "Fonts")
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    candidates.append(Path.home() / "Library" / "Fonts")
    return tuple(path for path in candidates if path.is_dir())


def first_installed_font(
    candidates: Sequence[FontCandidate],
    *,
    font_dirs: Sequence[Path] | None = None,
) -> str:
    """Return the family name of the first installed candidate.

    *candidates* is an ordered ``(family_name, filenames)`` chain. A candidate
    matches when one of its filenames is present in a font directory. If none
    match, the last family name is returned so the caller still has a face.
    """
    if not candidates:
        raise ValueError("font candidate chain is empty")
    directories = tuple(font_dirs) if font_dirs is not None else default_font_dirs()
    for family, filenames in candidates:
        for directory in directories:
            for filename in filenames:
                if (directory / filename).is_file():
                    return family
    return candidates[-1][0]


@dataclass(frozen=True)
class GuiPaint:
    """Colours, type, and radius for one coat of the parser window.

    ``ground`` is the page. ``stage`` is caption boxes, the title plate, and the
    DS Tree canvas. ``rule`` is the 1px border. ``ink`` is body text. ``quiet``
    is captions and inactive tabs. ``signal`` is the pointer, the active tab
    mark, and action-log lines. ``commit`` is the filled Load grammar and Parse
    buttons. Log tones keep their meanings: ok is a green, error a red, warning
    an amber.
    """

    key: str
    label: str
    radius: int
    ground: str
    stage: str
    rule: str
    ink: str
    quiet: str
    hint: str
    signal: str
    commit: str
    commit_text: str
    node_fill: str
    node_stroke: str
    edge: str
    ok: str
    error: str
    warning: str
    interpretation: str
    canvas_background: str
    pointer_stroke: str
    title_font_chain: tuple[FontCandidate, ...]
    caption_font_chain: tuple[FontCandidate, ...]

    def title_font(self, *, font_dirs: Sequence[Path] | None = None) -> str:
        """Return the installed title face for this coat."""
        return first_installed_font(self.title_font_chain, font_dirs=font_dirs)

    def caption_font(self, *, font_dirs: Sequence[Path] | None = None) -> str:
        """Return the installed caption face for this coat."""
        return first_installed_font(self.caption_font_chain, font_dirs=font_dirs)

    def canvas_theme(self) -> CanvasTreeTheme:
        """Return DS Tree colours for this coat at Zoom 100%."""
        return CanvasTreeTheme(
            background=self.canvas_background,
            edge_color=self.edge,
            node_fill=self.node_fill,
            node_stroke=self.node_stroke,
            pointer_fill=self.signal,
            pointer_stroke=self.pointer_stroke,
            text_color=self.ink,
        )


# The blue-grey coat the window used before the film paints. Title is Garamond.
ORIGINAL = GuiPaint(
    key="original",
    label="Original",
    radius=6,
    ground="#263238",
    stage="#1C262B",
    rule="#37474F",
    ink="#ECEFF1",
    quiet="#B0BEC5",
    hint="#90A4AE",
    signal="#4FC3F7",
    commit="#1976D2",
    commit_text="#FFFFFF",
    node_fill="#455A64",
    node_stroke="#263238",
    edge="#B0BEC5",
    ok="#66BB6A",
    error="#EF5350",
    warning="#FFB74D",
    interpretation="#FFE082",
    canvas_background="#263238",
    pointer_stroke="#01579B",
    title_font_chain=_ORIGINAL_TITLE,
    caption_font_chain=_BODY_CAPTION,
)

# Futuristic coat: a navy instrument panel. Yellow commits, phosphor trace,
# violet node frames, teal edges, magenta interpretation lines.
LIFE_AQUATIC = GuiPaint(
    key="life_aquatic",
    label="Life Aquatic",
    radius=4,
    ground="#061428",
    stage="#0A1C38",
    rule="#245E8C",
    ink="#E8F7FF",
    quiet="#8FB4C9",
    hint="#5E7F99",
    signal="#3EE0FF",
    commit="#F5C400",
    commit_text="#061428",
    node_fill="#102848",
    node_stroke="#8B7CFF",
    edge="#3DCFC0",
    ok="#2EE6A6",
    error="#FF4B6E",
    warning="#FFB020",
    interpretation="#FF6AD5",
    canvas_background="#0A1C38",
    pointer_stroke="#E8F7FF",
    title_font_chain=_LIFE_AQUATIC_TITLE,
    caption_font_chain=_BODY_CAPTION,
)

# Creative coat: a dusk diorama, inset stages, sand rules, turquoise pointer.
ASTEROID_CITY = GuiPaint(
    key="asteroid_city",
    label="Asteroid City",
    radius=2,
    ground="#14282B",
    stage="#0C1C1F",
    rule="#E4D2B0",
    ink="#F6EBD4",
    quiet="#CDB892",
    hint="#8EAA9F",
    signal="#3FCFC4",
    commit="#E37B3A",
    commit_text="#1A100C",
    node_fill="#1A3336",
    node_stroke="#3FCFC4",
    edge="#8EAA9F",
    ok="#7EAA62",
    error="#E25B4C",
    warning="#E6A23C",
    interpretation="#F0D48A",
    canvas_background="#0C1C1F",
    pointer_stroke="#F6EBD4",
    title_font_chain=_ASTEROID_TITLE,
    caption_font_chain=_BODY_CAPTION,
)

PAINTS: tuple[GuiPaint, ...] = (ORIGINAL, LIFE_AQUATIC, ASTEROID_CITY)
# Original first, then Life Aquatic, then Asteroid City.

PAINT_BY_KEY: dict[str, GuiPaint] = {paint.key: paint for paint in PAINTS}
# Lookup from GuiPaint.key to the coat.

DEFAULT_PAINT: GuiPaint = ORIGINAL
# The coat shown when the window opens.

BODY_FONT_FAMILY: str = "Segoe UI"
# Body face for every coat. Logs and DS Tree labels stay Consolas.
