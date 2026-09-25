"""Token checks for the Flet parser coats."""

from __future__ import annotations

from pathlib import Path

from dylan.gui.paint import (
    ASTEROID_CITY,
    DEFAULT_PAINT,
    LIFE_AQUATIC,
    ORIGINAL,
    PAINTS,
    first_installed_font,
)
from dylan.gui.tree_viz import theme_for_zoom


def test_coats_and_the_default() -> None:
    """The window opens on the original blue-grey coat. Tenenbaums is absent."""
    assert [paint.key for paint in PAINTS] == [
        "original",
        "life_aquatic",
        "asteroid_city",
    ]
    assert DEFAULT_PAINT is ORIGINAL
    assert "tenenbaums" not in {paint.key for paint in PAINTS}


def test_original_coat_matches_the_blue_grey_window() -> None:
    """Original keeps the pre-film panel, box, pointer, and primary blue."""
    assert ORIGINAL.ground == "#263238"
    assert ORIGINAL.stage == "#1C262B"
    assert ORIGINAL.rule == "#37474F"
    assert ORIGINAL.ink == "#ECEFF1"
    assert ORIGINAL.signal == "#4FC3F7"
    assert ORIGINAL.commit == "#1976D2"
    assert ORIGINAL.commit_text == "#FFFFFF"
    assert ORIGINAL.node_fill == "#455A64"
    assert ORIGINAL.pointer_stroke == "#01579B"
    assert ORIGINAL.canvas_background == "#263238"
    assert ORIGINAL.ok == "#66BB6A"
    assert ORIGINAL.error == "#EF5350"
    assert ORIGINAL.warning == "#FFB74D"
    assert ORIGINAL.interpretation == "#FFE082"
    assert ORIGINAL.radius == 6
    assert [name for name, _files in ORIGINAL.title_font_chain] == [
        "Garamond",
        "Palatino Linotype",
        "Georgia",
    ]


def test_life_aquatic_uses_several_luminous_accents() -> None:
    """Life Aquatic stays navy and splits yellow, cyan, violet, teal, and magenta."""
    assert LIFE_AQUATIC.ground == "#061428"
    assert LIFE_AQUATIC.commit == "#F5C400"
    assert LIFE_AQUATIC.signal == "#3EE0FF"
    assert LIFE_AQUATIC.node_stroke == "#8B7CFF"
    assert LIFE_AQUATIC.edge == "#3DCFC0"
    assert LIFE_AQUATIC.interpretation == "#FF6AD5"
    accents = {
        LIFE_AQUATIC.commit,
        LIFE_AQUATIC.signal,
        LIFE_AQUATIC.node_stroke,
        LIFE_AQUATIC.edge,
        LIFE_AQUATIC.interpretation,
    }
    assert len(accents) == 5
    assert LIFE_AQUATIC.radius == 4


def test_asteroid_city_tokens_stay() -> None:
    """Asteroid City keeps the dusk diorama colours."""
    assert ASTEROID_CITY.ground == "#14282B"
    assert ASTEROID_CITY.stage == "#0C1C1F"
    assert ASTEROID_CITY.rule == "#E4D2B0"
    assert ASTEROID_CITY.signal == "#3FCFC4"
    assert ASTEROID_CITY.commit == "#E37B3A"
    assert ASTEROID_CITY.radius == 2
    assert ASTEROID_CITY.canvas_background == ASTEROID_CITY.stage


def test_grounds_stay_distinct() -> None:
    """Each coat has its own dark ground. Navy belongs to Life Aquatic."""
    assert LIFE_AQUATIC.ground != ORIGINAL.ground
    assert ASTEROID_CITY.ground != LIFE_AQUATIC.ground
    assert len({paint.ground for paint in PAINTS}) == 3


def test_log_tones_stay_in_their_families() -> None:
    """Ok stays green, error stays red, and warning stays amber on every coat."""
    for paint in PAINTS:
        assert paint.ok != paint.error
        assert paint.warning not in {paint.ok, paint.error, paint.signal}


def test_canvas_theme_keeps_coat_colours_when_zoom_scales_strokes() -> None:
    """Zoom scales strokes and leaves the coat's DS Tree colours in place."""
    base = LIFE_AQUATIC.canvas_theme()
    themed = theme_for_zoom(2.0, base)
    assert themed.background == LIFE_AQUATIC.canvas_background
    assert themed.pointer_fill == LIFE_AQUATIC.signal
    assert themed.pointer_stroke == LIFE_AQUATIC.pointer_stroke
    assert themed.node_stroke == LIFE_AQUATIC.node_stroke
    assert themed.edge_color == LIFE_AQUATIC.edge
    assert themed.text_color == LIFE_AQUATIC.ink
    assert themed.edge_width == base.edge_width * 2.0
    assert themed.corner_radius == base.corner_radius * 2.0


def test_title_font_chain_falls_through_to_an_installed_face(tmp_path: Path) -> None:
    """The first candidate whose file exists wins; otherwise the last name is kept."""
    missing = tmp_path / "missing"
    missing.mkdir()
    present = tmp_path / "present"
    present.mkdir()
    (present / "GARA.TTF").write_bytes(b"")
    chain = ORIGINAL.title_font_chain
    assert first_installed_font(chain, font_dirs=(missing,)) == "Georgia"
    assert first_installed_font(chain, font_dirs=(present, missing)) == "Garamond"


def test_caption_faces() -> None:
    """Captions stay Segoe UI. Life Aquatic titles stay the geometric chain."""
    assert ORIGINAL.caption_font_chain == (("Segoe UI", ("segoeui.ttf",)),)
    assert LIFE_AQUATIC.caption_font_chain == (("Segoe UI", ("segoeui.ttf",)),)
    assert ASTEROID_CITY.caption_font_chain[0][0] == "Segoe UI"
    assert [name for name, _files in LIFE_AQUATIC.title_font_chain] == [
        "Century Gothic",
        "Trebuchet MS",
        "Segoe UI",
    ]
    assert [name for name, _files in ASTEROID_CITY.title_font_chain] == [
        "Copperplate Gothic Light",
        "Copperplate Gothic Bold",
        "Segoe UI",
    ]
