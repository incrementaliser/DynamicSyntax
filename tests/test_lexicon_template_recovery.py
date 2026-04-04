"""Templates referenced by lexicon but inside ``//*`` … ``*//`` are recovered from raw file."""

from __future__ import annotations

from dylan.action.lexicon import (
    Lexicon,
    _recover_template_from_raw_lines,
    _referenced_template_names_from_lexicon,
)


def test_referenced_template_names() -> None:
    """Second column of lexicon lines becomes template names."""
    lines = ["word1  tpl_a  x", None, "  tpl_b  y  z"]
    assert _referenced_template_names_from_lexicon(lines) == {"tpl_a", "tpl_b"}


def test_recover_det_quant_shape_from_synthetic_raw() -> None:
    """Recovery finds a header + body across a ``*//`` line (like English resource)."""
    raw = """
//* deprecated wrapper
det_quant(QUANT)
IF      ?ty(e)
THEN    put(ty(e))
ELSE    abort

*//

other(NAME)
IF      ty(t)
THEN    abort
ELSE    abort

""".splitlines()
    t = _recover_template_from_raw_lines(raw, "det_quant")
    assert t is not None
    assert t.name == "det_quant"
    assert t.metavars == ["QUANT"]
    assert any("IF" in ln for ln in t.lines)
    assert any("ELSE" in ln for ln in t.lines)


def test_full_lexicon_loads_det_quant_from_dyland_resource() -> None:
    """End-to-end: bundled DyLan English folder provides ``det_quant`` after recovery."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "DyLan" / "resource" / "2015-english-ttr"
    if not (root / "lexical-actions.txt").is_file():
        root = Path(r"C:\ArashMath\DyLan\DyLan\resource\2015-english-ttr")
    if not (root / "lexical-actions.txt").is_file():
        return
    lex = Lexicon(root)
    assert "det_quant" in lex._templates
    assert lex.get("the", []) or "the" in lex
