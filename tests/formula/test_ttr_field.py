"""Tests for `TTRField` parsing and structural operations."""

from __future__ import annotations

from dylan.formula.meta_ttr_label import MetaTTRLabel
from dylan.formula.ttr_field import TTRField
from dylan.formula.ttr_label import TTRLabel
from dylan.formula.variable import Variable
from dylan.type.dstype import DSType


def test_parse_simple_typed_field() -> None:
    """``label : dsType`` parses with manifest ``None``."""
    f = TTRField.parse("x:es")
    assert f is not None
    assert f.label == TTRLabel("x")
    assert f.ds_type == DSType.es
    assert f.manifest_type is None


def test_parse_manifest_before_label_separator() -> None:
    """``head==e0:es`` form binds manifest before the ``:`` DS type."""
    f = TTRField.parse("head==e0:es")
    assert f is not None
    assert f.label == TTRLabel("head")
    assert f.ds_type == DSType.es
    assert f.manifest_type is not None
    assert str(f.manifest_type) == "e0"


def test_meta_label_field_without_type_constructed() -> None:
    """Binder-only meta fields use :class:`MetaTTRLabel` with no DS type (see ``TTRField.parse``)."""
    f = TTRField(MetaTTRLabel("R"), None, None)
    assert isinstance(f.label, MetaTTRLabel)
    assert f.ds_type is None
    assert f.manifest_type is None


def test_clone_copies_manifest() -> None:
    """Cloning duplicates the manifest tree."""
    f = TTRField.parse("head==e0:es")
    assert f is not None
    c = f.clone()
    assert isinstance(c, TTRField)
    assert c.label == f.label
    assert str(c.manifest_type) == str(f.manifest_type)


def test_evaluate_propagates_to_manifest() -> None:
    """``evaluate`` delegates to the manifest when present."""
    f = TTRField.parse("head==e0:es")
    assert f is not None
    ev = f.evaluate()
    assert isinstance(ev, TTRField)
    assert str(ev.manifest_type) == "e0"


def test_substitute_renames_label_when_var_matches() -> None:
    """Substituting for the field’s variable rewrites the label via ``ttr_label_from_variable``."""
    f = TTRField(TTRLabel("x"), DSType.e, Variable("y"))
    out = f.substitute(Variable("x"), Variable("z"))
    assert isinstance(out, TTRField)
    assert out.label == TTRLabel("z")
    assert str(out.manifest_type) == "y"
