"""Java-parity coverage for the expanded `TTRRecordType` API surface."""

from __future__ import annotations

from dylan.formula.ttr_field import TTRField
from dylan.formula.ttr_label import TTRLabel
from dylan.formula.ttr_record_type import DrawnDimensions, ResetIndicesResult, TTRRecordType
from dylan.formula.variable import Variable
from dylan.tree.node_address import NodeAddress
from dylan.type.dstype import DSType


def test_java_style_field_accessors_and_head_helpers() -> None:
    """Java-compatible accessors expose labels, records, DS types, and head fields."""
    rt = TTRRecordType.parse("[x:e|head==x:e|p==named(x):t]")
    assert rt is not None

    assert rt.numFields() == 3
    assert rt.hasLabel(Variable("x"))
    assert rt.getField(Variable("x")) is not None
    assert rt.getHeadField() == rt.getField(TTRLabel("x"))
    assert rt.hasHead()
    assert rt.getDSType() == DSType.e
    assert TTRLabel("p") in rt.getLabels()
    assert TTRLabel("head") in rt.getRecord()


def test_relabel_remove_subsumes_and_mcs_helpers() -> None:
    """Record transforms and conservative subsumption helpers behave predictably."""
    base = TTRRecordType.parse("[x:e|p==person(x):t|head==x:e]")
    richer = TTRRecordType.parse("[x:e|p==person(x):t|q==happy(x):t|head==x:e]")
    assert base is not None and richer is not None

    relabelled = base.relabel(TTRLabel("x"), TTRLabel("z"))
    assert relabelled.hasLabel(TTRLabel("z"))
    assert not relabelled.hasLabel(TTRLabel("x"))
    assert base.subsumesBasic(richer)
    assert base.mcs(richer, {}).hasLabel(TTRLabel("p"))
    assert richer.subtract(base, {}).hasLabel(TTRLabel("q"))
    left, right = richer.minus(base)
    assert left.hasLabel(TTRLabel("q"))
    assert right.isEmpty()


def test_dependency_serialization_drawing_and_nn_helpers() -> None:
    """Dependency, rendering, and NN conversion helpers cover Java peripheral behavior."""
    rt = TTRRecordType.parse("[x:e|p==person(x):t|head==x:e]")
    assert rt is not None
    field_x = rt.getField(TTRLabel("x"))
    assert field_x is not None

    assert rt.hasDependent(field_x)
    assert rt.getDependents(field_x)
    assert "person" in rt.toPythonDictString()
    assert "\\left" in rt.toLatex()
    assert "person" in rt.toDebugString()
    assert isinstance(rt.draw(), DrawnDimensions)
    # Round-trip preserves content; field order need not match the original list order.
    back = TTRRecordType.nn2RT(rt.rt2nnNoFiller())
    assert rt.subsumes(back) and back.subsumes(rt)


def test_abstraction_reset_and_java_aliases() -> None:
    """Abstraction and reset-index helpers return typed compatibility objects."""
    rt = TTRRecordType.parse("[x7:e|p9==person(x7):t|head==x7:e]")
    assert rt is not None

    abstractions = rt.getAbstractions(DSType.e, 1)
    assert len(abstractions) == 1
    trees = rt.getEmptyAbstractions(NodeAddress())
    assert len(trees) == 1
    reset = rt.resetAllIndices()
    assert isinstance(reset, ResetIndicesResult)
    assert isinstance(reset.record_type, TTRRecordType)


def test_java_style_put_and_add_overloads() -> None:
    """`add`, `put`, and `putAtEnd` support Java-style overload patterns."""
    rt = TTRRecordType()
    label = rt.add(TTRLabel("x"), None, DSType.e)
    assert label == TTRLabel("x")
    rt.put(TTRLabel("head"), Variable("x"), DSType.e)
    rt.putAtEnd(TTRField(TTRLabel("p"), DSType.t, None))
    assert [field.label for field in rt.getFields()] == [TTRLabel("x"), TTRLabel("head"), TTRLabel("p")]
