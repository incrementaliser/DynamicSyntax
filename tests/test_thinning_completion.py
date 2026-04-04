"""Regression: *thinning metavariable binding and completion-related tree state."""

from __future__ import annotations

from dylan.action.atomic.effect_factory import EffectFactory
from dylan.action.meta.element import reset_all_meta_bindings
from dylan.tree.label.labels import MetaLabel, Requirement, TypeLabel, label_factory_create
from dylan.tree.node import Node
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree
from dylan.type.dstype import DSType


def test_label_factory_meta_label_for_v_to_z_and_meta() -> None:
    """Uppercase ``V``–``Z`` (optional digits) and ``META`` become side-effecting :class:`MetaLabel`."""
    assert isinstance(label_factory_create("X"), MetaLabel)
    assert isinstance(label_factory_create("V0"), MetaLabel)
    assert isinstance(label_factory_create("META"), MetaLabel)
    assert not isinstance(label_factory_create("x"), MetaLabel)


def test_thinning_deletes_requirement_when_pattern_matches() -> None:
    """``*thinning`` IF ``?X`` + ``X`` must bind ``X`` to an on-node label and ``delete(?X)`` drops the requirement."""
    reset_all_meta_bindings()
    lines = [
        "IF      ?X",
        "        X",
        "THEN    delete(?X)",
        "ELSE    abort",
    ]
    eff = EffectFactory.create_lines(lines)
    addr = NodeAddress()
    node = Node(
        addr,
        [Requirement(TypeLabel(DSType.t)), TypeLabel(DSType.t)],
    )
    t: Tree = Tree.__new__(Tree)
    dict.__init__(t)
    t.root_addr = addr
    t.pointer = addr
    t._entity_pool = []
    t._event_pool = []
    t._proposition_pool = []
    t._record_type_pool = []
    t._predicate_pool = []
    t[addr] = node

    out = eff.exec_tuple_context(t, None)
    assert out is not None
    assert not any(isinstance(lab, Requirement) for lab in out.pointed_node.labels)
    assert out.is_complete()
    reset_all_meta_bindings()


def test_requirement_instantiate_propagates_inner() -> None:
    """Bound metavar inside ``?`` must resolve via :meth:`Requirement.instantiate`."""
    reset_all_meta_bindings()
    try:
        z = MetaLabel.get("Z")
        assert z == TypeLabel(DSType.e)
        req = Requirement(z)
        inst = req.instantiate()
        assert isinstance(inst, Requirement)
        assert isinstance(inst.inner, TypeLabel)
        assert inst.inner.type == DSType.e
    finally:
        reset_all_meta_bindings()
