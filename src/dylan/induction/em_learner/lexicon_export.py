"""Serialize learned :class:`~dylan.action.atomic.effect.Effect` trees to lexical source lines for file export."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from dylan.action.atomic.if_then_else import IfThenElse
from dylan.induction.em_learner.lexical_hypothesis import _CompositeEffect
from dylan.tree.label.labels import Requirement, TypeLabel
from dylan.type.dstype import DSType

if TYPE_CHECKING:
    from dylan.action.action import Action
    from dylan.action.atomic.effect import Effect


def spine_actions_to_lexical_source_lines(spine: Iterable["Action"]) -> list[str]:
    """Flatten a linear candidate spine like Java ``LexicalAction.flatten`` (``LexicalAction(String, ArrayList<Action>)``) into reloadable source lines."""
    from dylan.action.action import Action
    from dylan.action.atomic.effect import Effect
    from dylan.action.lexical_action import LexicalAction
    from dylan.induction.em_learner.lexical_hypothesis import LexicalHypothesis

    effects: list[Effect] = []
    for a in spine:
        if isinstance(a, LexicalAction):
            effects.extend(list(a.effects))
        elif isinstance(a, LexicalHypothesis):
            eff = a.get_effect() if hasattr(a, "get_effect") else None
            if eff is None:
                eff = getattr(a, "effect", None)
            if eff is None:
                continue
            effects.append(eff)
        elif isinstance(a, Action):
            ge = a.get_effect() if hasattr(a, "get_effect") else None
            if ge is not None:
                effects.append(ge)
    if not effects:
        return []
    blocks: list[list[str]] = []
    for e in effects:
        block = effect_to_lexical_lines(e)
        if block and block != _minimal_abort_block():
            blocks.append(block)
    if not blocks:
        return []
    out: list[str] = []
    for i, block in enumerate(blocks):
        if i:
            out.append("")
        out.extend(block)
    return out


def effect_to_lexical_lines(effect: "Effect | None") -> list[str]:
    """Turn an induction *effect* into ``IF``/``THEN``/``ELSE`` source lines parseable by :class:`~dylan.action.atomic.effect_factory.EffectFactory`."""
    if effect is None:
        return _minimal_abort_block()
    if isinstance(effect, IfThenElse):
        return _if_then_else_to_lines(effect, base_indent="", top_level=True)
    if isinstance(effect, _CompositeEffect):
        return _composite_effect_to_lines(effect)
    spec = str(effect).strip()
    if not spec:
        return _minimal_abort_block()
    guard = str(Requirement(TypeLabel(DSType.t)))
    return [f"IF      {guard}", f"THEN    {spec}", "ELSE    abort"]


def _minimal_abort_block() -> list[str]:
    """Return a trivial failing block when there is no effect to serialize."""
    guard = str(Requirement(TypeLabel(DSType.t)))
    return [f"IF      {guard}", "THEN    abort", "ELSE    abort"]


def _composite_effect_to_lines(composite: _CompositeEffect) -> list[str]:
    """Wrap a flat list of atomic effects as an IF/THEN/ELSE block (induced ``put`` sequences)."""
    guard = str(Requirement(TypeLabel(DSType.t)))
    specs = [str(e).strip() for e in composite.effects if str(e).strip()]
    if not specs:
        return _minimal_abort_block()
    lines = [f"IF      {guard}", f"THEN    {specs[0]}"]
    lines.extend(f"        {s}" for s in specs[1:])
    lines.append("ELSE    abort")
    return lines


def _if_then_else_to_lines(
    ite: IfThenElse,
    *,
    base_indent: str,
    top_level: bool,
) -> list[str]:
    """Serialize *ite* with Java-like ``IF``/``THEN``/``ELSE`` layout and optional *base_indent*."""
    out: list[str] = []
    kw_if = "IF      " if top_level else "if      "
    if_labels = list(ite.if_labels)
    if not if_labels:
        guard = str(Requirement(TypeLabel(DSType.t)))
        out.append(f"{base_indent}{kw_if}{guard}")
    else:
        for i, lab in enumerate(if_labels):
            s = str(lab)
            if i == 0:
                out.append(f"{base_indent}{kw_if}{s}")
            else:
                out.append(f"{base_indent}        {s}")

    inner_pad = base_indent + "        "
    first_then = True
    for eff in ite.then_effects:
        if isinstance(eff, IfThenElse):
            sub = _if_then_else_to_lines(eff, base_indent=inner_pad, top_level=False)
            if not sub:
                continue
            if first_then:
                first_ln = sub[0]
                remainder = first_ln[len(inner_pad) :] if first_ln.startswith(inner_pad) else first_ln.lstrip()
                out.append(f"{base_indent}THEN    {remainder}")
                out.extend(sub[1:])
                first_then = False
            else:
                out.extend(sub)
        else:
            spec = str(eff).strip()
            if first_then:
                out.append(f"{base_indent}THEN    {spec}")
                first_then = False
            else:
                out.append(f"{base_indent}        {spec}")

    if not ite.else_effects:
        out.append(f"{base_indent}ELSE    abort")
        return out

    first_else = True
    for eff in ite.else_effects:
        if isinstance(eff, IfThenElse):
            sub = _if_then_else_to_lines(eff, base_indent=inner_pad, top_level=False)
            if not sub:
                continue
            if first_else:
                first_ln = sub[0]
                remainder = first_ln[len(inner_pad) :] if first_ln.startswith(inner_pad) else first_ln.lstrip()
                out.append(f"{base_indent}ELSE    {remainder}")
                out.extend(sub[1:])
                first_else = False
            else:
                out.extend(sub)
        else:
            spec = str(eff).strip()
            if first_else:
                out.append(f"{base_indent}ELSE    {spec}")
                first_else = False
            else:
                out.append(f"{base_indent}        {spec}")
    return out
