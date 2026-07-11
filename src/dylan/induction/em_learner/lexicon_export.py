"""Serialize learned :class:`~dylan.action.atomic.effect.Effect` trees to lexical source lines for file export."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from dylan.action.atomic.if_then_else import IfThenElse
from dylan.induction.em_learner.lexical_hypothesis import _CompositeEffect

if TYPE_CHECKING:
    from dylan.action.action import Action
    from dylan.action.atomic.effect import Effect

# Java ``IfThenElse.tabSizeForPrinting = 6``: ``IF`` + 4 spaces, ``THEN``/``ELSE`` + 2 spaces.
_TAB = 6
_IF_PAD = " " * (_TAB - len("IF"))
_THEN_PAD = " " * (_TAB - len("THEN"))
_ELSE_PAD = " " * (_TAB - len("ELSE"))
_CONT = " " * _TAB


def spine_actions_to_lexical_source_lines(spine: Iterable["Action"]) -> list[str]:
    """Flatten a linear candidate spine like Java ``LexicalAction.flatten`` into reloadable source lines."""
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
    for block in blocks:
        out.extend(block)
    return out


def effect_to_lexical_lines(effect: "Effect | None") -> list[str]:
    """Turn an induction *effect* into ``IF``/``THEN``/``ELSE`` source lines (Java ``IfThenElse.toString`` layout)."""
    if effect is None:
        return _minimal_abort_block()
    if isinstance(effect, IfThenElse):
        return _if_then_else_to_lines(effect, embedding=0)
    if isinstance(effect, _CompositeEffect):
        return _composite_effect_to_lines(effect)
    spec = str(effect).strip()
    if not spec:
        return _minimal_abort_block()
    return [f"IF{_IF_PAD}", f"THEN{_THEN_PAD}{spec}", f"ELSE{_ELSE_PAD}abort"]


def _minimal_abort_block() -> list[str]:
    """Return a trivial failing block when there is no effect to serialize."""
    return [f"IF{_IF_PAD}", f"THEN{_THEN_PAD}abort", f"ELSE{_ELSE_PAD}abort"]


def _composite_effect_to_lines(composite: _CompositeEffect) -> list[str]:
    """Wrap a flat list of atomic effects as an IF/THEN/ELSE block (induced ``put`` sequences)."""
    specs = [str(e).strip() for e in composite.effects if str(e).strip()]
    if not specs:
        return _minimal_abort_block()
    lines = [f"IF{_IF_PAD}", f"THEN{_THEN_PAD}{specs[0]}"]
    lines.extend(f"{_CONT}{s}" for s in specs[1:])
    lines.append(f"ELSE{_ELSE_PAD}abort")
    return lines


def _if_then_else_to_lines(ite: IfThenElse, *, embedding: int) -> list[str]:
    """Serialize *ite* with Java ``tabSizeForPrinting=6`` layout."""
    tabs = _CONT * embedding
    out: list[str] = []
    if_labels = list(ite.if_labels)
    if not if_labels:
        out.append(f"{tabs}IF{_IF_PAD}")
    else:
        for i, lab in enumerate(if_labels):
            s = str(lab)
            if i == 0:
                out.append(f"{tabs}IF{_IF_PAD}{s}")
            else:
                out.append(f"{tabs}{_CONT}{s}")

    first_then = True
    for eff in ite.then_effects:
        if isinstance(eff, IfThenElse):
            sub = _if_then_else_to_lines(eff, embedding=embedding + 1)
            if not sub:
                continue
            if first_then:
                first_ln = sub[0]
                payload = first_ln[len(tabs) + _TAB :] if first_ln.startswith(tabs + _CONT) else first_ln.lstrip()
                out.append(f"{tabs}THEN{_THEN_PAD}{payload}")
                out.extend(sub[1:])
                first_then = False
            else:
                out.extend(sub)
        else:
            spec = str(eff).strip()
            if first_then:
                out.append(f"{tabs}THEN{_THEN_PAD}{spec}")
                first_then = False
            else:
                out.append(f"{tabs}{_CONT}{spec}")

    if not ite.else_effects:
        out.append(f"{tabs}ELSE{_ELSE_PAD}abort")
        return out

    first_else = True
    for eff in ite.else_effects:
        if isinstance(eff, IfThenElse):
            sub = _if_then_else_to_lines(eff, embedding=embedding + 1)
            if not sub:
                continue
            if first_else:
                first_ln = sub[0]
                payload = first_ln[len(tabs) + _TAB :] if first_ln.startswith(tabs + _CONT) else first_ln.lstrip()
                out.append(f"{tabs}ELSE{_ELSE_PAD}{payload}")
                out.extend(sub[1:])
                first_else = False
            else:
                out.extend(sub)
        else:
            spec = str(eff).strip()
            if first_else:
                out.append(f"{tabs}ELSE{_ELSE_PAD}{spec}")
                first_else = False
            else:
                out.append(f"{tabs}{_CONT}{spec}")
    return out
