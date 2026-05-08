"""Build effects from spec strings (partial ``EffectFactory``).

Recognised action keywords are dispatched to concrete ``Effect``
subclasses; macro calls expand like Java; remaining lines use
:class:`GenericEffect` (no-op).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from dylan.action.atomic.abort import Abort
from dylan.action.atomic.effect import Effect
from dylan.action.atomic.empty_effect import EmptyEffect

logger = logging.getLogger(__name__)

_MACRO_HEADER = re.compile(r"^(.+?)\((.*)\)\s*$")


def _parse_macro_header(line: str) -> tuple[str, list[str]]:
    """Return macro base name and metavar names (Java macro header rules)."""
    m = _MACRO_HEADER.match(line.strip())
    if not m:
        return line.strip(), []
    base = m.group(1).strip()
    inner = m.group(2).strip()
    if not inner:
        return base, []
    parts = [x.strip() for x in inner.split(",") if x.strip()]
    metavars = [p for p in parts if re.match(r"^[A-Z][A-Z0-9]*$", p)]
    if metavars:
        return base, metavars
    return line.strip(), []


class GenericEffect(Effect):
    """Fallback no-op effect for action specs not yet ported."""

    def __init__(self, spec: str) -> None:
        self.spec = spec

    def exec_tuple_context(self, tree: Any, context: Any) -> Any:
        """Return *tree* unchanged (no-op)."""
        return tree

    def instantiate(self) -> Effect:
        return GenericEffect(self.spec)

    def __str__(self) -> str:
        return f"GenericEffect({self.spec})"


class EffectFactory:
    """Factory mirroring Java ``EffectFactory.create``."""

    _macro_templates: dict[str, tuple[list[str], list[str]]] = {}

    @classmethod
    def clear_macro_templates(cls) -> None:
        """Reset macro definitions."""
        cls._macro_templates.clear()

    @classmethod
    def init_macro_templates(
        cls,
        cleaned_lines: list[str | None],
    ) -> tuple[int, int, tuple[str, ...]]:
        """Load ``lexical-macros.txt`` (already block-comment-stripped like Java).

        Returns ``(macros_loaded, macros_failed, failed_macro_names)`` where *macros_failed*
        counts macro headers that never received a non-empty body (including at EOF),
        and *failed_macro_names* lists those macro base names in parse order.
        """
        cls.clear_macro_templates()
        name: str | None = None
        metavars: list[str] = []
        body: list[str] = []
        loaded = 0
        failed = 0
        failed_names: list[str] = []
        for raw in cleaned_lines:
            if raw is None:
                continue
            line = raw.strip()
            if not line and not body:
                continue
            if not line and body and name is not None:
                cls._macro_templates[name] = (list(metavars), list(body))
                loaded += 1
                name, metavars, body = None, [], []
                continue
            if name is None:
                name, metavars = _parse_macro_header(line)
            else:
                body.append(line)
        if name is not None and body:
            cls._macro_templates[name] = (list(metavars), list(body))
            loaded += 1
        elif name is not None:
            failed += 1
            failed_names.append(name)
        return loaded, failed, tuple(failed_names)

    @classmethod
    def _expand_macro_body(cls, name: str, metavals: list[str]) -> list[str]:
        """Return body lines with metavars substituted (Java ``MacroTemplate.create``)."""
        entry = cls._macro_templates.get(name)
        if entry is None:
            raise KeyError(name)
        metavars, lines = entry
        if len(metavars) != len(metavals):
            raise ValueError(
                f"macro {name!r}: expected {len(metavars)} arg(s), got {len(metavals)}",
            )
        out: list[str] = []
        for raw_ln in lines:
            ln = raw_ln
            for i, mv in enumerate(metavars):
                ln = ln.replace(mv, metavals[i])
            out.append(ln)
        return out

    @classmethod
    def _create_lexical_macro(cls, line: str) -> Effect | None:
        """Parse ``name`` or ``name(a,b)`` and build a macro effect."""
        from dylan.action.atomic.lexical_macro import LexicalMacro

        s = line.strip()
        if s in cls._macro_templates:
            _, body_lines = cls._macro_templates[s]
            actions = [EffectFactory.create(x.strip()) for x in body_lines]
            return LexicalMacro(s, actions)
        m = _MACRO_HEADER.match(s)
        if not m:
            return None
        base = m.group(1).strip()
        inner = m.group(2).strip()
        arg_values = [x.strip() for x in inner.split(",") if x.strip()] if inner else []
        if base not in cls._macro_templates:
            return None
        metavars, _ = cls._macro_templates[base]
        if len(metavars) != len(arg_values):
            logger.debug("macro %s arity mismatch: want %s got %s", base, metavars, arg_values)
            return None
        expanded = cls._expand_macro_body(base, arg_values)
        actions = [EffectFactory.create(x.strip()) for x in expanded]
        return LexicalMacro(s, actions)

    @staticmethod
    def get_if_indices(lines: list[str]) -> list[int]:
        """Return indices of lines starting with ``IF`` (case-insensitive)."""
        return [i for i, ln in enumerate(lines) if ln.strip().lower().startswith("if")]

    @staticmethod
    def create_lines(lines: list[str]) -> Effect:
        """Create an effect from one or more source lines."""
        from dylan.action.atomic.if_then_else import IfThenElse

        if len(lines) == 1:
            return EffectFactory.create(lines[0])
        if lines[0].strip().lower().startswith("if"):
            return IfThenElse.from_lines(lines)
        return GenericEffect(" / ".join(lines))

    @staticmethod
    def create(line: str) -> Effect:
        """Create a single-line effect from its text specification."""
        from dylan.action.atomic.add_axiom import AddAxiom
        from dylan.action.atomic.beta_reduce import BetaReduce
        from dylan.action.atomic.conjoin import Conjoin
        from dylan.action.atomic.copy_content import CopyContent
        from dylan.action.atomic.delete import Delete
        from dylan.action.atomic.do_effect import Do
        from dylan.action.atomic.fresh_put import FreshPut
        from dylan.action.atomic.go import Go
        from dylan.action.atomic.go_first import GoFirst
        from dylan.action.atomic.go_local_event import GoLocalEvent
        from dylan.action.atomic.ground_to_root import GroundToRoot
        from dylan.action.atomic.infer_speech_act import InferSpeechAct
        from dylan.action.atomic.make import Make
        from dylan.action.atomic.merge import Merge
        from dylan.action.atomic.open_floor import OpenFloor
        from dylan.action.atomic.put import Put
        from dylan.action.atomic.saturate_scope_dep import SaturateScopeDep
        from dylan.action.atomic.ttr_fresh_put import TTRFreshPut
        from dylan.action.atomic.unassert import Unassert
        from dylan.action.atomic.unreduce import Unreduce

        s = line.strip()
        low = s.lower()

        if low.startswith(Abort.FUNCTOR.lower()):
            return Abort()
        if low.startswith(AddAxiom.FUNCTOR):
            return AddAxiom()
        if low.startswith(InferSpeechAct.FUNCTOR):
            return InferSpeechAct()
        if low.startswith(OpenFloor.FUNCTOR):
            return OpenFloor()
        if low.startswith(Unreduce.FUNCTOR):
            return Unreduce()
        if low.startswith(Unassert.FUNCTOR):
            return Unassert()
        if low.startswith(GroundToRoot.FUNCTOR):
            return GroundToRoot()

        eff: Effect | None
        eff = Make.parse(s)
        if eff is not None:
            return eff
        if low.startswith(EmptyEffect.FUNCTOR.lower()):
            return EmptyEffect()
        eff = CopyContent.parse(s)
        if eff is not None:
            return eff
        eff = Put.parse(s)
        if eff is not None:
            return eff
        eff = Delete.parse(s)
        if eff is not None:
            return eff
        eff = GoFirst.parse(s)
        if eff is not None:
            return eff
        eff = GoLocalEvent.parse(s)
        if eff is not None:
            return eff
        eff = Go.parse(s)
        if eff is not None:
            return eff
        if low.startswith(BetaReduce.FUNCTOR):
            return BetaReduce()
        eff = Merge.parse(s)
        if eff is not None:
            return eff
        eff = Conjoin.parse(s)
        if eff is not None:
            return eff
        eff = Do.parse(s)
        if eff is not None:
            return eff
        eff = FreshPut.parse(s)
        if eff is not None:
            return eff
        eff = SaturateScopeDep.parse(s)
        if eff is not None:
            return eff
        eff = TTRFreshPut.parse(s)
        if eff is not None:
            return eff

        try:
            macro = EffectFactory._create_lexical_macro(s)
        except (KeyError, ValueError) as ex:
            logger.debug("macro resolution failed for %r: %s", s, ex)
            macro = None
        if macro is not None:
            return macro

        return GenericEffect(s)

    @staticmethod
    def create_multiple(lines: list[str], if_indices: list[int]) -> list[Effect]:
        """Split *lines* at ``IF`` boundaries and build one effect per block."""
        from dylan.action.atomic.if_then_else import IfThenElse

        if not if_indices:
            return [GenericEffect(" / ".join(lines))]
        chunks: list[Effect] = []
        starts = if_indices + [len(lines)]
        for i in range(len(if_indices)):
            sub = lines[starts[i] : starts[i + 1]]
            chunks.append(IfThenElse.from_lines(sub))
        return chunks
