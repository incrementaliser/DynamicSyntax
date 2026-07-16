"""Extract predicate constants from DS formula labels.

DS-VSS lexical lookup is keyed by *predicate constants* (``like``, ``john``,
…).  In DyLan these occur inside several formula formalisms:

- TTR record types, e.g. ``[x==john : e|head==x : e|p==male(x) : t]``;
- TTR lambda abstracts (verbs), whose bodies contain an eventuality field
  such as ``e1==like : es``;
- plain epsilon-calculus/FOL terms (``john``, ``λyλx.like(x, y)`` …).

The extraction here is deliberately syntactic and tolerant: it scans the
printed form of a formula for manifest constants of the requested TTR/DS
basic type, and falls back to a bare alphanumeric token.
"""

from __future__ import annotations

import re

# Manifest constants in TTR record types: ``label==constant : type``.
_MANIFEST_RE = re.compile(r"==\s*([A-Za-z][\w'-]*)\s*:\s*([A-Za-z]+)")

# Bare single-token formulae, optionally with FOL prime/backquote decoration.
_BARE_RE = re.compile(r"^[^A-Za-z]*([A-Za-z][\w]*)[^A-Za-z]*$")


def extract_constant(formula: object, ds_type: str) -> str | None:
    """Return the first manifest constant of basic type *ds_type* in *formula*.

    :param formula: a :class:`~dylan.formula.formula.Formula` (or any object
        whose string form follows the conventions above).
    :param ds_type: basic type name, e.g. ``"e"`` for entity constants or
        ``"es"`` for eventuality (verb predicate) constants.
    """
    if formula is None:
        return None
    text = str(formula)
    best: str | None = None
    for match in _MANIFEST_RE.finditer(text):
        const, ty = match.group(1), match.group(2)
        if ty == ds_type:
            # Prefer the earliest constant of the requested type.
            best = const
            break
    if best is not None:
        return best
    # Fall back to a bare token (epsilon-calculus style constants such as
    # ``john'`` or plain FOL predicates such as ``like``).
    bare = _BARE_RE.match(text.strip())
    if bare is not None:
        return bare.group(1)
    return None


def extract_entity(formula: object) -> str | None:
    """Entity constant of an entity-type node formula (``john`` in ``x==john : e``)."""
    return extract_constant(formula, "e")


def extract_event(formula: object) -> str | None:
    """Eventuality constant of a verb node formula (``like`` in ``e1==like : es``)."""
    return extract_constant(formula, "es")
