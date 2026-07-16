"""Incremental DS-VSS: plausibility trajectories and expectation.

Two word-by-word services on top of :class:`~dylan.vss.decorate.VSSDecorator`:

- **incremental plausibility**: decorate each successive partial tree of a
  DyLan parse trace and read off the root plausibility — the trajectory of
  Sadrzadeh et al. (2018), section 4 (incomplete utterances are dense,
  high-entropy sentence vectors; completed ones sharpen);
- **expectation**: rank candidate continuations by the plausibility they
  would induce (section 5.2) — either verb continuations given a subject, or
  object continuations given a subject–verb pair.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from dylan.tree.tree import Tree
from dylan.vss.decorate import VSSDecoration, VSSDecorator
from dylan.vss.lexicon import VSSLexicon
from dylan.vss.spaces import VSSValue, contract, plausibility


def decorate_traces(
    trees: Sequence[Tree], decorator: VSSDecorator
) -> list[VSSDecoration]:
    """Decorate each tree of an incremental parse trace."""
    return [decorator.decorate(t) for t in trees]


def plausibility_trajectory(
    decorations: Iterable[VSSDecoration],
) -> list[float | None]:
    """Root plausibility of each decoration, in order."""
    out: list[float | None] = []
    for dec in decorations:
        p = dec.plausibility()
        out.append(p if isinstance(p, float) or p is None else None)
    return out


def verb_continuations(
    subject: VSSValue,
    verbs: Iterable[str],
    lexicon: VSSLexicon,
) -> dict[str, float]:
    """Plausibility induced by each candidate intransitive *verb*.

    ``subject`` is the word-space vector decorating the subject node; each
    verb's ``W ⊗ S`` matrix is contracted with it and the resulting
    sentence vector's plausibility is returned — the model of hearer
    expectation in section 5.2 of the paper.
    """
    spaces = (lexicon.word_space, lexicon.sentence_space)
    out: dict[str, float] = {}
    for verb in verbs:
        mat = lexicon.lookup(verb, spaces)
        if mat is None:
            continue
        out[verb] = plausibility(contract(mat, subject))
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def object_continuations(
    subject: VSSValue,
    verb: str,
    objects: Iterable[str],
    lexicon: VSSLexicon,
) -> dict[str, float]:
    """Plausibility induced by each candidate *object* of transitive *verb*.

    The verb cube ``T_ijk`` is contracted with the object vector (index
    ``k``) and then with the *subject* vector (index ``i``), as in
    ``T_i^subj T_ijk^verb T_k^obj``.
    """
    cube_spaces = (
        lexicon.word_space,
        lexicon.sentence_space,
        lexicon.word_space,
    )
    cube = lexicon.lookup(verb, cube_spaces)
    if cube is None:
        return {}
    out: dict[str, float] = {}
    for obj in objects:
        vec = lexicon.lookup(obj, (lexicon.word_space,))
        if vec is None:
            continue
        out[obj] = plausibility(contract(contract(cube, vec), subject))
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))
