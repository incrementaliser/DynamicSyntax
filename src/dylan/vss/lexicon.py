"""Vector-space lexicon for DS-VSS.

A :class:`VSSLexicon` assigns:

- **entity vectors** in the word space ``W`` to entity predicates
  (``john``, ``baby``, …);
- **matrices** in ``W ⊗ S`` to one-place (intransitive) predicates;
- **cubes** in ``W ⊗ S ⊗ W`` to two-place (transitive) predicates.

Two construction routes are provided, mirroring section 4 of Sadrzadeh et
al. (2018):

- direct assignment of distributional vectors/matrices/cubes (e.g. the
  paper's worked ``baby``/``footballer`` example); and
- *learning* from text: word-space vectors from co-occurrence counts (with
  PPMI weighting), and verb plausibility tensors from (verb, entity)
  co-occurrence — plausibility approximated by co-occurrence of verb and
  entity in the same context, implausibility by occurrence of the verb
  without the entity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from dylan.vss.spaces import (
    VSSValue,
    VectorSpace,
    plausibility_space,
)

#: Tensor-type signatures used for lexicon lookup.
MATRIX = 2
CUBE = 3


def ppm(matrix: np.ndarray, *, positive: bool = True) -> np.ndarray:
    """Pointwise mutual information of a raw count matrix.

    Rows are targets, columns contexts.  With ``positive=True`` (the
    default) this is PPMI, the standard weighting for distributional word
    vectors.
    """
    m = np.asarray(matrix, dtype=float)
    total = m.sum()
    if total <= 0:
        return np.zeros_like(m)
    row_marg = m.sum(axis=1, keepdims=True)
    col_marg = m.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((m * total) / (row_marg @ col_marg))
    pmi[~np.isfinite(pmi)] = 0.0
    return np.maximum(pmi, 0.0) if positive else pmi


@dataclass
class VSSLexicon:
    """Distributional lexicon for DS-VSS composition.

    :param word_space: the word/entity space ``W`` (basis = context words).
    :param sentence_space: the sentence space ``S`` (defaults to the
        two-dimensional plausibility space with basis ``(⊤, ⊥)``).
    """

    word_space: VectorSpace
    sentence_space: VectorSpace = field(default_factory=plausibility_space)

    def __post_init__(self) -> None:
        self._entities: dict[str, np.ndarray] = {}
        self._matrices: dict[str, np.ndarray] = {}
        self._cubes: dict[str, np.ndarray] = {}
        self._generic: dict[tuple[str, tuple[str, ...]], VSSValue] = {}

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------
    def add_entity(self, predicate: str, vector: Sequence[float]) -> None:
        """Register the word-space vector of an entity predicate."""
        vec = np.asarray(vector, dtype=float)
        if vec.shape != (self.word_space.dim,):
            raise ValueError(
                f"entity {predicate!r}: expected vector of dim {self.word_space.dim}, "
                f"got shape {vec.shape}"
            )
        self._entities[predicate] = vec

    def add_intransitive(self, predicate: str, matrix: Sequence[Sequence[float]]) -> None:
        """Register the ``W ⊗ S`` matrix of a one-place predicate.

        ``matrix[i][j]`` is the value on basis pair ``(W_i, S_j)``; for the
        plausibility instance, column 0 is ``⊤`` and column 1 is ``⊥``.
        """
        mat = np.asarray(matrix, dtype=float)
        expected = (self.word_space.dim, self.sentence_space.dim)
        if mat.shape != expected:
            raise ValueError(
                f"intransitive {predicate!r}: expected matrix of shape {expected}, "
                f"got {mat.shape}"
            )
        self._matrices[predicate] = mat

    def add_transitive(self, predicate: str, cube: Sequence[Sequence[Sequence[float]]]) -> None:
        """Register the ``W ⊗ S ⊗ W`` cube of a two-place predicate.

        Indices are ``(subject, sentence, object)`` following the paper's
        ``T_ijk`` convention.
        """
        arr = np.asarray(cube, dtype=float)
        expected = (
            self.word_space.dim,
            self.sentence_space.dim,
            self.word_space.dim,
        )
        if arr.shape != expected:
            raise ValueError(
                f"transitive {predicate!r}: expected cube of shape {expected}, got {arr.shape}"
            )
        self._cubes[predicate] = arr

    def add_value(self, predicate: str, value: VSSValue) -> None:
        """Register an arbitrary tensor for *predicate* (advanced use)."""
        self._generic[(predicate, value.space_names())] = value

    # ------------------------------------------------------------------
    # lookup
    # ------------------------------------------------------------------
    def predicates(self) -> set[str]:
        """All predicates with at least one registered tensor."""
        return (
            set(self._entities)
            | set(self._matrices)
            | set(self._cubes)
            | {p for p, _ in self._generic}
        )

    def lookup(self, predicate: str, spaces: tuple[VectorSpace, ...]) -> VSSValue | None:
        """Tensor of *predicate* for the factor-space tuple *spaces*.

        Lookup is by tensor *shape signature*: entities serve any order-1
        word-space request; matrices any ``W ⊗ S`` request; cubes any
        ``W ⊗ S ⊗ W`` request.  Returns ``None`` if nothing is registered.
        """
        names = tuple(s.name for s in spaces)
        generic = self._generic.get((predicate, names))
        if generic is not None:
            return generic
        if len(spaces) == 1 and spaces[0] == self.word_space:
            vec = self._entities.get(predicate)
            return None if vec is None else VSSValue(spaces, vec)
        if (
            len(spaces) == MATRIX
            and spaces[0] == self.word_space
            and spaces[1] == self.sentence_space
        ):
            mat = self._matrices.get(predicate)
            return None if mat is None else VSSValue(spaces, mat)
        if (
            len(spaces) == CUBE
            and spaces[0] == self.word_space
            and spaces[1] == self.sentence_space
            and spaces[2] == self.word_space
        ):
            cube = self._cubes.get(predicate)
            return None if cube is None else VSSValue(spaces, cube)
        return None

    def entries_of_type(self, spaces: tuple[VectorSpace, ...]) -> list[VSSValue]:
        """All registered tensors of the requested tensor type.

        For ``W ⊗ S`` requests this includes the *phrase tensors* of
        transitive verbs — each cube pre-contracted with every entity vector
        (``T^{control baby}``, ``T^{control milk}``, … in the paper's
        notation) — which is what the ``T+`` requirement interpretation sums
        over.
        """
        out: list[VSSValue] = []
        for (pred, names), value in self._generic.items():
            if names == tuple(s.name for s in spaces):
                out.append(value)
        if len(spaces) == 1 and spaces[0] == self.word_space:
            out.extend(VSSValue(spaces, v) for v in self._entities.values())
        elif (
            len(spaces) == MATRIX
            and spaces[0] == self.word_space
            and spaces[1] == self.sentence_space
        ):
            out.extend(VSSValue(spaces, m) for m in self._matrices.values())
            for cube in self._cubes.values():
                for vec in self._entities.values():
                    phrase = np.einsum("ijk,k->ij", cube, vec)
                    out.append(VSSValue(spaces, phrase))
        elif (
            len(spaces) == CUBE
            and spaces[0] == self.word_space
            and spaces[1] == self.sentence_space
            and spaces[2] == self.word_space
        ):
            out.extend(VSSValue(spaces, c) for c in self._cubes.values())
        return out

    # ------------------------------------------------------------------
    # distributional construction
    # ------------------------------------------------------------------
    @classmethod
    def from_cooccurrence(
        cls,
        targets: Sequence[str],
        contexts: Sequence[str],
        counts: np.ndarray | Sequence[Sequence[float]],
        *,
        weighting: str = "ppmi",
        sentence_space: VectorSpace | None = None,
    ) -> "VSSLexicon":
        """Build a lexicon whose entity vectors come from a co-occurrence matrix.

        :param targets: row labels — the entity predicates.
        :param contexts: column labels — the basis words of ``W``.
        :param counts: raw co-occurrence counts, shape ``(len(targets),
            len(contexts))``.
        :param weighting: ``"ppmi"`` (default), ``"pmi"`` or ``"count"``.
        """
        counts = np.asarray(counts, dtype=float)
        if counts.shape != (len(targets), len(contexts)):
            raise ValueError(
                f"counts shape {counts.shape} != ({len(targets)}, {len(contexts)})"
            )
        lex = cls(
            word_space=VectorSpace("W", len(contexts), tuple(contexts)),
            sentence_space=sentence_space or plausibility_space(),
        )
        if weighting == "ppmi":
            mat = ppm(counts, positive=True)
        elif weighting == "pmi":
            mat = ppm(counts, positive=False)
        elif weighting == "count":
            mat = counts
        else:
            raise ValueError(f"unknown weighting {weighting!r}")
        for i, target in enumerate(targets):
            lex.add_entity(target, mat[i])
        return lex

    def plausibility_matrix(
        self,
        with_counts: Mapping[str, float] | Sequence[float],
        without_counts: Mapping[str, float] | Sequence[float] | None = None,
    ) -> np.ndarray:
        """Assemble a ``W ⊗ S`` plausibility matrix from per-basis counts.

        ``with_counts[i]`` counts contexts where the verb co-occurs with
        basis word ``W_i`` (plausibility, ``⊤`` column); ``without_counts[i]``
        counts contexts where the verb occurs without ``W_i``
        (implausibility, ``⊥`` column) — the approximation of Sadrzadeh et
        al. (2018), section 4.  Omitted ``without_counts`` default to zero.
        """

        def dense(x: Mapping[str, float] | Sequence[float] | None) -> np.ndarray:
            if x is None:
                return np.zeros(self.word_space.dim)
            if isinstance(x, Mapping):
                return np.array(
                    [float(x.get(b, 0.0)) for b in self.word_space.basis], dtype=float
                )
            arr = np.asarray(x, dtype=float)
            if arr.shape != (self.word_space.dim,):
                raise ValueError(f"expected {self.word_space.dim} counts, got {arr.shape}")
            return arr

        top = dense(with_counts)
        bottom = dense(without_counts)
        return np.stack([top, bottom], axis=1)

    def learn_plausibility_verbs(
        self,
        contexts: Iterable[tuple[Iterable[str], str]],
        *,
        transitive: Iterable[str] = (),
    ) -> None:
        """Learn verb tensors from (context-words, verb) pairs.

        :param contexts: iterable of ``(context_words, verb)`` where
            ``context_words`` are the other words of the excerpt.
        :param transitive: predicates to register as cubes; all others are
            registered as matrices.  Cube entries accumulate the same counts
            uniformly across the object axis (a neutral object prior) —
            refine them afterwards with :meth:`add_transitive` if object
            information is available.
        """
        transitive = set(transitive)
        with_c: dict[str, np.ndarray] = {}
        without_c: dict[str, np.ndarray] = {}
        basis = list(self.word_space.basis)
        for words, verb in contexts:
            words = set(words)
            hit = np.array([1.0 if b in words else 0.0 for b in basis])
            w = with_c.setdefault(verb, np.zeros(self.word_space.dim))
            wo = without_c.setdefault(verb, np.zeros(self.word_space.dim))
            w += hit
            wo += 1.0 - hit
        for verb in with_c:
            mat = self.plausibility_matrix(with_c[verb], without_c[verb])
            if verb in transitive:
                cube = np.repeat(mat[:, :, None], self.word_space.dim, axis=2)
                self.add_transitive(verb, cube)
            else:
                self.add_intransitive(verb, mat)
