"""Vector spaces and tensor values for DS-VSS.

This module implements the linear-algebraic substrate of *Dynamic Syntax with
Vector Space Semantics* (DS-VSS), following:

    Sadrzadeh, M., Purver, M., Hough, J. & Kempson, R. (2018).
    "Exploring Semantic Incrementality with Dynamic Syntax and Vector Space
    Semantics." arXiv:1811.00614 [cs.CL].

DS types are mapped to (tensor products of) vector spaces:

- the entity type ``Ty(e)`` maps to a *word space* ``W``;
- the proposition type ``Ty(t)`` maps to a *sentence space* ``S``;
- function types map to higher-order tensors (see :mod:`dylan.vss.typemap`).

DS function application (its sole composition operation, ``O``) is mapped to
*tensor contraction*; LINK-tree conjunction is mapped to the Frobenius
``mu`` map (pointwise product), as in the paper's relative-clause analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

#: Canonical basis labels of the two-dimensional plausibility sentence space
#: of Sadrzadeh et al. (2018), section 4 ("true" / "false").
TRUE = "⊤"
FALSE = "⊥"


@dataclass(frozen=True)
class VectorSpace:
    """A finite-dimensional real vector space with a named, labelled basis."""

    name: str
    dim: int
    basis: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"VectorSpace {self.name!r}: dim must be positive, got {self.dim}")
        if self.basis and len(self.basis) != self.dim:
            raise ValueError(
                f"VectorSpace {self.name!r}: basis has {len(self.basis)} labels but dim is {self.dim}"
            )
        if not self.basis:
            object.__setattr__(self, "basis", tuple(f"{self.name}{i}" for i in range(self.dim)))

    def basis_index(self, label: str) -> int:
        """Return the index of basis element *label*."""
        return self.basis.index(label)

    def __str__(self) -> str:
        return self.name


def plausibility_space() -> VectorSpace:
    """Return the two-dimensional sentence space ``S`` with basis ``(⊤, ⊥)``."""
    return VectorSpace("S", 2, (TRUE, FALSE))


@dataclass
class VSSValue:
    """A tensor value living in a tensor product of :class:`VectorSpace` s.

    ``spaces[i]`` is the space of axis ``i`` of ``array``; the array's shape
    must equal ``tuple(s.dim for s in spaces)``.  An order-1 value is a
    vector, order-2 a matrix, order-3 a cube, and a order-0 value a scalar.
    """

    spaces: tuple[VectorSpace, ...]
    array: np.ndarray

    def __post_init__(self) -> None:
        self.array = np.asarray(self.array, dtype=float)
        expected = tuple(s.dim for s in self.spaces)
        if self.array.shape != expected:
            raise ValueError(
                f"VSSValue: array shape {self.array.shape} does not match spaces "
                f"{tuple(s.name for s in self.spaces)} with dims {expected}"
            )

    @property
    def order(self) -> int:
        """Tensor order (number of indices)."""
        return len(self.spaces)

    def space_names(self) -> tuple[str, ...]:
        """Names of the factor spaces, in axis order."""
        return tuple(s.name for s in self.spaces)

    def __repr__(self) -> str:
        names = " ⊗ ".join(self.space_names()) or "R"
        return f"VSSValue({names}, {self.array!r})"


def unit_value(spaces: Sequence[VectorSpace]) -> VSSValue:
    """The all-ones tensor of a tensor-product space (neutral for composition).

    This is the *unit* interpretation of DS requirements ``?Ty(X)`` of
    Sadrzadeh et al. (2018), section 3: an element "neutral with regards to
    composition".
    """
    spaces = tuple(spaces)
    return VSSValue(spaces, np.ones(tuple(s.dim for s in spaces), dtype=float))


def contract(functor: VSSValue, argument: VSSValue) -> VSSValue:
    """Contract *functor* with *argument* over their shared space.

    The argument must be a vector (order 1).  The contraction index is the
    **rightmost** functor axis whose space equals the argument's space; this
    reproduces the subject/object contraction order of the paper, where a
    transitive-verb cube ``T_ijk`` first contracts with the object vector
    (index ``k``) and only then with the subject (index ``i``)::

        T_i^subj · T_ijk^verb · T_k^obj  →  S-vector

    :raises ValueError: if the argument is not a vector or its space does
        not occur in the functor.
    """
    if argument.order != 1:
        raise ValueError(
            f"DS-VSS contraction expects an order-1 argument, got order {argument.order} "
            f"with spaces {argument.space_names()}"
        )
    arg_space = argument.spaces[0]
    axis = None
    for i in range(functor.order - 1, -1, -1):
        if functor.spaces[i] == arg_space:
            axis = i
            break
    if axis is None:
        raise ValueError(
            f"cannot contract argument in space {arg_space.name!r} with functor over "
            f"{functor.space_names()}: shared space not found"
        )
    out_spaces = functor.spaces[:axis] + functor.spaces[axis + 1 :]
    out = np.tensordot(functor.array, argument.array, axes=([axis], [0]))
    return VSSValue(out_spaces, out)


def mu(a: VSSValue, b: VSSValue) -> VSSValue:
    """Frobenius ``mu`` map: pointwise (Hadamard) product of two same-space values.

    Used to combine a matrix tree with its LINKed tree (relative clauses,
    apposition, tense LINKs), following Sadrzadeh et al. (2018), section 3.
    """
    if a.space_names() != b.space_names():
        raise ValueError(
            f"mu requires matching spaces, got {a.space_names()} and {b.space_names()}"
        )
    return VSSValue(a.spaces, a.array * b.array)


def plausibility(value: VSSValue) -> float:
    """Normalised plausibility of a sentence-space vector: ``⊤ / (⊤ + ⊥)``.

    A scalar value is returned as-is.  For general sentence spaces (dim > 2 or
    an unlabelled basis) the *share* of the first basis element is returned.
    """
    if value.order == 0:
        return float(value.array)
    if value.order != 1:
        raise ValueError(f"plausibility expects a sentence vector, got order {value.order}")
    top = float(value.array[0])
    if value.array.shape[0] < 2:
        return top
    rest = float(np.sum(value.array[1:]))
    denom = top + rest
    return top / denom if denom else 0.0


def direct_sum(values: Sequence[VSSValue]) -> "VSSDirectSum":
    """The *direct sum* interpretation of DS requirements (a tuple of options)."""
    return VSSDirectSum(tuple(values))


@dataclass
class VSSDirectSum:
    """Tuple of alternative tensors: the ``T⊕`` requirement interpretation.

    Keeps the possible developments of a requirement node separate rather
    than accumulated (as in the ``T+`` sum), enumerating the possibilities;
    see Sadrzadeh et al. (2018), section 3.
    """

    values: tuple[VSSValue, ...]

    def map_contract(self, argument: VSSValue) -> "VSSDirectSum":
        """Distribute contraction with *argument* over the alternatives."""
        return VSSDirectSum(tuple(contract(v, argument) for v in self.values))

    def map_mu(self, other: VSSValue) -> "VSSDirectSum":
        """Distribute the Frobenius ``mu`` map with *other* over the alternatives."""
        return VSSDirectSum(tuple(mu(v, other) for v in self.values))

    def plausibilities(self) -> list[float]:
        """Plausibility of each alternative (for sentence-space values)."""
        return [plausibility(v) for v in self.values]


def einsum_contraction_spec(spaces: Sequence[VectorSpace]) -> str:
    """Human-readable index specification of a tensor type (for docs/logging)."""
    names = [s.name for s in spaces]
    counts: dict[str, int] = {}
    labels = []
    for n in names:
        counts[n] = counts.get(n, 0) + 1
        labels.append(n if counts[n] == 1 else f"{n}^{counts[n]}")
    return " ⊗ ".join(labels) or "R"
