"""Mapping from DS types to tensor types (tensor products of vector spaces).

Follows Sadrzadeh et al. (2018), section 3:

- ``Ty(e)`` (and the nominal types ``cn``, ``cnev``) map to the word space ``W``;
- ``Ty(t)`` (and the eventuality type ``es``) map to the sentence space ``S``;
- one-place predicates ``Ty(⟨e,t⟩)`` map to matrices in ``W ⊗ S``;
- two-place predicates ``Ty(⟨e,⟨e,t⟩⟩)`` map to cubes in ``W ⊗ S ⊗ W``, with
  indices ordered *(subject, sentence, object)* — the paper's ``T_ijk^like``,
  so that contraction with the rightmost ``W`` index consumes the object
  first and contraction with the remaining ``W`` index consumes the subject.

Other function types ``A>B`` fall back to the concatenation
``type(A) ++ type(B)`` (input indices first, then output indices), which
agrees with the special case above for one-place predicates.
"""

from __future__ import annotations

from dylan.type.dstype import BasicType, ConstructedType, DSType
from dylan.vss.spaces import VectorSpace

#: DS basic types that denote entities (mapped to the word space ``W``).
ENTITY_TYPES = ("e", "cn", "cnev")

#: DS basic types that denote propositions/eventualities (mapped to ``S``).
SENTENCE_TYPES = ("t", "es")


class TensorTypeMap:
    """Map :class:`~dylan.type.dstype.DSType` s to tuples of factor spaces.

    :param word_space: the entity/word space ``W``.
    :param sentence_space: the proposition/sentence space ``S``.
    """

    def __init__(self, word_space: VectorSpace, sentence_space: VectorSpace) -> None:
        self.word_space = word_space
        self.sentence_space = sentence_space

    def _basic(self, t: BasicType) -> tuple[VectorSpace, ...]:
        if t.name in ENTITY_TYPES:
            return (self.word_space,)
        if t.name in SENTENCE_TYPES:
            return (self.sentence_space,)
        raise ValueError(f"no vector-space mapping for DS basic type {t.name!r}")

    def __call__(self, ds_type: DSType) -> tuple[VectorSpace, ...]:
        """Return the factor spaces of the tensor denoting *ds_type*."""
        if isinstance(ds_type, BasicType):
            return self._basic(ds_type)
        if not isinstance(ds_type, ConstructedType):
            instantiated = ds_type.instantiate()
            if instantiated is ds_type:
                raise ValueError(f"no vector-space mapping for DS type {ds_type!r}")
            return self(instantiated)

        # Collect the argument types (outermost first) and the final result type.
        args: list[DSType] = []
        result: DSType = ds_type
        while isinstance(result, ConstructedType):
            args.append(result.from_type)
            result = result.to_type

        verb_like = (
            isinstance(result, BasicType)
            and result.name in SENTENCE_TYPES
            and all(isinstance(a, BasicType) and a.name in ENTITY_TYPES for a in args)
        )
        if verb_like:
            # (subject, S, object, …): subject index first, sentence index
            # second, then the remaining arguments in parse (contraction)
            # order — the rightmost matching axis is consumed first.
            spaces = [self._basic(args[0])[0], self.sentence_space]
            spaces.extend(self._basic(a)[0] for a in args[1:])
            return tuple(spaces)
        # Generic fallback: input indices first, then output indices.
        spaces = []
        for a in args:
            spaces.extend(self(a))
        spaces.extend(self(result))
        return tuple(spaces)
